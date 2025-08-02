import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

from processed_dataset import ProcessedQRTDataset
from model import MultiHeadMLP, create_model
from evaluate import evaluate_model, qrt_score, calculate_metrics


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss with sample weights
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(WeightedBCELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                sample_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: model predictions [batch_size]
            targets: binary targets [batch_size]
            sample_weights: sample weights [batch_size]
        
        Returns:
            loss: weighted BCE loss
        """
        # Use BCEWithLogitsLoss for numerical stability
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        losses = bce_loss(logits, targets)
        
        # Apply sample weights
        weighted_losses = losses * sample_weights
        
        if self.reduction == 'mean':
            return weighted_losses.mean()
        elif self.reduction == 'sum':
            return weighted_losses.sum()
        else:
            return weighted_losses


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Args:
            score: current validation score (higher is better)
            model: PyTorch model
        
        Returns:
            should_stop: whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class Trainer:
    """
    Main training class for QRT Challenge
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.criterion = WeightedBCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.learning_rates = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: training data loader
        
        Returns:
            avg_loss: average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            X, industry_ids, target_idx, target_sign, sample_weight = batch[:5]
            
            X = X.to(self.device)
            industry_ids = industry_ids.to(self.device)
            target_idx = target_idx.to(self.device)
            # Convert target_sign to binary labels (0 or 1)
            y = (target_sign > 0).float().to(self.device)
            sample_weight = sample_weight.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X, industry_ids, target_idx)
            loss = self.criterion(logits, y, sample_weight)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            val_loader: validation data loader
        
        Returns:
            avg_loss: average validation loss
            qrt_score: QRT validation score
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_weights = []
        
        with torch.no_grad():
            for batch in val_loader:
                X, industry_ids, target_idx, target_sign, sample_weight = batch[:5]
                
                X = X.to(self.device)
                industry_ids = industry_ids.to(self.device)
                target_idx = target_idx.to(self.device)
                # Convert target_sign to binary labels (0 or 1)
                y = (target_sign > 0).float().to(self.device)
                sample_weight = sample_weight.to(self.device)
                
                # Forward pass
                logits = self.model(X, industry_ids, target_idx)
                loss = self.criterion(logits, y, sample_weight)
                
                # Get predictions
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_weights.append(sample_weight.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_weights = np.concatenate(all_weights)
        
        # Calculate QRT score
        y_true_signs = 2 * all_targets - 1
        y_pred_signs = 2 * all_predictions - 1
        score = qrt_score(y_true_signs, y_pred_signs)
        
        return avg_loss, score
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, early_stopping_patience: int = 15,
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            num_epochs: maximum number of epochs
            early_stopping_patience: patience for early stopping
            save_path: path to save best model
        
        Returns:
            history: training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_score = 0.0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_score = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_score)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_scores.append(val_score)
            self.learning_rates.append(current_lr)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s): "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Score: {val_score:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_score > best_score:
                best_score = val_score
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_score': val_score,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }, save_path)
                    print(f"Saved best model with score: {val_score:.4f}")
            
            # Early stopping
            if early_stopping(val_score, self.model):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"Training completed. Best validation score: {best_score:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores,
            'learning_rates': self.learning_rates
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation score
        axes[0, 1].plot(self.val_scores, label='Val Score', color='green')
        axes[0, 1].set_title('Validation QRT Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('QRT Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates, label='Learning Rate', color='red')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Loss vs Score
        axes[1, 1].scatter(self.val_losses, self.val_scores, alpha=0.6)
        axes[1, 1].set_title('Validation Loss vs Score')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('QRT Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def train_and_evaluate(config):
    """
    外部自动调参用：根据config训练并返回最佳验证分数
    """
    import torch
    import numpy as np
    from processed_dataset import ProcessedQRTDataset
    from model import create_model
    from torch.utils.data import DataLoader, random_split
    import os

    # Set random seeds
    torch.manual_seed(getattr(config, 'random_seed', 42))
    np.random.seed(getattr(config, 'random_seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training dataset
    train_dataset = ProcessedQRTDataset(config.train_data_path)
    train_size = int(getattr(config, 'train_split', 0.8) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(getattr(config, 'random_seed', 42))
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    # Create model
    model = create_model(
        model_type=config.model_type,
        input_dim=train_dataset.dataset.input_dim if hasattr(train_dataset, 'dataset') else train_dataset.input_dim,
        hidden_dims=config.hidden_dims,
        num_heads=100,
        dropout=config.dropout,
        activation=config.activation
    )
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience,
        save_path=None
    )
    # 返回最佳验证分数
    best_val_score = max(history['val_scores']) if history['val_scores'] else 0.0
    return best_val_score


def main():
    """
    Main training script
    """
    # Import required modules
    from config import Config
    from processed_dataset import ProcessedQRTDataset
    from model import create_model
    from evaluate import evaluate_model
    from torch.utils.data import DataLoader, random_split
    import os
    
    # Load configuration
    config = Config()
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    
    # Load training dataset
    train_dataset = ProcessedQRTDataset(config.train_data_path)
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Load test dataset
    test_dataset = ProcessedQRTDataset(config.test_data_path, is_training=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=config.model_type,
        input_dim=100,  # 100 RET features
        hidden_dims=config.hidden_dims,
        num_heads=100,  # 100 liquid assets
        dropout=config.dropout,
        activation=config.activation
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience,
        save_path=config.model_save_path
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='training_history.png')
    
    # Final evaluation
    print("Final evaluation...")
    final_metrics = evaluate_model(model, val_loader, device)
    print(f"Final validation metrics: {final_metrics}")
    
    print("Training completed!")


if __name__ == "__main__":
    main() 