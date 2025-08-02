import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def qrt_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official QRT Challenge scoring function
    
    Args:
        y_true: true returns (can be positive or negative)
        y_pred: predicted signs (-1 or 1)
    
    Returns:
        score: weighted accuracy based on absolute returns
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Convert predictions to signs if needed
    if y_pred.dtype != np.int64 and y_pred.dtype != np.int32:
        y_pred = np.sign(y_pred)
    
    # Calculate weighted accuracy
    numerator = np.sum(np.abs(y_true) * (np.sign(y_true) == y_pred))
    denominator = np.sum(np.abs(y_true))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: true binary labels (0 or 1)
        y_pred: predicted binary labels (0 or 1)
        sample_weights: optional sample weights
    
    Returns:
        metrics: dictionary of evaluation metrics
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weights),
        'precision': precision_score(y_true, y_pred, sample_weight=sample_weights, zero_division=0),
        'recall': recall_score(y_true, y_pred, sample_weight=sample_weights, zero_division=0),
        'f1': f1_score(y_true, y_pred, sample_weight=sample_weights, zero_division=0)
    }
    
    # Calculate per-class accuracy
    for class_label in [0, 1]:
        mask = (y_true == class_label)
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            metrics[f'accuracy_class_{class_label}'] = class_acc
    
    return metrics


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                  device: torch.device, return_predictions: bool = False) -> Dict[str, float]:
    """
    Evaluate model on a data loader
    
    Args:
        model: PyTorch model
        data_loader: data loader for evaluation
        device: device to run evaluation on
        return_predictions: whether to return predictions
    
    Returns:
        metrics: evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_weights = []
    all_logits = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 兼容DataLoader默认collate_fn输出tuple/list
            if isinstance(batch, (list, tuple)) and len(batch) >= 5:
                X, industry_ids, target_idx, target_sign, sample_weight = batch[:5]
                y = (target_sign > 0).float().to(device)
                X = X.to(device)
                industry_ids = industry_ids.to(device)
                target_idx = target_idx.to(device)
                y = y.to(device)
                sample_weight = sample_weight.to(device)
                logits = model(X, industry_ids, target_idx)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                all_logits.append(logits.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_weights.append(sample_weight.cpu().numpy())
            else:
                # Test mode or unexpected batch
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = X.to(device)
                logits = model(X, torch.zeros(X.size(0), dtype=torch.long, device=X.device))
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                all_logits.append(logits.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate results
    all_predictions = np.concatenate(all_predictions)
    all_logits = np.concatenate(all_logits)
    
    if all_targets:
        all_targets = np.concatenate(all_targets)
        all_weights = np.concatenate(all_weights)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions, all_weights)
        
        # Calculate QRT score (convert binary to signs)
        y_true_signs = 2 * all_targets - 1  # Convert 0/1 to -1/1
        y_pred_signs = 2 * all_predictions - 1
        metrics['qrt_score'] = qrt_score(y_true_signs, y_pred_signs)
        
        if return_predictions:
            return metrics, all_predictions, all_logits, all_targets, all_weights
        else:
            return metrics
    else:
        # Test mode - no targets
        if return_predictions:
            return {}, all_predictions, all_logits, None, None
        else:
            return {}


def evaluate_by_asset(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                     device: torch.device) -> pd.DataFrame:
    """
    Evaluate model performance for each asset separately
    
    Args:
        model: PyTorch model
        data_loader: data loader for evaluation
        device: device to run evaluation on
    
    Returns:
        asset_metrics: DataFrame with metrics per asset
    """
    model.eval()
    
    asset_results = {}
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) >= 5:
                X, industry_ids, target_idx, target_sign, sample_weight = batch[:5]
                # Convert target_sign to binary labels (0 or 1)
                y = (target_sign > 0).float().to(device)
            else:
                continue  # Skip if no targets
            
            X = X.to(device)
            industry_ids = industry_ids.to(device)
            target_idx = target_idx.to(device)
            sample_weight = sample_weight.to(device)
            
            # Forward pass
            logits = model(X, industry_ids, target_idx)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            # Group by asset ID
            for i in range(len(target_idx)):
                asset_id = target_idx[i].item()
                
                if asset_id not in asset_results:
                    asset_results[asset_id] = {
                        'predictions': [],
                        'targets': [],
                        'weights': [],
                        'logits': []
                    }
                
                asset_results[asset_id]['predictions'].append(predictions[i].item())
                asset_results[asset_id]['targets'].append(y[i].item())
                asset_results[asset_id]['weights'].append(sample_weight[i].item())
                asset_results[asset_id]['logits'].append(logits[i].item())
    
    # Calculate metrics per asset
    asset_metrics = []
    for asset_id, results in asset_results.items():
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        weights = np.array(results['weights'])
        logits = np.array(results['logits'])
        
        # Calculate metrics
        metrics = calculate_metrics(targets, predictions, weights)
        
        # Calculate QRT score
        y_true_signs = 2 * targets - 1
        y_pred_signs = 2 * predictions - 1
        metrics['qrt_score'] = qrt_score(y_true_signs, y_pred_signs)
        
        # Add asset info
        metrics['asset_id'] = asset_id
        metrics['num_samples'] = len(targets)
        metrics['avg_weight'] = np.mean(weights)
        
        asset_metrics.append(metrics)
    
    return pd.DataFrame(asset_metrics)


def plot_asset_performance(asset_metrics: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot performance metrics by asset
    
    Args:
        asset_metrics: DataFrame with metrics per asset
        save_path: optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # QRT Score by asset
    axes[0, 0].bar(asset_metrics['asset_id'], asset_metrics['qrt_score'])
    axes[0, 0].set_title('QRT Score by Asset')
    axes[0, 0].set_xlabel('Asset ID')
    axes[0, 0].set_ylabel('QRT Score')
    
    # Accuracy by asset
    axes[0, 1].bar(asset_metrics['asset_id'], asset_metrics['accuracy'])
    axes[0, 1].set_title('Accuracy by Asset')
    axes[0, 1].set_xlabel('Asset ID')
    axes[0, 1].set_ylabel('Accuracy')
    
    # Number of samples by asset
    axes[1, 0].bar(asset_metrics['asset_id'], asset_metrics['num_samples'])
    axes[1, 0].set_title('Number of Samples by Asset')
    axes[1, 0].set_xlabel('Asset ID')
    axes[1, 0].set_ylabel('Number of Samples')
    
    # Average weight by asset
    axes[1, 1].bar(asset_metrics['asset_id'], asset_metrics['avg_weight'])
    axes[1, 1].set_title('Average Sample Weight by Asset')
    axes[1, 1].set_xlabel('Asset ID')
    axes[1, 1].set_ylabel('Average Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        save_path: optional path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing evaluation functions...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # True returns (can be positive or negative)
    true_returns = np.random.normal(0, 1, n_samples)
    
    # True signs
    true_signs = np.sign(true_returns)
    
    # Predicted signs (with some noise)
    pred_signs = true_signs * np.random.choice([1, -1], n_samples, p=[0.8, 0.2])
    
    # Calculate QRT score
    score = qrt_score(true_returns, pred_signs)
    print(f"QRT Score: {score:.4f}")
    
    # Test binary metrics
    true_binary = (true_signs > 0).astype(int)
    pred_binary = (pred_signs > 0).astype(int)
    
    metrics = calculate_metrics(true_binary, pred_binary)
    print(f"Binary metrics: {metrics}") 