"""
Configuration file for QRT Challenge
"""

import torch
from typing import Dict, Any


class Config:
    """
    Configuration class for QRT Challenge
    """
    
    def __init__(self):
        # Data paths
        self.train_input_file = '../X_train_itDkypA.csv'  # Original files (for reference)
        self.train_output_file = '../y_train_3LeeT2g.csv'
        self.test_input_file = '../X_test_Beg4ey3.csv'
        
        # Processed data paths
        self.train_processed_file = 'train_processed.csv'
        self.test_processed_file = 'test_processed.csv'
        self.train_data_path = 'train_processed.csv'  # For compatibility
        self.test_data_path = 'test_processed.csv'    # For compatibility
        self.target_mapping_file = 'target_mapping.pkl'
        self.imputer_file = 'imputer.pkl'
        self.scaler_file = 'scaler.pkl'
        
        self.assets_file = 'assets.csv'  # Optional: for asset information
        
        # Model paths
        self.model_save_path = 'best_model.pth'
        self.model_load_path = 'best_model.pth'
        self.submission_file = 'submit.csv'
        
        # Output paths
        self.history_plot_path = 'training_history.png'
        self.analysis_plot_path = 'prediction_analysis.png'
        self.analysis_file = 'prediction_analysis.csv'
        
        # Model architecture
        self.model_type = 'multihead'  # 'multihead' or 'multitask'
        self.input_dim = None  # Will be set automatically from data
        self.hidden_dims = [256, 128, 64]
        self.num_heads = 100  # Number of liquid assets (0-99)
        self.dropout = 0.2
        self.activation = 'relu'  # 'relu', 'leaky_relu', 'tanh'
        
        # Training parameters
        self.batch_size = 512
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.early_stopping_patience = 15
        self.train_split = 0.8  # Fraction of data for training
        self.random_seed = 42
        
        # Loss function
        self.loss_reduction = 'mean'  # 'mean' or 'sum'
        
        # Optimizer
        self.optimizer_type = 'AdamW'  # 'Adam', 'AdamW', 'SGD'
        self.scheduler_type = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau', 'CosineAnnealingLR'
        self.scheduler_patience = 5
        self.scheduler_factor = 0.5
        self.scheduler_min_lr = 1e-6
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inference
        self.inference_batch_size = 256
        self.save_analysis = True
        
        # Ensemble (if using multiple models)
        self.use_ensemble = False
        self.ensemble_model_paths = []
        self.ensemble_weights = []
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'activation': self.activation
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration dictionary"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'train_split': self.train_split,
            'random_seed': self.random_seed,
            'loss_reduction': self.loss_reduction,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_min_lr': self.scheduler_min_lr
        }
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration dictionary"""
        return {
            'batch_size': self.inference_batch_size,
            'save_analysis': self.save_analysis,
            'use_ensemble': self.use_ensemble,
            'ensemble_model_paths': self.ensemble_model_paths,
            'ensemble_weights': self.ensemble_weights
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=== QRT Challenge Configuration ===")
        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Hidden dimensions: {self.hidden_dims}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Random seed: {self.random_seed}")
        print("===================================")


# Predefined configurations for different experiments
class ConfigPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def baseline() -> Config:
        """Baseline configuration"""
        config = Config()
        return config
    
    @staticmethod
    def large_model() -> Config:
        """Large model configuration"""
        config = Config()
        config.hidden_dims = [512, 256, 128, 64]
        config.dropout = 0.3
        config.learning_rate = 5e-4
        config.batch_size = 128
        return config
    
    @staticmethod
    def small_model() -> Config:
        """Small model configuration"""
        config = Config()
        config.hidden_dims = [128, 64]
        config.dropout = 0.1
        config.learning_rate = 2e-3
        config.batch_size = 512
        return config
    
    @staticmethod
    def multitask() -> Config:
        """Multi-task model configuration"""
        config = Config()
        config.model_type = 'multitask'
        config.hidden_dims = [256, 128]
        config.dropout = 0.2
        return config
    
    @staticmethod
    def high_lr() -> Config:
        """High learning rate configuration"""
        config = Config()
        config.learning_rate = 5e-3
        config.scheduler_patience = 3
        config.early_stopping_patience = 10
        return config
    
    @staticmethod
    def low_lr() -> Config:
        """Low learning rate configuration"""
        config = Config()
        config.learning_rate = 1e-4
        config.num_epochs = 200
        config.early_stopping_patience = 25
        return config


def create_config(preset: str = 'baseline') -> Config:
    """
    Create configuration from preset
    
    Args:
        preset: configuration preset name
    
    Returns:
        config: configuration object
    """
    if preset == 'baseline':
        return ConfigPresets.baseline()
    elif preset == 'large':
        return ConfigPresets.large_model()
    elif preset == 'small':
        return ConfigPresets.small_model()
    elif preset == 'multitask':
        return ConfigPresets.multitask()
    elif preset == 'high_lr':
        return ConfigPresets.high_lr()
    elif preset == 'low_lr':
        return ConfigPresets.low_lr()
    else:
        print(f"Unknown preset: {preset}. Using baseline configuration.")
        return ConfigPresets.baseline()


if __name__ == "__main__":
    # Test configuration
    config = create_config('baseline')
    config.print_config()
    
    print("\nModel config:")
    print(config.get_model_config())
    
    print("\nTraining config:")
    print(config.get_training_config()) 