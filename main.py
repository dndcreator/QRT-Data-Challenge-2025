#!/usr/bin/env python3
"""
Main script for QRT Challenge

Usage:
    python main.py train [--config CONFIG] [--preset PRESET]
    python main.py predict [--config CONFIG] [--model MODEL]
    python main.py evaluate [--config CONFIG] [--model MODEL]
    python main.py test [--config CONFIG]
"""

import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd

from config import create_config, Config
from processed_dataset import create_processed_data_loaders, ProcessedQRTDataset, load_target_mapping
from model import create_model
from train import Trainer
from predict import load_trained_model, predict_on_test_data, create_submission_file, analyze_predictions
from evaluate import evaluate_model, evaluate_by_asset, plot_asset_performance


def train_model(config: Config):
    """Train the model"""
    print("=== Starting Training ===")
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Create data loaders using processed data
    print("Creating data loaders...")
    train_loader, val_loader, test_dataset = create_processed_data_loaders(
        train_file=config.train_processed_file,
        test_file=config.test_processed_file,
        batch_size=config.batch_size,
        train_split=config.train_split,
        random_seed=config.random_seed
    )
    
    # Get input dimension from dataset if not set
    if config.input_dim is None:
        # Load a small sample to get input dimension
        temp_dataset = ProcessedQRTDataset(config.train_processed_file, is_training=True)
        config.input_dim = temp_dataset.input_dim
        print(f"Auto-detected input dimension: {config.input_dim}")
    
    # Create model
    print("Creating model...")
    model = create_model(**config.get_model_config())
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=config.device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
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
    trainer.plot_training_history(save_path=config.history_plot_path)
    
    # Final evaluation
    print("Final evaluation...")
    final_metrics = evaluate_model(model, val_loader, config.device)
    print(f"Final validation metrics: {final_metrics}")
    
    # Asset-wise evaluation
    print("Asset-wise evaluation...")
    asset_metrics = evaluate_by_asset(model, val_loader, config.device)
    asset_metrics.to_csv('asset_metrics.csv', index=False)
    print(f"Asset metrics saved to asset_metrics.csv")
    
    # Plot asset performance
    try:
        plot_asset_performance(asset_metrics, save_path='asset_performance.png')
    except Exception as e:
        print(f"Could not create asset performance plot: {e}")
    
    print("Training completed!")


def predict_model(config: Config):
    """Generate predictions"""
    print("=== Starting Prediction ===")
    
    # Check if model file exists
    if not os.path.exists(config.model_load_path):
        print(f"Model file {config.model_load_path} not found!")
        print("Please train a model first using: python main.py train")
        return
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ProcessedQRTDataset(config.test_processed_file, is_training=False)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model(config.model_load_path, config.get_model_config(), config.device)
    
    # Generate predictions
    predictions_df = predict_on_test_data(
        model, test_dataset, config.device, config.inference_batch_size
    )
    
    # Create submission file
    submission_df = create_submission_file(predictions_df, config.submission_file)
    
    # Analyze predictions
    analyze_predictions(predictions_df, save_analysis=config.save_analysis)
    
    print("Prediction completed!")


def evaluate_model_script(config: Config):
    """Evaluate trained model"""
    print("=== Starting Evaluation ===")
    
    # Check if model file exists
    if not os.path.exists(config.model_load_path):
        print(f"Model file {config.model_load_path} not found!")
        print("Please train a model first using: python main.py train")
        return
    
    # Create data loaders using processed data
    print("Creating data loaders...")
    train_loader, val_loader, test_dataset = create_processed_data_loaders(
        train_file=config.train_processed_file,
        test_file=config.test_processed_file,
        batch_size=config.batch_size,
        train_split=config.train_split,
        random_seed=config.random_seed
    )
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model(config.model_load_path, config.get_model_config(), config.device)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, config.device)
    print(f"Validation metrics: {val_metrics}")
    
    # Asset-wise evaluation
    print("Asset-wise evaluation...")
    asset_metrics = evaluate_by_asset(model, val_loader, config.device)
    asset_metrics.to_csv('asset_metrics.csv', index=False)
    print(f"Asset metrics saved to asset_metrics.csv")
    
    # Print summary statistics
    print("\n=== Asset Performance Summary ===")
    print(f"Average QRT Score: {asset_metrics['qrt_score'].mean():.4f}")
    print(f"Best Asset QRT Score: {asset_metrics['qrt_score'].max():.4f}")
    print(f"Worst Asset QRT Score: {asset_metrics['qrt_score'].min():.4f}")
    print(f"Average Accuracy: {asset_metrics['accuracy'].mean():.4f}")
    
    # Plot asset performance
    try:
        plot_asset_performance(asset_metrics, save_path='asset_performance.png')
    except Exception as e:
        print(f"Could not create asset performance plot: {e}")
    
    print("Evaluation completed!")


def test_components(config: Config):
    """Test individual components"""
    print("=== Testing Components ===")
    
    # Test dataset
    print("Testing dataset...")
    try:
        if os.path.exists(config.train_processed_file):
            dataset = ProcessedQRTDataset(config.train_processed_file, is_training=True)
            print(f"Dataset loaded successfully: {len(dataset)} samples")
            
            # Test sample
            sample = dataset[0]
            print(f"Sample format: {len(sample)} elements")
            if len(sample) == 4:
                X, target_idx, target_sign, weight = sample
                print(f"X shape: {X.shape}, target_idx: {target_idx}, target_sign: {target_sign}, weight: {weight}")
        else:
            print("Training files not found, skipping dataset test")
    except Exception as e:
        print(f"Dataset test failed: {e}")
    
    # Test model
    print("Testing model...")
    try:
        model = create_model(**config.get_model_config())
        print(f"Model created successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 32
        X = torch.randn(batch_size, config.input_dim)
        id_targets = torch.randint(0, config.num_heads, (batch_size,))
        
        logits = model(X, id_targets)
        print(f"Forward pass successful: logits shape {logits.shape}")
        
    except Exception as e:
        print(f"Model test failed: {e}")
    
    # Test evaluation
    print("Testing evaluation...")
    try:
        # Generate dummy data
        y_true = np.random.choice([0, 1], 1000)
        y_pred = np.random.choice([0, 1], 1000)
        
        from evaluate import qrt_score, calculate_metrics
        
        # Test QRT score
        score = qrt_score(y_true * 2 - 1, y_pred * 2 - 1)  # Convert to signs
        print(f"QRT score test: {score:.4f}")
        
        # Test metrics
        metrics = calculate_metrics(y_true, y_pred)
        print(f"Metrics test: {metrics}")
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
    
    print("Component testing completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='QRT Challenge')
    parser.add_argument('command', choices=['train', 'predict', 'evaluate', 'test'],
                       help='Command to run')
    parser.add_argument('--config', type=str, default='baseline',
                       help='Configuration preset (baseline, large, small, multitask, high_lr, low_lr)')
    parser.add_argument('--preset', type=str, default='baseline',
                       help='Configuration preset (alias for --config)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (for predict/evaluate)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config(args.config or args.preset)
    
    # Override model path if specified
    if args.model:
        config.model_load_path = args.model
    
    # Print configuration
    config.print_config()
    
    # Run command
    if args.command == 'train':
        train_model(config)
    elif args.command == 'predict':
        predict_model(config)
    elif args.command == 'evaluate':
        evaluate_model_script(config)
    elif args.command == 'test':
        test_components(config)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main() 