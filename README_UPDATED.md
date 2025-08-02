# QRT Challenge - Updated Quant Trading Model

This repository contains a complete, updated framework for the QRT (Qube Research & Technologies) quant challenge. The project has been optimized to use preprocessed data for better performance and consistency.

## 🎯 Challenge Overview

- **Input**: 100 illiquid asset returns (RET_0 to RET_99)
- **Output**: Direction prediction for 100 liquid assets (ID_TARGET 0-99)
- **Task**: Multi-output binary classification
- **Metric**: Weighted accuracy based on absolute returns

## 📁 Updated Project Structure

```
model/
├── processed_dataset.py      # NEW: Processed data loader
├── prepare_data.py          # NEW: Data preprocessing pipeline
├── check_consistency.py     # NEW: Project consistency checker
├── model.py                 # Multi-head MLP model architecture
├── train.py                 # Training pipeline with weighted BCE loss
├── evaluate.py              # Evaluation metrics and QRT scoring
├── predict.py               # Prediction and submission generation
├── config.py                # Updated configuration management
├── main.py                  # Updated main script with CLI interface
├── requirements.txt         # Python dependencies
├── README.md               # Original README
└── README_UPDATED.md       # This updated README
```

## 🚀 Quick Start (Updated)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data (NEW)

First, run the data preparation script to process the raw data:

```bash
python prepare_data.py
```

This will generate:
- `train_processed.csv` - Processed training data
- `test_processed.csv` - Processed test data
- `target_mapping.pkl` - Target variable mapping
- `imputer.pkl` - Missing value imputer
- `scaler.pkl` - Feature scaler

### 3. Check Project Consistency (NEW)

Verify that all files are consistent:

```bash
python check_consistency.py
```

### 4. Train Model

```bash
# Train with baseline configuration
python main.py train

# Train with different configuration
python main.py train --config large
python main.py train --config small
python main.py train --config multitask
```

### 5. Generate Predictions

```bash
# Generate predictions and submission file
python main.py predict
```

### 6. Evaluate Model

```bash
# Evaluate trained model on validation set
python main.py evaluate
```

### 7. Test Components

```bash
# Test individual components
python main.py test
```

## 🔄 Data Processing Pipeline

### Original Data Files
- `X_train_itDkypA.csv` - Training input data (267,100 samples)
- `y_train_3LeeT2g.csv` - Training labels (267,100 samples)
- `X_test_Beg4ey3.csv` - Test input data (114,468 samples)

### Processing Steps
1. **Data Loading** - Load all data files
2. **Target Mapping** - Map ID_TARGET (1-292) to indices (0-99)
3. **Data Merging** - Merge X_train and y_train
4. **Missing Value Handling** - Fill missing values with mean
5. **Feature Standardization** - Standardize RET features
6. **Data Saving** - Save processed data and preprocessors

### Processed Data Features
- **Standardized Features** - All RET features are standardized (mean=0, std=1)
- **No Missing Values** - All missing values are filled
- **Consistent Target Mapping** - Targets mapped to 0-99 range
- **Sample Weights** - Based on absolute RET_TARGET values

## 📊 Updated Model Architecture

### MultiHeadMLP
- **Input**: 100 standardized RET features
- **Hidden Layers**: [256, 128, 64] (configurable)
- **Output**: 100 binary classifiers (one per asset)
- **Loss**: Weighted BCE Loss with sample weights

### Key Improvements
- **Processed Data**: Uses preprocessed, standardized data
- **Consistent Preprocessing**: Same preprocessing applied to train/test
- **Better Performance**: Faster training with preprocessed data
- **Reproducibility**: Consistent preprocessing parameters saved

## 🎛️ Updated Configuration

### New Configuration Options
```python
# Processed data paths
config.train_processed_file = 'train_processed.csv'
config.test_processed_file = 'test_processed.csv'
config.target_mapping_file = 'target_mapping.pkl'
config.imputer_file = 'imputer.pkl'
config.scaler_file = 'scaler.pkl'
```

### Available Presets
- `baseline`: Default configuration
- `large`: Larger model with more parameters
- `small`: Smaller model for faster training
- `multitask`: Multi-task learning approach
- `high_lr`: High learning rate configuration
- `low_lr`: Low learning rate configuration

## 📈 Training Features

### Loss Function
- **Weighted BCE Loss**: Uses `abs(RET_TARGET)` as sample weights
- **Importance weighting**: High volatility samples have higher weight

### Training Pipeline
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Validation Monitoring**: QRT score tracking
- **Model Checkpointing**: Saves best model

### Evaluation Metrics
- **QRT Score**: Official weighted accuracy metric
- **Standard Metrics**: Accuracy, Precision, Recall, F1
- **Asset-wise Analysis**: Performance per liquid asset

## 📤 Output Files

### Training Outputs
- `best_model.pth`: Trained model checkpoint
- `training_history.png`: Training curves
- `asset_metrics.csv`: Per-asset performance metrics
- `asset_performance.png`: Asset performance visualization

### Prediction Outputs
- `submit.csv`: Submission file (ID, RET_TARGET)
- `prediction_analysis.csv`: Detailed prediction analysis
- `prediction_analysis.png`: Prediction visualization

## 🔧 Advanced Usage

### Using Processed Data Directly
```python
from processed_dataset import ProcessedQRTDataset, create_processed_data_loaders

# Load processed training data
train_dataset = ProcessedQRTDataset('train_processed.csv', is_training=True)

# Load processed test data
test_dataset = ProcessedQRTDataset('test_processed.csv', is_training=False)

# Create data loaders
train_loader, val_loader, test_dataset = create_processed_data_loaders(
    train_file='train_processed.csv',
    test_file='test_processed.csv',
    batch_size=256
)
```

### Loading Target Mapping
```python
from processed_dataset import load_target_mapping

# Load target mapping
target_to_idx, idx_to_target = load_target_mapping('target_mapping.pkl')

# Convert between original targets and indices
original_target = 139
target_idx = target_to_idx[original_target]  # e.g., 0
back_to_original = idx_to_target[target_idx]  # e.g., 139
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing Processed Data**
   ```bash
   # Run data preparation first
   python prepare_data.py
   ```

2. **Inconsistent Files**
   ```bash
   # Check project consistency
   python check_consistency.py
   ```

3. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use smaller model architecture

4. **Training Not Converging**
   - Try different learning rates
   - Adjust model architecture
   - Check data preprocessing

### Debug Mode

```bash
# Test components individually
python main.py test

# Check processed data loading
python -c "from processed_dataset import ProcessedQRTDataset; print(ProcessedQRTDataset('train_processed.csv', is_training=True)[0])"

# Test model forward pass
python -c "from model import MultiHeadMLP; import torch; model = MultiHeadMLP(); print(model(torch.randn(32, 100)))"
```

## 📝 Submission Format

The submission file `submit.csv` should contain:
```csv
ID,RET_TARGET
0,1
1,-1
2,1
...
```

Where:
- `ID`: Sample ID from test data
- `RET_TARGET`: Predicted sign (-1 or 1)

## 🔄 Migration from Old Version

If you have the old version:

1. **Backup your data**: Keep original data files
2. **Run data preparation**: `python prepare_data.py`
3. **Check consistency**: `python check_consistency.py`
4. **Update your scripts**: Use new processed data paths
5. **Retrain models**: Models need to be retrained with processed data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python check_consistency.py`
5. Submit a pull request

## 📄 License

This project is for educational purposes and the QRT Challenge. 