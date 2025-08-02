import itertools
from config import Config
from train import train_and_evaluate, Trainer
from processed_dataset import ProcessedQRTDataset
from model import create_model
import torch
from torch.utils.data import DataLoader

# 参数网格
param_grid = {
    'hidden_dims': [[256,128,64], [512,256,128,64]],
    'dropout': [0.2, 0.3],
    'learning_rate': [1e-3, 5e-4],
    'batch_size': [128, 256]
}

# 生成所有参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_score = -float('inf')
best_params = None

for params in param_combinations:
    print(f"Testing params: {params}")
    config = Config()
    for k, v in params.items():
        setattr(config, k, v)
    val_score = train_and_evaluate(config)
    print(f"Val score: {val_score}")
    if val_score > best_score:
        best_score = val_score
        best_params = params

print(f"Best params: {best_params}, best val score: {best_score}")

# 用最优参数全量训练并保存模型
print(f"\nRetraining on full data with best params: {best_params}")
config = Config()
for k, v in best_params.items():
    setattr(config, k, v)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_train_dataset = ProcessedQRTDataset(config.train_data_path)
full_train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

model = create_model(
    model_type=config.model_type,
    input_dim=full_train_dataset.input_dim,
    hidden_dims=config.hidden_dims,
    num_heads=100,
    dropout=config.dropout,
    activation=config.activation
)
trainer = Trainer(
    model=model,
    device=device,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay
)
# 只训练num_epochs轮，不早停
for epoch in range(config.num_epochs):
    loss = trainer.train_epoch(full_train_loader)
    print(f"Full train epoch {epoch+1}/{config.num_epochs}, loss: {loss:.4f}")

# 保存最终模型
torch.save(model.state_dict(), config.model_save_path)
print(f"Final model saved to {config.model_save_path}") 