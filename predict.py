import torch
import pandas as pd
from processed_dataset import ProcessedQRTDataset
from model import create_model
from config import Config


def main():
    # 加载配置
    config = Config()
    device = config.device

    # 检查test_processed.csv是否有ID列，没有则加上train的列名
    test_path = config.test_data_path
    df_test = pd.read_csv(test_path, nrows=5)
    if 'ID' not in df_test.columns:
        train_cols = pd.read_csv(config.train_data_path, nrows=0).columns
        df_test_full = pd.read_csv(test_path, header=None)
        df_test_full.columns = train_cols
        df_test_full.to_csv(test_path, index=False)
        print("已为 test_processed.csv 添加列名。")

    # 加载测试数据
    print("Loading test dataset...")
    test_dataset = ProcessedQRTDataset(config.test_data_path, is_training=False)

    # 加载模型
    print("Loading trained model...")
    model = create_model(
        model_type=config.model_type,
        input_dim=100,
        hidden_dims=config.hidden_dims,
        num_heads=100,
        dropout=config.dropout,
        activation=config.activation
    )
    checkpoint = torch.load(config.model_load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 预测
    print("Generating predictions...")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.inference_batch_size, shuffle=False, num_workers=0
    )
    all_predictions = []
    all_ids = []
    for batch in test_loader:
        X, industry_ids, ids = batch
        X = X.to(device)
        industry_ids = industry_ids.to(device)
        # 假设test_processed.csv有ID_TARGET列
        if hasattr(test_dataset, 'data') and 'ID_TARGET' in test_dataset.data.columns:
            id_targets = torch.tensor(test_dataset.data.loc[ids.cpu().numpy(), 'ID_TARGET'].values, dtype=torch.long, device=device)
        else:
            id_targets = torch.arange(X.size(0), device=device) % 100
        logits = model(X, industry_ids, id_targets)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float() * 2 - 1
        all_predictions.extend(preds.cpu().numpy().astype(int).tolist())
        all_ids.extend(ids.cpu().numpy().astype(int).tolist())

    # 生成提交文件
    print("Saving submission file...")
    submission = pd.DataFrame({
        'ID': all_ids,
        'RET_TARGET': all_predictions
    })
    submission.to_csv(config.submission_file, index=False)
    print(f"Submission saved to {config.submission_file}")

if __name__ == "__main__":
    main() 