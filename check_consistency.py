import os
import pandas as pd
import pickle
import torch
import numpy as np
from config import Config
from processed_dataset import ProcessedQRTDataset, load_target_mapping

def check_file_existence():
    """检查必要文件是否存在"""
    print("=== 文件存在性检查 ===")
    
    required_files = [
        'train_processed.csv',
        'test_processed.csv', 
        'target_mapping.pkl',
        'imputer.pkl',
        'scaler.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"✓ {file} ({size:.1f} MB)")
        else:
            print(f"✗ {file} - 缺失")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n警告: 缺少 {len(missing_files)} 个文件")
        return False
    else:
        print("\n✓ 所有必要文件都存在")
        return True

def check_data_consistency():
    """检查数据一致性"""
    print("\n=== 数据一致性检查 ===")
    
    # 加载处理后的数据
    try:
        train_data = pd.read_csv('train_processed.csv')
        test_data = pd.read_csv('test_processed.csv')
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        
        # 检查特征列（排除RET_TARGET）
        train_features = [col for col in train_data.columns if col.startswith('RET_') and col != 'RET_TARGET']
        test_features = [col for col in test_data.columns if col.startswith('RET_')]
        
        print(f"训练数据特征数: {len(train_features)}")
        print(f"测试数据特征数: {len(test_features)}")
        
        # 检查特征列是否一致
        if train_features == test_features:
            print("✓ 训练和测试数据特征列一致")
        else:
            print("✗ 训练和测试数据特征列不一致")
            return False
        
        # 检查训练数据的目标变量
        if 'target_idx' in train_data.columns:
            target_range = train_data['target_idx'].min(), train_data['target_idx'].max()
            print(f"目标索引范围: {target_range}")
            
            if target_range == (0, 99):
                print("✓ 目标索引范围正确 (0-99)")
            else:
                print(f"✗ 目标索引范围异常: {target_range}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载错误: {e}")
        return False

def check_target_mapping():
    """检查目标映射"""
    print("\n=== 目标映射检查 ===")
    
    try:
        target_to_idx, idx_to_target = load_target_mapping()
        
        print(f"映射字典大小: {len(target_to_idx)}")
        print(f"映射范围: {min(target_to_idx.values())} - {max(target_to_idx.values())}")
        
        # 检查映射的一致性
        for target, idx in target_to_idx.items():
            if idx_to_target[idx] != target:
                print(f"✗ 映射不一致: {target} -> {idx} -> {idx_to_target[idx]}")
                return False
        
        print("✓ 目标映射一致")
        return True
        
    except Exception as e:
        print(f"✗ 目标映射加载错误: {e}")
        return False

def check_preprocessors():
    """检查预处理器"""
    print("\n=== 预处理器检查 ===")
    
    try:
        # 加载预处理器
        with open('imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"缺失值填充器: {type(imputer).__name__}")
        print(f"特征标准化器: {type(scaler).__name__}")
        
        # 检查预处理器参数
        if hasattr(imputer, 'statistics_'):
            print(f"填充器统计信息形状: {imputer.statistics_.shape}")
        
        if hasattr(scaler, 'mean_'):
            print(f"标准化器均值形状: {scaler.mean_.shape}")
            print(f"标准化器标准差形状: {scaler.scale_.shape}")
        
        print("✓ 预处理器加载成功")
        return True
        
    except Exception as e:
        print(f"✗ 预处理器加载错误: {e}")
        return False

def check_dataset_compatibility():
    """检查数据集兼容性"""
    print("\n=== 数据集兼容性检查 ===")
    
    try:
        # 测试训练数据集
        train_dataset = ProcessedQRTDataset('train_processed.csv', is_training=True)
        print(f"训练数据集样本数: {len(train_dataset)}")
        print(f"输入维度: {train_dataset.input_dim}")
        
        # 测试一个样本
        sample = train_dataset[0]
        if len(sample) == 4:
            X, target_idx, target_sign, weight = sample
            print(f"样本格式正确: X({X.shape}), target_idx({target_idx}), target_sign({target_sign}), weight({weight})")
        else:
            print(f"✗ 样本格式错误: {len(sample)} 个元素")
            return False
        
        # 检查目标符号分布（修复负值问题）
        try:
            unique_signs, counts = np.unique(train_dataset.target_sign, return_counts=True)
            sign_dist = dict(zip(unique_signs, counts))
            print(f"目标符号分布: {sign_dist}")
        except Exception as e:
            print(f"目标符号分布检查失败: {e}")
            return False
        
        # 测试测试数据集
        test_dataset = ProcessedQRTDataset('test_processed.csv', is_training=False)
        print(f"测试数据集样本数: {len(test_dataset)}")
        print(f"输入维度: {test_dataset.input_dim}")
        
        # 测试一个样本
        sample = test_dataset[0]
        if isinstance(sample, np.ndarray):
            print(f"测试样本格式正确: X({sample.shape})")
        else:
            print(f"✗ 测试样本格式错误: {type(sample)}")
            return False
        
        print("✓ 数据集兼容性检查通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据集兼容性检查失败: {e}")
        return False

def check_config_consistency():
    """检查配置一致性"""
    print("\n=== 配置一致性检查 ===")
    
    try:
        config = Config()
        
        # 检查配置中的路径
        if os.path.exists(config.train_processed_file):
            print(f"✓ 训练数据路径正确: {config.train_processed_file}")
        else:
            print(f"✗ 训练数据路径错误: {config.train_processed_file}")
            return False
        
        if os.path.exists(config.test_processed_file):
            print(f"✓ 测试数据路径正确: {config.test_processed_file}")
        else:
            print(f"✗ 测试数据路径错误: {config.test_processed_file}")
            return False
        
        # 检查模型配置
        print(f"模型类型: {config.model_type}")
        print(f"隐藏层维度: {config.hidden_dims}")
        print(f"目标头数: {config.num_heads}")
        print(f"批次大小: {config.batch_size}")
        print(f"学习率: {config.learning_rate}")
        
        print("✓ 配置一致性检查通过")
        return True
        
    except Exception as e:
        print(f"✗ 配置一致性检查失败: {e}")
        return False

def main():
    """主检查函数"""
    print("开始项目一致性检查...\n")
    
    checks = [
        check_file_existence,
        check_data_consistency,
        check_target_mapping,
        check_preprocessors,
        check_dataset_compatibility,
        check_config_consistency
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"检查失败: {e}")
            results.append(False)
    
    # 总结
    print("\n=== 检查总结 ===")
    passed = sum(results)
    total = len(results)
    
    print(f"通过检查: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有检查都通过了！项目文件保持一致。")
        print("\n下一步可以:")
        print("1. 训练模型: python main.py train")
        print("2. 生成预测: python main.py predict")
        print("3. 评估模型: python main.py evaluate")
    else:
        print("⚠️  部分检查未通过，请检查上述问题。")
    
    return passed == total

if __name__ == "__main__":
    main() 