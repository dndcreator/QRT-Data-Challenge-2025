import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

def load_all_data():
    """加载所有数据文件"""
    print("=== 加载所有数据文件 ===")
    
    # 加载训练输入数据
    print("加载训练输入数据...")
    X_train = pd.read_csv('../X_train_itDkypA.csv')
    print(f"X_train形状: {X_train.shape}")
    
    # 加载训练标签数据
    print("加载训练标签数据...")
    y_train = pd.read_csv('../y_train_3LeeT2g.csv')
    print(f"y_train形状: {y_train.shape}")
    
    # 加载测试数据
    print("加载测试数据...")
    X_test = pd.read_csv('../X_test_Beg4ey3.csv')
    print(f"X_test形状: {X_test.shape}")
    
    return X_train, y_train, X_test

def analyze_target_mapping(X_train, y_train):
    """分析目标变量映射"""
    print("\n=== 目标变量分析 ===")
    
    # 检查ID匹配
    print("检查ID匹配...")
    train_ids = set(X_train['ID'])
    label_ids = set(y_train['ID'])
    common_ids = train_ids.intersection(label_ids)
    print(f"X_train ID数量: {len(train_ids)}")
    print(f"y_train ID数量: {len(label_ids)}")
    print(f"共同ID数量: {len(common_ids)}")
    
    # 分析ID_TARGET分布
    print("\nID_TARGET分布:")
    target_counts = X_train['ID_TARGET'].value_counts().sort_index()
    print(f"唯一ID_TARGET数量: {len(target_counts)}")
    print(f"ID_TARGET范围: {target_counts.index.min()} - {target_counts.index.max()}")
    print(f"前10个ID_TARGET: {target_counts.head(10).to_dict()}")
    
    # 分析RET_TARGET分布
    print("\nRET_TARGET分布:")
    ret_stats = y_train['RET_TARGET'].describe()
    print(f"RET_TARGET统计: {ret_stats}")
    
    # 检查正负样本分布
    positive_samples = (y_train['RET_TARGET'] > 0).sum()
    negative_samples = (y_train['RET_TARGET'] < 0).sum()
    zero_samples = (y_train['RET_TARGET'] == 0).sum()
    total_samples = len(y_train)
    
    print(f"\n收益率分布:")
    print(f"  正收益率: {positive_samples} ({positive_samples/total_samples*100:.1f}%)")
    print(f"  负收益率: {negative_samples} ({negative_samples/total_samples*100:.1f}%)")
    print(f"  零收益率: {zero_samples} ({zero_samples/total_samples*100:.1f}%)")

def create_target_mapping(X_train):
    """创建目标变量映射"""
    print("\n=== 创建目标变量映射 ===")
    
    # 获取唯一的ID_TARGET值并排序
    unique_targets = sorted(X_train['ID_TARGET'].unique())
    print(f"唯一ID_TARGET值: {unique_targets}")
    
    # 创建映射字典
    target_to_idx = {target: idx for idx, target in enumerate(unique_targets)}
    idx_to_target = {idx: target for target, idx in target_to_idx.items()}
    
    print(f"映射字典大小: {len(target_to_idx)}")
    print(f"映射范围: 0-{len(target_to_idx)-1}")
    
    # 保存映射
    with open('target_mapping.pkl', 'wb') as f:
        pickle.dump({'target_to_idx': target_to_idx, 'idx_to_target': idx_to_target}, f)
    
    return target_to_idx, idx_to_target

def merge_and_prepare_data(X_train, y_train, X_test, target_to_idx):
    """合并和准备数据"""
    print("\n=== 合并和准备数据 ===")
    
    # 合并训练数据
    print("合并训练数据...")
    train_merged = pd.merge(X_train, y_train, on='ID', how='inner')
    print(f"合并后训练数据形状: {train_merged.shape}")
    
    # 创建目标变量
    print("创建目标变量...")
    train_merged['target_idx'] = train_merged['ID_TARGET'].map(target_to_idx)
    train_merged['target_sign'] = np.sign(train_merged['RET_TARGET'])
    
    # 检查目标变量分布
    print(f"目标索引分布: {train_merged['target_idx'].value_counts().sort_index().to_dict()}")
    print(f"目标符号分布: {train_merged['target_sign'].value_counts().to_dict()}")
    
    # 合并行业特征
    industry_df = pd.read_csv('supplementary_data_Vkoyn8z.csv')
    industry_cols = [col for col in industry_df.columns if col.startswith('CLASS_LEV')]
    print(f"原始行业特征: {industry_cols}")
    
    # 将行业特征转换为one-hot编码
    print("转换行业特征为one-hot编码...")
    industry_encoders = {}
    industry_num_classes = {}
    
    # 先进行LabelEncoder
    for col in industry_cols:
        industry_df[col] = industry_df[col].fillna('unknown')
        le = LabelEncoder()
        industry_df[col] = le.fit_transform(industry_df[col].astype(str))
        industry_encoders[col] = le
        industry_num_classes[col] = len(le.classes_)
    
    # 转换为真正的one-hot编码
    print("转换为one-hot编码...")
    one_hot_columns = []
    for col in industry_cols:
        # 创建one-hot编码，使用CLASS_LEVEL_前缀以保持一致性
        one_hot = pd.get_dummies(industry_df[col], prefix=col.replace('CLASS_LEV', 'CLASS_LEVEL_'))
        # 将True/False转换为1/0
        one_hot = one_hot.astype(int)
        one_hot_columns.append(one_hot)
    
    # 合并所有one-hot特征，并保留ID_asset列用于合并
    industry_one_hot = pd.concat([industry_df[['ID_asset']]] + one_hot_columns, axis=1)
    print(f"One-hot编码完成: {industry_one_hot.shape[1]} 个特征")
    
    # 保存编码器和类别数
    with open('industry_encoders.pkl', 'wb') as f:
        pickle.dump({'encoders': industry_encoders, 'num_classes': industry_num_classes}, f)
    
    # 合并到训练集和测试集
    train_merged = train_merged.merge(industry_one_hot, left_on='ID_TARGET', right_on='ID_asset', how='left')
    X_test = X_test.merge(industry_one_hot, left_on='ID_TARGET', right_on='ID_asset', how='left')

    id_cols = ['ID', 'ID_DAY']
    # 更新特征列
    feature_cols = [col for col in X_train.columns if col.startswith('RET_') and col != 'RET_TARGET']
    
    # 提取特征数据
    X_train_features = train_merged[feature_cols].copy()
    X_test_features = X_test[feature_cols].copy()
    industry_train = train_merged[industry_one_hot.columns].copy()
    industry_test = X_test[industry_one_hot.columns].copy()
    
    # 处理行业特征的缺失值
    print("处理行业特征缺失值...")
    for col in industry_one_hot.columns:
        # 检查缺失值
        train_missing = industry_train[col].isnull().sum()
        test_missing = industry_test[col].isnull().sum()
        if train_missing > 0 or test_missing > 0:
            print(f"⚠️  {col}: 训练集缺失 {train_missing}, 测试集缺失 {test_missing}")
            # 对于one-hot特征，缺失值填充为0
            industry_train[col] = industry_train[col].fillna(0)
            industry_test[col] = industry_test[col].fillna(0)
            print(f"✅  {col}: 缺失值填充为0")
        else:
            print(f"✅  {col}: 无缺失值")
    
    # 处理缺失值（只对收益率特征）
    print("处理收益率特征缺失值...")
    
    # 检查RET_105在原始数据中的情况
    if 'RET_105' in X_train_features.columns:
        print(f"RET_105 原始数据统计:")
        print(f"  非零值: {(X_train_features['RET_105'] != 0).sum()}")
        print(f"  零值: {(X_train_features['RET_105'] == 0).sum()}")
        print(f"  缺失值: {X_train_features['RET_105'].isnull().sum()}")
        print(f"  均值: {X_train_features['RET_105'].mean():.6f}")
        print(f"  标准差: {X_train_features['RET_105'].std():.6f}")
    
    # 检查缺失值比例，如果某个列的缺失值比例过高，使用不同的策略
    print("检查缺失值比例...")
    missing_ratios = X_train_features.isnull().sum() / len(X_train_features)
    high_missing_cols = missing_ratios[missing_ratios > 0.5].index.tolist()
    
    if high_missing_cols:
        print(f"⚠️  高缺失值列 ({len(high_missing_cols)}个): {high_missing_cols}")
        print("使用中位数填充高缺失值列，均值填充其他列")
        
        # 对高缺失值列使用中位数填充
        imputer_median = SimpleImputer(strategy='median')
        X_train_high_missing = X_train_features[high_missing_cols]
        X_test_high_missing = X_test_features[high_missing_cols]
        
        X_train_high_missing_imputed = pd.DataFrame(
            imputer_median.fit_transform(X_train_high_missing),
            columns=high_missing_cols,
            index=X_train_features.index
        )
        X_test_high_missing_imputed = pd.DataFrame(
            imputer_median.transform(X_test_high_missing),
            columns=high_missing_cols,
            index=X_test_features.index
        )
        
        # 对其他列使用均值填充
        other_cols = [col for col in X_train_features.columns if col not in high_missing_cols]
        imputer_mean = SimpleImputer(strategy='mean')
        X_train_other = X_train_features[other_cols]
        X_test_other = X_test_features[other_cols]
        
        X_train_other_imputed = pd.DataFrame(
            imputer_mean.fit_transform(X_train_other),
            columns=other_cols,
            index=X_train_features.index
        )
        X_test_other_imputed = pd.DataFrame(
            imputer_mean.transform(X_test_other),
            columns=other_cols,
            index=X_test_features.index
        )
        
        # 合并结果
        X_train_imputed = pd.concat([X_train_other_imputed, X_train_high_missing_imputed], axis=1)
        X_test_imputed = pd.concat([X_test_other_imputed, X_test_high_missing_imputed], axis=1)
        
        # 保存两个imputer
        imputers = {'mean': imputer_mean, 'median': imputer_median, 'high_missing_cols': high_missing_cols}
    else:
        print("✅ 所有列的缺失值比例都正常，使用均值填充")
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train_features),
            columns=X_train_features.columns,
            index=X_train_features.index
        )
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test_features),
            columns=X_test_features.columns,
            index=X_test_features.index
        )
        imputers = {'mean': imputer}
    
    # 检查SimpleImputer的填充值
    if 'RET_105' in X_train_features.columns:
        print(f"RET_105 SimpleImputer填充值: {imputer.statistics_[X_train_features.columns.get_loc('RET_105')]:.6f}")
        print(f"RET_105 原始缺失值比例: {X_train_features['RET_105'].isnull().sum() / len(X_train_features) * 100:.2f}%")
        print(f"RET_105 填充后缺失值比例: {X_train_imputed['RET_105'].isnull().sum() / len(X_train_imputed) * 100:.2f}%")
    
    # 检查RET_105在填充后的情况
    if 'RET_105' in X_train_imputed.columns:
        print(f"RET_105 填充后统计:")
        print(f"  非零值: {(X_train_imputed['RET_105'] != 0).sum()}")
        print(f"  零值: {(X_train_imputed['RET_105'] == 0).sum()}")
        print(f"  缺失值: {X_train_imputed['RET_105'].isnull().sum()}")
        print(f"  均值: {X_train_imputed['RET_105'].mean():.6f}")
        print(f"  标准差: {X_train_imputed['RET_105'].std():.6f}")
    
    # 标准化特征（只对收益率特征进行标准化）
    print("标准化收益率特征...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train_imputed.columns,
        index=X_train_imputed.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X_test_imputed.columns,
        index=X_test_imputed.index
    )
    
    # 检查RET_105在标准化后的情况
    if 'RET_105' in X_train_scaled.columns:
        print(f"RET_105 标准化后统计:")
        print(f"  非零值: {(X_train_scaled['RET_105'] != 0).sum()}")
        print(f"  零值: {(X_train_scaled['RET_105'] == 0).sum()}")
        print(f"  缺失值: {X_train_scaled['RET_105'].isnull().sum()}")
        print(f"  均值: {X_train_scaled['RET_105'].mean():.6f}")
        print(f"  标准差: {X_train_scaled['RET_105'].std():.6f}")
        print(f"  最小值: {X_train_scaled['RET_105'].min():.6f}")
        print(f"  最大值: {X_train_scaled['RET_105'].max():.6f}")
    
    # 行业特征保持原始one-hot值，不进行标准化
    print("保持行业特征为原始one-hot值...")
    # industry_train 和 industry_test 已经是one-hot编码，不需要处理
    
    # 创建最终数据集
    print("创建最终数据集...")
    train_final = pd.concat([
        train_merged[id_cols],
        X_train_scaled,  # 标准化后的收益率特征
        industry_train,   # 原始one-hot的行业特征
        train_merged[['target_idx', 'target_sign', 'RET_TARGET']]
    ], axis=1)
    
    test_final = pd.concat([
        X_test[id_cols],
        X_test_scaled,   # 标准化后的收益率特征
        industry_test    # 原始one-hot的行业特征
    ], axis=1)
    
    # 保存预处理器
    print("保存预处理器...")
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputers, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return train_final, test_final, imputer, scaler

def save_processed_data(train_final, test_final):
    """保存处理后的数据"""
    print("\n=== 保存处理后的数据 ===")
    
    # 保存训练数据
    train_final.to_csv('train_processed.csv', index=False)
    print(f"训练数据已保存: train_processed.csv ({train_final.shape})")
    
    # 保存测试数据
    test_final.to_csv('test_processed.csv', index=False)
    print(f"测试数据已保存: test_processed.csv ({test_final.shape})")
    
    # 保存数据统计信息
    stats = {
        'train_samples': len(train_final),
        'test_samples': len(test_final),
        'features': len([col for col in train_final.columns if col.startswith('RET_') and col != 'RET_TARGET']),
        'target_classes': train_final['target_idx'].nunique(),
        'positive_samples': (train_final['target_sign'] > 0).sum(),
        'negative_samples': (train_final['target_sign'] < 0).sum(),
        'zero_samples': (train_final['target_sign'] == 0).sum()
    }
    
    print(f"\n数据统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def main():
    """主函数"""
    print("开始数据准备...")
    
    # 加载数据
    X_train, y_train, X_test = load_all_data()
    
    # 分析目标变量
    analyze_target_mapping(X_train, y_train)
    
    # 创建目标映射
    target_to_idx, idx_to_target = create_target_mapping(X_train)
    
    # 合并和准备数据
    train_final, test_final, imputer, scaler = merge_and_prepare_data(
        X_train, y_train, X_test, target_to_idx
    )
    
    # 保存数据
    save_processed_data(train_final, test_final)
    
    print("\n=== 数据准备完成 ===")
    print("生成的文件:")
    print("  - train_processed.csv: 处理后的训练数据")
    print("  - test_processed.csv: 处理后的测试数据")
    print("  - target_mapping.pkl: 目标变量映射")
    print("  - imputer.pkl: 缺失值填充器")
    print("  - scaler.pkl: 特征标准化器")

if __name__ == "__main__":
    main() 