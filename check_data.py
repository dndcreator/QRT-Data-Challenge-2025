import pandas as pd
import numpy as np
import sys
import os

def analyze_ret_columns(df, dataset_name):
    """详细分析RET列的情况"""
    print(f"\n=== {dataset_name} RET列详细分析 ===")
    
    # 获取所有RET列
    ret_cols = [col for col in df.columns if col.startswith('RET_') and col != 'RET_TARGET']
    print(f"RET列总数: {len(ret_cols)}")
    
    if len(ret_cols) == 0:
        print("没有找到RET列")
        return
    
    # 按列名排序
    ret_cols.sort()
    print(f"RET列范围: {ret_cols[0]} 到 {ret_cols[-1]}")
    
    # 分析每个RET列
    print(f"\n{'列名':<12} {'非零值':<8} {'零值':<8} {'缺失值':<8} {'最小值':<10} {'最大值':<10} {'均值':<10} {'标准差':<10}")
    print("-" * 80)
    
    all_zero_cols = []
    all_missing_cols = []
    constant_cols = []
    
    for col in ret_cols:
        # 基本统计
        non_zero = (df[col] != 0).sum()
        zero_count = (df[col] == 0).sum()
        missing_count = df[col].isnull().sum()
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        # 检查特殊情况
        if non_zero == 0:
            all_zero_cols.append(col)
        if missing_count == len(df):
            all_missing_cols.append(col)
        if std_val == 0 and non_zero > 0:
            constant_cols.append(col)
        
        print(f"{col:<12} {non_zero:<8} {zero_count:<8} {missing_count:<8} {min_val:<10.4f} {max_val:<10.4f} {mean_val:<10.4f} {std_val:<10.4f}")
    
    # 总结特殊情况
    print(f"\n=== 特殊情况总结 ===")
    if all_zero_cols:
        print(f"⚠️  全为零的列 ({len(all_zero_cols)}个): {all_zero_cols}")
    else:
        print("✅ 没有全为零的列")
        
    if all_missing_cols:
        print(f"⚠️  全为缺失值的列 ({len(all_missing_cols)}个): {all_missing_cols}")
    else:
        print("✅ 没有全为缺失值的列")
        
    if constant_cols:
        print(f"⚠️  常数列 ({len(constant_cols)}个): {constant_cols}")
    else:
        print("✅ 没有常数列")
    
    # 检查是否有RET_105
    if 'RET_105' in ret_cols:
        print(f"\n🔍 RET_105 详细分析:")
        ret105_stats = df['RET_105'].describe()
        print(f"  统计信息: {ret105_stats}")
        print(f"  非零值数量: {(df['RET_105'] != 0).sum()}")
        print(f"  零值数量: {(df['RET_105'] == 0).sum()}")
        print(f"  缺失值数量: {df['RET_105'].isnull().sum()}")
        print(f"  唯一值: {df['RET_105'].unique()}")
    else:
        print(f"\n❌ 数据集中没有RET_105列")
        print(f"  可用的RET列: {ret_cols[:10]}... (共{len(ret_cols)}个)")

def check_single_file(file_path, dataset_name):
    """检查单个数据文件"""
    print(f"\n{'='*60}")
    print(f"检查文件: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    print(f"数据形状: {df.shape}")
    print(f"列数: {len(df.columns)}")
    
    # 检查缺失值
    print(f"\n=== 缺失值分析 ===")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        print("有缺失值的列:")
        print(missing_cols)
        print(f"总缺失值: {missing_counts.sum()}")
        print(f"缺失比例: {missing_counts.sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    else:
        print("✅ 没有缺失值")
    
    # 检查数据类型
    print(f"\n=== 数据类型分析 ===")
    print(df.dtypes.value_counts())
    
    # 详细分析RET列
    analyze_ret_columns(df, dataset_name)
    
    # 显示前几行
    print(f"\n=== 数据预览 ===")
    print("前3行数据:")
    print(df.head(3))
    
    # 检查目标变量分布
    if 'ID_TARGET' in df.columns:
        print(f"\n=== ID_TARGET分布 ===")
        target_counts = df['ID_TARGET'].value_counts().sort_index()
        print(f"唯一目标数: {len(target_counts)}")
        print(f"目标范围: {target_counts.index.min()} 到 {target_counts.index.max()}")
        print(f"最常见目标: {target_counts.head(5).to_dict()}")
    
    # 检查ID_DAY分布
    if 'ID_DAY' in df.columns:
        print(f"\n=== ID_DAY分布 ===")
        day_counts = df['ID_DAY'].value_counts().sort_index()
        print(f"唯一天数: {len(day_counts)}")
        print(f"天数范围: {day_counts.index.min()} 到 {day_counts.index.max()}")
        print(f"平均每天样本数: {len(df) / len(day_counts):.1f}")
    
    # 检查RET_TARGET分布（如果是标签文件）
    if 'RET_TARGET' in df.columns:
        print(f"\n=== RET_TARGET分布 ===")
        ret_target_stats = df['RET_TARGET'].describe()
        print(f"统计信息: {ret_target_stats}")
        positive_samples = (df['RET_TARGET'] > 0).sum()
        negative_samples = (df['RET_TARGET'] < 0).sum()
        zero_samples = (df['RET_TARGET'] == 0).sum()
        total_samples = len(df)
        print(f"正收益率: {positive_samples} ({positive_samples/total_samples*100:.1f}%)")
        print(f"负收益率: {negative_samples} ({negative_samples/total_samples*100:.1f}%)")
        print(f"零收益率: {zero_samples} ({zero_samples/total_samples*100:.1f}%)")

def main():
    # 定义要检查的文件列表
    files_to_check = [
        ('../X_train_itDkypA.csv', '训练输入数据'),
        ('../X_test_Beg4ey3.csv', '测试输入数据'),
        ('../y_train_3LeeT2g.csv', '训练标签数据')
    ]
    
    print("开始检查所有数据文件...")
    print("="*60)
    
    # 检查每个文件
    for file_path, dataset_name in files_to_check:
        check_single_file(file_path, dataset_name)
    
    print(f"\n{'='*60}")
    print("所有文件检查完成！")
    print("="*60)
    
    # 总结RET_105的情况
    print(f"\n🔍 RET_105 总结:")
    ret105_found = False
    for file_path, dataset_name in files_to_check:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'RET_105' in df.columns:
                    ret105_found = True
                    non_zero = (df['RET_105'] != 0).sum()
                    zero_count = (df['RET_105'] == 0).sum()
                    missing_count = df['RET_105'].isnull().sum()
                    print(f"  {dataset_name}: 非零={non_zero}, 零={zero_count}, 缺失={missing_count}")
                else:
                    print(f"  {dataset_name}: 没有RET_105列")
            except Exception as e:
                print(f"  {dataset_name}: 读取失败 - {e}")
        else:
            print(f"  {dataset_name}: 文件不存在")
    
    if not ret105_found:
        print("  ❌ 在所有原始数据文件中都没有找到RET_105列")

if __name__ == "__main__":
    main() 