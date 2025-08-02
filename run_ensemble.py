#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 双模型 Ensemble 一键运行脚本 (支持WBCE优化 + Optuna调参)
============================================================
使用基础特征，支持Weighted Binary Cross Entropy优化QRT Score
强制重新训练以获得最优结果

执行顺序：
1. 训练LightGBM模型并保存概率 (WBCE优化 + Optuna调参)
2. 训练CatBoost模型并保存概率 (WBCE优化 + Optuna调参)  
3. 执行双模型Stacking融合

使用方法：
python run_ensemble.py
"""

import subprocess
import sys
import os
import time
import shutil

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"开始执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True, encoding='utf-8')
        end_time = time.time()
        print(f"\n✅ {description} 完成 (耗时: {end_time - start_time:.1f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失败")
        print(f"错误信息: {e}")
        return False

def check_required_files():
    """检查必要的数据文件"""
    print("=== 检查必要文件 ===")
    
    required_files = [
        'train_processed.csv',
        'test_processed.csv'
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
        print(f"\n❌ 缺少必要文件: {missing_files}")
        print("请先运行: python prepare_data.py")
        return False
    
    print("✓ 所有必要文件都存在")
    return True

def train_wbce_models():
    """训练WBCE优化模型（使用最优参数）"""
    print("\n🚀 训练WBCE优化模型（使用最优参数）")
    print("=" * 60)
    print("🎯 使用已找到的最优参数直接训练")
    print("📊 预计训练时间: LightGBM约5分钟，CatBoost约5分钟")
    print("=" * 60)
    
    # 1. 训练LightGBM WBCE（使用最优参数）
    print("\n📊 步骤1: 训练LightGBM WBCE模型（最优参数）")
    if not run_command("python lightgbm_train.py", "LightGBM WBCE训练（最优参数）"):
        print("❌ LightGBM WBCE训练失败")
        return False
    
    # 2. 训练CatBoost WBCE（使用最优参数）
    print("\n📊 步骤2: 训练CatBoost WBCE模型（最优参数）")
    if not run_command("python catboost_train.py", "CatBoost WBCE训练（最优参数）"):
        print("❌ CatBoost WBCE训练失败")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 双模型 Ensemble 一键运行脚本 (支持WBCE优化)")
    print("="*70)
    print("使用Weighted Binary Cross Entropy优化QRT Score")
    print("🎯 使用最优参数直接训练")
    print("🔍 使用已找到的最优参数进行训练")
    print("="*70)
    
    # 1. 检查必要文件
    if not check_required_files():
        sys.exit(1)
    
    # 2. 训练WBCE模型（使用最优参数）
    print("\n🎯 训练WBCE优化模型（使用最优参数）")
    if not train_wbce_models():
        print("❌ WBCE模型训练失败")
        sys.exit(1)
    
    # 3. 执行Stacking融合
    print("\n" + "="*60)
    print("🔗 步骤3: 执行Stacking融合")
    print("="*60)
    
    if not run_command("python stacking_ensemble.py", "Stacking融合"):
        print("❌ Stacking融合失败")
        sys.exit(1)
    
    # 4. 最终检查
    print("\n" + "="*60)
    print("✅ 融合完成！检查输出文件")
    print("="*60)
    
    # 检查WBCE输出文件
    output_files = [
        'submit_lgb_wbce.csv',
        'submit_catboost_wbce.csv',
        'submit_加权平均_wbce.csv',
        'submit_高级stacking_wbce.csv',
        'submit_logistic_regression_wbce.csv',
        'ensemble_probability_comparison_wbce.png'
    ]
    
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✓ {file} ({size:.1f} KB)")
        else:
            print(f"⚠️  {file} - 未生成")
    
    print("\n🎉 所有步骤完成！")
    print("\n📋 输出文件说明 (WBCE优化):")
    print("- submit_lgb_wbce.csv: LightGBM WBCE单独预测结果")
    print("- submit_catboost_wbce.csv: CatBoost WBCE单独预测结果")
    print("- submit_加权平均_wbce.csv: 双模型WBCE加权平均融合结果")
    print("- submit_高级stacking_wbce.csv: 双模型WBCE Stacking融合结果")
    print("- submit_logistic_regression_wbce.csv: 双模型WBCE Logistic Regression融合结果")
    print("- ensemble_probability_comparison_wbce.png: WBCE概率分布对比图")
    print("\n✅ 使用WBCE优化模型，训练目标与评估指标一致")
    print("\n💡 建议: 选择验证集QRT Score更高的方法作为最终提交")
    print("\n🎯 使用最优参数直接训练，快速获得结果")

if __name__ == "__main__":
    main() 