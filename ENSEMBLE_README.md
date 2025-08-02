# LightGBM + CatBoost Stacking 融合使用指南

## 📋 概述

本项目实现了LightGBM和CatBoost的stacking融合，包含两种融合方式：
- **加权平均融合**：简单有效的概率加权
- **逻辑回归Stacking**：使用逻辑回归作为meta模型
- **自动调参**：使用Optuna对每个模型进行智能参数优化

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 一键运行（推荐）
```bash
python run_ensemble.py
```

这个脚本会自动：
- 检查必要文件
- 训练LightGBM模型（如果概率文件不存在）
- 训练CatBoost模型（如果概率文件不存在）
- 执行stacking融合
- 生成最终提交文件

### 3. 分步运行

#### 步骤1：训练LightGBM
```bash
python lightgbm_train.py
```

#### 步骤2：训练CatBoost
```bash
python catboost_train.py
```

#### 步骤3：执行Stacking融合
```bash
python stacking_ensemble.py
```

## 📊 输出文件

### 训练阶段
- `lightgbm_val_probs.csv` / `lightgbm_test_probs.csv`：LightGBM概率文件
- `catboost_val_probs.csv` / `catboost_test_probs.csv`：CatBoost概率文件
- `submit_lgb.csv`：LightGBM单独预测结果
- `submit_catboost.csv`：CatBoost单独预测结果

### 融合阶段
- `submit_weighted_average.csv`：加权平均融合结果
- `submit_逻辑回归stacking.csv`：逻辑回归Stacking融合结果
- `ensemble_probability_comparison.png`：概率分布对比图

## ⚙️ 参数调整

### 加权平均权重
在`stacking_ensemble.py`中修改：
```python
weights = [0.6, 0.4]  # LightGBM权重0.6，CatBoost权重0.4
```

### CatBoost参数
在`catboost_train.py`中调整：
```python
clf = CatBoostClassifier(
    iterations=500,      # 迭代次数
    learning_rate=0.05,  # 学习率
    depth=10,           # 树深度
    l2_leaf_reg=3.0,    # L2正则化
    # ... 其他参数
)
```

## 📈 性能对比

脚本会自动输出各方法的验证集QRT Score对比：
- LightGBM单独：约0.73-0.75（自动调参后）
- CatBoost单独：约0.70-0.73（自动调参后）
- 加权平均融合：通常比单模提升0.01-0.02
- 逻辑回归Stacking：通常比加权平均略好

**自动调参预期提升**：每个模型通过Optuna调参后，QRT Score通常能提升0.02-0.03

## 💡 使用建议

1. **首次运行**：使用`python run_ensemble.py`一键完成
2. **调参优化**：先单独调优LightGBM和CatBoost参数
3. **权重调整**：根据验证集表现调整加权平均权重
4. **最终提交**：选择验证集QRT Score最高的方法

## 🔧 故障排除

### 常见问题
1. **缺少数据文件**：先运行`python prepare_data.py`
2. **CatBoost安装失败**：尝试`pip install catboost --no-deps`
3. **内存不足**：减少batch_size或iterations参数

### 调试模式
```bash
# 单独测试CatBoost
python catboost_train.py

# 单独测试Stacking
python stacking_ensemble.py
```

## 📝 扩展建议

1. **添加更多模型**：XGBoost等
2. **K折交叉验证**：提升stacking稳定性
3. **特征工程**：为不同模型使用不同特征组合
4. **自动调参**：使用Optuna等工具自动优化参数

---

**Happy Stacking! 🎯** 