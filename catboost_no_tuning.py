import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import optuna

# QRT Score 计算函数
def qrt_score(y_true, y_pred, sample_weight):
    return np.sum((y_true == y_pred) * sample_weight) / np.sum(sample_weight)

# 1. 读取数据
train = pd.read_csv('train_processed.csv')
test = pd.read_csv('test_processed.csv')

# 2. 特征与标签
feature_cols = [col for col in train.columns if col.startswith('RET_') and col != 'RET_TARGET']
industry_cols = [col for col in train.columns if col.startswith('CLASS_LEV')]
X = train[feature_cols + industry_cols].copy()
y = (train['target_sign'] > 0).astype(int)  # 1:涨, 0:不涨

# 处理类别特征：确保行业特征是整数类型
for col in industry_cols:
    X[col] = X[col].astype(int)

# 3. 划分验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算验证集权重
val_weights = train.loc[X_val.index, 'RET_TARGET'].abs().values

# 4. 使用默认参数（不调参）
print("=== 使用默认参数训练CatBoost ===")

# 5. 使用默认参数初始化模型
# 检测GPU可用性并优化性能
try:
    test_clf = CatBoostClassifier(task_type='GPU')
    task_type = 'GPU'
    print("✓ 使用GPU训练")
except:
    task_type = 'CPU'
    print("⚠️ 使用CPU训练")

# 使用默认参数
clf = CatBoostClassifier(
    iterations=1000,
    random_seed=42,
    verbose=100,
    task_type=task_type,
    thread_count=-1,  # 使用所有CPU核心
    bootstrap_type='Bernoulli'
)

# 6. 训练
print('Training CatBoost with default parameters...')
print(f"训练数据大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"验证数据大小: {X_val.shape[0]} 样本")

# 确保训练和验证数据都是整数类型
X_train_processed = X_train.copy()
X_val_processed = X_val.copy()
for col in industry_cols:
    X_train_processed[col] = X_train_processed[col].astype(int)
    X_val_processed[col] = X_val_processed[col].astype(int)

clf.fit(
    X_train_processed, y_train,
    eval_set=(X_val_processed, y_val),
    cat_features=industry_cols,
    early_stopping_rounds=50,  # 增加早停轮数，给模型更多机会
    verbose=100
)

# 7. 验证集评估
val_probs = clf.predict_proba(X_val_processed)[:, 1]
y_val_true = y_val.values

# 调优阈值
best_score = -1
best_thresh = 0.5
for thresh in np.arange(0.1, 0.9, 0.01):
    val_pred = (val_probs > thresh).astype(int) * 2 - 1
    score = qrt_score(y_val_true * 2 - 1, val_pred, val_weights)
    if score > best_score:
        best_score = score
        best_thresh = thresh
print(f'最优阈值: {best_thresh:.2f}, 验证集QRT Score: {best_score:.4f}')

# 8. 预测
X_test = test[feature_cols + industry_cols].copy()
# 处理测试集的类别特征
for col in industry_cols:
    X_test[col] = X_test[col].astype(int)
test_probs = clf.predict_proba(X_test)[:, 1]
test_pred = (test_probs > best_thresh).astype(int) * 2 - 1

# 9. 生成提交文件
submit = pd.DataFrame({'ID': test['ID'], 'RET_TARGET': test_pred})
submit.to_csv('submit_catboost_no_tuning.csv', index=False)
print('CatBoost预测结果已保存到 submit_catboost_no_tuning.csv')

# 10. 保存概率文件（用于stacking）
val_prob_df = pd.DataFrame({
    'ID': X_val.index,
    'catboost_prob': val_probs
})
val_prob_df.to_csv('catboost_val_probs_no_tuning.csv', index=False)

test_prob_df = pd.DataFrame({
    'ID': test['ID'],
    'catboost_prob': test_probs
})
test_prob_df.to_csv('catboost_test_probs_no_tuning.csv', index=False)
print('CatBoost概率文件已保存，可用于stacking')

# 11. 可选：特征重要性
try:
    import matplotlib.pyplot as plt
    feature_importance = clf.get_feature_importance()
    feature_names = X.columns
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n=== 特征重要性 Top 20 ===")
    print(importance_df.head(20))
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('CatBoost Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('catboost_feature_importance_no_tuning.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("特征重要性图已保存: catboost_feature_importance_no_tuning.png")
    
except ImportError:
    print("matplotlib未安装，跳过特征重要性可视化")
except Exception as e:
    print(f"特征重要性可视化失败: {e}")

print("\n=== CatBoost默认参数训练完成 ===") 