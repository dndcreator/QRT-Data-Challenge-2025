import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
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
industry_cols = [col for col in train.columns if col.startswith('CLASS_LEVEL_')]  # 修正：CLASS_LEVEL_

# 确保特征列在测试集中也存在
common_cols = [col for col in feature_cols if col in test.columns]
common_industry_cols = [col for col in industry_cols if col in test.columns]

if len(common_cols) != len(feature_cols):
    print(f"⚠️ 警告: 训练集有 {len(feature_cols)} 个收益率特征，测试集只有 {len(common_cols)} 个共同特征")
    print(f"缺失收益率特征: {set(feature_cols) - set(test.columns)}")

if len(common_industry_cols) != len(industry_cols):
    print(f"⚠️ 警告: 训练集有 {len(industry_cols)} 个行业特征，测试集只有 {len(common_industry_cols)} 个共同特征")
    print(f"缺失行业特征: {set(industry_cols) - set(test.columns)}")

# 使用共同的特征
X = train[common_cols + common_industry_cols].copy()  # 包含行业分类特征
y = (train['target_sign'] > 0).astype(int)  # 1:涨, 0:不涨
sample_weights = train['RET_TARGET'].abs().values  # 使用RET_TARGET绝对值作为权重

# 处理分类特征：one-hot特征不需要特殊处理
# for col in common_industry_cols:
#     X[col] = X[col].astype(int)
#     if X[col].min() != 0:
#         print(f"⚠️ 调整 {col}: 最小值从 {X[col].min()} 调整为 0")
#         X[col] = X[col] - X[col].min()

print(f"特征统计: 收益率特征 {len(common_cols)} 个, 行业特征 {len(common_industry_cols)} 个")
print(f"行业特征: {common_industry_cols[:5]}...")  # 只显示前5个

# 3. 使用TimeSeriesSplit进行时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=3)

# 划分验证集（使用时间序列分割）
train_idx, val_idx = list(tscv.split(X))[-1]  # 使用最后一个分割作为验证集
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
train_weights = sample_weights[train_idx]
val_weights = sample_weights[val_idx]

print(f"CatBoost训练数据: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"CatBoost验证数据: {X_val.shape[0]} 样本")
print(f"权重统计: 训练集平均权重={np.mean(train_weights):.4f}, 验证集平均权重={np.mean(val_weights):.4f}")

# 4. 自动调参函数
def tune_catboost_params(X_train, X_val, y_train, y_val, train_weights, val_weights, common_industry_cols):
    """使用Optuna自动调参CatBoost - 使用样本权重"""
    print("=== 开始CatBoost自动调参 (WBCE优化) ===")
    
    def objective(trial):
        # 参考量化竞赛优秀做法：保守的参数范围，避免过拟合
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),  # 更保守的学习率
            'depth': trial.suggest_int('depth', 4, 8),  # 更保守的深度
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 10.0),  # 更强的正则化
            'bootstrap_type': 'Bernoulli',
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # 更保守的采样
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),  # 更严格的过拟合控制
            'random_seed': 42,
            'verbose': 0,
            'thread_count': -1
        }
        
        # 参考量化竞赛标准做法：使用时间序列交叉验证避免过拟合
        tscv = TimeSeriesSplit(n_splits=3)
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_cv_train = train_weights[train_idx]  # 使用对应的权重
            
            # one-hot特征不需要特殊处理
            X_cv_train_processed = X_cv_train.copy()
            X_cv_val_processed = X_cv_val.copy()
            # for col in common_industry_cols:
            #     X_cv_train_processed[col] = X_cv_train_processed[col].astype(int)
            #     X_cv_val_processed[col] = X_cv_val_processed[col].astype(int)
            
                    # 确保参数兼容性
        safe_params = params.copy()
        # Bernoulli bootstrap type 支持 subsample，其他类型不支持
        if safe_params.get('bootstrap_type') != 'Bernoulli':
            safe_params.pop('subsample', None)
            
            # 检测GPU并处理兼容性
            try:
                test_clf = CatBoostClassifier(task_type='GPU')
                task_type = 'GPU'
            except:
                task_type = 'CPU'
            
            if task_type == 'GPU':
                # GPU模式下不支持colsample_bylevel，移除该参数
                if 'colsample_bylevel' in safe_params:
                    safe_params.pop('colsample_bylevel')
            
            clf = CatBoostClassifier(task_type=task_type, **safe_params)
            clf.fit(
                X_cv_train_processed, y_cv_train,
                sample_weight=w_cv_train,  # 添加样本权重
                eval_set=(X_cv_val_processed, y_cv_val),
                early_stopping_rounds=50,
                verbose=0
            )
            
            val_probs = clf.predict_proba(X_cv_val_processed)[:, 1]
            # 使用QRT Score而不是AUC，更符合量化竞赛目标
            val_pred = (val_probs > 0.5).astype(int) * 2 - 1
            cv_weights = train.loc[X_cv_val.index, 'RET_TARGET'].abs().values
            qrt_score_cv = qrt_score(y_cv_val.values * 2 - 1, val_pred, cv_weights)
            cv_scores.append(qrt_score_cv)
        
        # 使用交叉验证的平均QRT Score
        mean_qrt_score = np.mean(cv_scores)
        
        return mean_qrt_score
        

    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)  # 增加到50次试验以获得最优结果
    
    print(f"CatBoost最佳综合评分: {study.best_value:.4f}")
    print(f"CatBoost最佳参数: {study.best_params}")
    
    return study.best_params

# 4. 使用最优参数（跳过调参）
print("=== 使用最优参数训练 (WBCE优化) ===")
print("🔍 使用已找到的最优参数直接训练...")

# 使用优化后的激进参数 - 最后一搏！
best_params = {
    'iterations': 1500,  # 增加迭代次数
    'learning_rate': 0.02,  # 降低学习率，更稳定
    'depth': 8,  # 增加深度
    'l2_leaf_reg': 3.0,  # 降低正则化
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,  # 增加采样
    'colsample_bylevel': 0.85,  # 增加特征采样
    'min_data_in_leaf': 15,  # 降低最小样本数
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss'
}

print(f"✅ 使用最优参数: {best_params}")

# 5. 训练模型 - 使用样本权重
print("开始训练CatBoost模型 (使用WBCE)...")

# 检测GPU并处理兼容性
try:
    test_clf = CatBoostClassifier(task_type='GPU')
    task_type = 'GPU'
    print("使用GPU训练")
except:
    task_type = 'CPU'
    print("使用CPU训练")

# 确保参数兼容性
if best_params.get('bootstrap_type') != 'Bernoulli':
    best_params.pop('subsample', None)

if task_type == 'GPU':
    # GPU模式下不支持colsample_bylevel，移除该参数
    if 'colsample_bylevel' in best_params:
        best_params.pop('colsample_bylevel')

clf = CatBoostClassifier(task_type=task_type, **best_params)
clf.fit(
    X_train, y_train,
    sample_weight=train_weights,  # 添加样本权重
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=0
)

# 6. 验证集评估
val_probs = clf.predict_proba(X_val)[:, 1]
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

# 7. 测试集预测
X_test = test[common_cols + common_industry_cols].copy()
# one-hot特征不需要特殊处理
# for col in common_industry_cols:
#     X_test[col] = X_test[col].astype(int)
#     # 确保测试集的分类特征值与训练集一致
#     if X_test[col].min() != 0:
#         X_test[col] = X_test[col] - X_test[col].min()

test_probs = clf.predict_proba(X_test)[:, 1]
test_pred = (test_probs > best_thresh).astype(int) * 2 - 1

# 8. 保存结果
submit = pd.DataFrame({'ID': test['ID'], 'RET_TARGET': test_pred})
submit.to_csv('submit_catboost_wbce.csv', index=False)

# 保存概率文件
val_prob_df = pd.DataFrame({
    'ID': X_val.index,
    'catboost_wbce_prob': val_probs
})
val_prob_df.to_csv('catboost_wbce_val_probs.csv', index=False)

test_prob_df = pd.DataFrame({
    'ID': test['ID'],
    'catboost_wbce_prob': test_probs
})
test_prob_df.to_csv('catboost_wbce_test_probs.csv', index=False)

print("✓ CatBoost (WBCE) 训练完成")
print(f"验证集QRT Score: {best_score:.4f}")
print(f"提交文件: submit_catboost_wbce.csv")
print(f"概率文件: catboost_wbce_val_probs.csv, catboost_wbce_test_probs.csv") 