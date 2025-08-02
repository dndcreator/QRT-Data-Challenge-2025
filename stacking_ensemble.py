import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def qrt_score(y_true, y_pred, sample_weight):
    """QRT Score 计算函数"""
    # 确保输入是numpy数组，避免pandas Series索引问题
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = np.array(sample_weight)
    return np.sum((y_true == y_pred) * sample_weight) / np.sum(sample_weight)

def load_probabilities():
    """加载模型概率文件（支持WBCE和基础模型）"""
    print("=== 加载模型概率文件 ===")
    
    # 检查WBCE模型概率文件
    wbce_files = [
        'lightgbm_wbce_val_probs.csv',
        'lightgbm_wbce_test_probs.csv',
        'catboost_wbce_val_probs.csv',
        'catboost_wbce_test_probs.csv'
    ]
    
    # 检查基础特征的概率文件
    basic_files = [
        'lightgbm_val_probs.csv',
        'lightgbm_test_probs.csv',
        'catboost_val_probs.csv',
        'catboost_test_probs.csv'
    ]
    
    # 优先使用WBCE模型
    if all(os.path.exists(f) for f in wbce_files):
        print("✓ 使用WBCE模型概率文件")
        lgb_val = pd.read_csv('lightgbm_wbce_val_probs.csv')
        cat_val = pd.read_csv('catboost_wbce_val_probs.csv')
        lgb_test = pd.read_csv('lightgbm_wbce_test_probs.csv')
        cat_test = pd.read_csv('catboost_wbce_test_probs.csv')
        
        print(f"LightGBM WBCE验证集概率: {len(lgb_val)} 样本")
        print(f"CatBoost WBCE验证集概率: {len(cat_val)} 样本")
        print(f"LightGBM WBCE测试集概率: {len(lgb_test)} 样本")
        print(f"CatBoost WBCE测试集概率: {len(cat_test)} 样本")
        
        print("⚠️ 使用双模型WBCE融合")
        return lgb_val, cat_val, None, lgb_test, cat_test, None, 'wbce'
    
    # 如果没有WBCE文件，使用基础模型
    elif all(os.path.exists(f) for f in basic_files):
        print("✓ 使用基础特征的概率文件")
        lgb_val = pd.read_csv('lightgbm_val_probs.csv')
        cat_val = pd.read_csv('catboost_val_probs.csv')
        lgb_test = pd.read_csv('lightgbm_test_probs.csv')
        cat_test = pd.read_csv('catboost_test_probs.csv')
        
        print(f"LightGBM验证集概率: {len(lgb_val)} 样本")
        print(f"CatBoost验证集概率: {len(cat_val)} 样本")
        print(f"LightGBM测试集概率: {len(lgb_test)} 样本")
        print(f"CatBoost测试集概率: {len(cat_test)} 样本")
        
        print("⚠️ 使用双模型融合")
        return lgb_val, cat_val, None, lgb_test, cat_test, None, 'basic'
    else:
        print("❌ 缺少概率文件")
        return None, None, None, None, None, None, None

def weighted_average_ensemble(lgb_val, cat_val, lgb_test, cat_test, model_type='basic', weights=None):
    """加权平均融合（支持WBCE和基础模型）"""
    print(f"\n=== 加权平均融合 ({model_type.upper()}) ===")
    
    # 加载训练数据以获取标签和权重信息
    train_data = pd.read_csv('train_processed.csv')
    
    # 根据模型类型自动检测列名
    if model_type == 'wbce':
        lgb_col = 'lgb_wbce_prob' if 'lgb_wbce_prob' in lgb_val.columns else 'lightgbm_wbce_prob'
        cat_col = 'catboost_wbce_prob' if 'catboost_wbce_prob' in cat_val.columns else 'catboost_wbce_prob'
    else:
        lgb_col = 'lightgbm_prob' if 'lightgbm_prob' in lgb_val.columns else 'lightgbm_enhanced_prob'
        cat_col = 'catboost_prob' if 'catboost_prob' in cat_val.columns else 'catboost_enhanced_prob'
    
    # 双模型融合 - 自动优化权重
    if weights is None:
        print("🔍 自动优化双模型权重...")
        # 从训练数据中获取对应的标签和权重
        val_indices = lgb_val['ID'].values
        target_sign = train_data.loc[train_data['ID'].isin(val_indices), 'target_sign'].values
        ret_target = train_data.loc[train_data['ID'].isin(val_indices), 'RET_TARGET'].values
        weights = optimize_weights(lgb_val[lgb_col], cat_val[cat_col], target_sign, ret_target)
    
    print(f"使用优化权重: LightGBM={weights[0]:.3f}, CatBoost={weights[1]:.3f}")
    print(f"使用列名: {lgb_col}, {cat_col}")
    
    # 验证集加权平均
    val_probs = (weights[0] * lgb_val[lgb_col] + 
                 weights[1] * cat_val[cat_col])
    
    # 测试集加权平均
    test_probs = (weights[0] * lgb_test[lgb_col] + 
                  weights[1] * cat_test[cat_col])
    
    return val_probs, test_probs

def optimize_weights(lgb_probs, cat_probs, target_sign, ret_target):
    """智能权重优化 - 最后一搏策略"""
    print("🔍 开始智能权重优化...")
    
    # 准备样本权重
    sample_weights = np.abs(ret_target)
    
    # 🚀 策略1: 基于模型性能的动态权重
    lgb_score = 0
    cat_score = 0
    
    # 计算每个模型的单独性能
    for thresh in np.arange(0.1, 0.9, 0.01):
        lgb_pred = (lgb_probs > thresh).astype(int) * 2 - 1
        cat_pred = (cat_probs > thresh).astype(int) * 2 - 1
        
        lgb_curr_score = qrt_score(target_sign * 2 - 1, lgb_pred, sample_weights)
        cat_curr_score = qrt_score(target_sign * 2 - 1, cat_pred, sample_weights)
        
        if lgb_curr_score > lgb_score:
            lgb_score = lgb_curr_score
        if cat_curr_score > cat_score:
            cat_score = cat_curr_score
    
    # 基于性能的权重
    total_score = lgb_score + cat_score
    if total_score > 0:
        perf_weights = [lgb_score / total_score, cat_score / total_score]
    else:
        perf_weights = [0.5, 0.5]
    
    # 🚀 策略2: 精细网格搜索
    best_score = -1
    best_weights = perf_weights  # 从性能权重开始
    
    # 更精细的网格搜索
    for w1 in np.arange(0.0, 1.01, 0.02):  # 更精细的步长
        w2 = 1.0 - w1
        if w2 < 0:
            continue
            
        # 加权平均
        ensemble_probs = w1 * lgb_probs + w2 * cat_probs
        
        # 调优阈值
        best_thresh = 0.5
        best_model_score = -1
        for thresh in np.arange(0.1, 0.9, 0.01):
            ensemble_pred = (ensemble_probs > thresh).astype(int) * 2 - 1
            score = qrt_score(target_sign * 2 - 1, ensemble_pred, sample_weights)
            if score > best_model_score:
                best_model_score = score
                best_thresh = thresh
        
        if best_model_score > best_score:
            best_score = best_model_score
            best_weights = [w1, w2]
    
    print(f"✅ 最优权重: {best_weights}, QRT Score: {best_score:.4f}")
    print(f"📊 模型性能: LightGBM={lgb_score:.4f}, CatBoost={cat_score:.4f}")
    return best_weights

def advanced_stacking(lgb_val, cat_val, lgb_test, cat_test, train_data, model_type='basic'):
    """高级Stacking融合（支持WBCE和基础模型）- 创新性优化"""
    print(f"\n=== 高级Stacking融合 ({model_type.upper()}) - 创新性优化 ===")
    
    # 根据模型类型自动检测列名
    if model_type == 'wbce':
        lgb_col = 'lgb_wbce_prob' if 'lgb_wbce_prob' in lgb_val.columns else 'lightgbm_wbce_prob'
        cat_col = 'catboost_wbce_prob' if 'catboost_wbce_prob' in cat_val.columns else 'catboost_wbce_prob'
    else:
        lgb_col = 'lightgbm_prob' if 'lightgbm_prob' in lgb_val.columns else 'lightgbm_enhanced_prob'
        cat_col = 'catboost_prob' if 'catboost_prob' in cat_val.columns else 'catboost_enhanced_prob'
    
    # 🚀 创新性特征工程 - 添加交互特征和非线性变换
    lgb_probs = lgb_val[lgb_col].values
    cat_probs = cat_val[cat_col].values
    
    # 基础特征
    X_train = np.column_stack([
        lgb_probs,
        cat_probs
    ])
    
    # 🎯 创新性特征工程
    # 1. 交互特征
    interaction_feature = lgb_probs * cat_probs
    
    # 2. 差异特征
    diff_feature = lgb_probs - cat_probs
    
    # 3. 比率特征
    ratio_feature = np.where(cat_probs > 0, lgb_probs / (cat_probs + 1e-8), 0)
    
    # 4. 非线性变换
    log_lgb = np.log(np.clip(lgb_probs, 1e-8, 1-1e-8))
    log_cat = np.log(np.clip(cat_probs, 1e-8, 1-1e-8))
    
    # 5. 多项式特征
    lgb_squared = lgb_probs ** 2
    cat_squared = cat_probs ** 2
    
    # 6. 统计特征
    mean_probs = (lgb_probs + cat_probs) / 2
    std_probs = np.sqrt(((lgb_probs - mean_probs) ** 2 + (cat_probs - mean_probs) ** 2) / 2)
    
    # 组合所有特征
    X_train = np.column_stack([
        lgb_probs, cat_probs,  # 原始概率
        interaction_feature,    # 交互特征
        diff_feature,          # 差异特征
        ratio_feature,         # 比率特征
        log_lgb, log_cat,      # 对数变换
        lgb_squared, cat_squared,  # 多项式特征
        mean_probs, std_probs      # 统计特征
    ])
    
    # 对测试集应用相同的特征工程
    lgb_test_probs = lgb_test[lgb_col].values
    cat_test_probs = cat_test[cat_col].values
    
    interaction_test = lgb_test_probs * cat_test_probs
    diff_test = lgb_test_probs - cat_test_probs
    ratio_test = np.where(cat_test_probs > 0, lgb_test_probs / (cat_test_probs + 1e-8), 0)
    log_lgb_test = np.log(np.clip(lgb_test_probs, 1e-8, 1-1e-8))
    log_cat_test = np.log(np.clip(cat_test_probs, 1e-8, 1-1e-8))
    lgb_squared_test = lgb_test_probs ** 2
    cat_squared_test = cat_test_probs ** 2
    mean_test = (lgb_test_probs + cat_test_probs) / 2
    std_test = np.sqrt(((lgb_test_probs - mean_test) ** 2 + (cat_test_probs - mean_test) ** 2) / 2)
    
    X_test = np.column_stack([
        lgb_test_probs, cat_test_probs,
        interaction_test, diff_test, ratio_test,
        log_lgb_test, log_cat_test,
        lgb_squared_test, cat_squared_test,
        mean_test, std_test
    ])

    print(f"🚀 创新性特征工程: 原始2维 → 增强10维特征")
    print(f"特征包括: 原始概率、交互、差异、比率、对数、多项式、统计特征")
    
    # 准备标签和样本权重
    y_train = train_data.loc[lgb_val['ID'], 'target_sign'].values
    y_train = (y_train > 0).astype(int)
    sample_weights = train_data.loc[lgb_val['ID'], 'RET_TARGET'].abs().values
    
    # 训练多个Stacking模型 - 创新性元模型组合
    models = {}
    
    # 1. Logistic Regression - 基础线性模型
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['LogisticRegression'] = lr_model
    
    # 2. Ridge Regression - 正则化线性模型
    ridge_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['Ridge'] = ridge_model
    
    # 3. Lasso Regression - 稀疏线性模型
    lasso_model = LassoCV(cv=5, random_state=42, max_iter=1000, alphas=[0.001, 0.01, 0.1, 1.0])
    lasso_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['Lasso'] = lasso_model
    
    # 4. Elastic Net - 弹性网络 (简化版本，避免收敛问题)
    from sklearn.linear_model import ElasticNet
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
    elastic_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['ElasticNet'] = elastic_model
    
    # 5. 支持向量机 - 非线性模型 (简化版本，避免长时间训练)
    from sklearn.svm import SVC
    svm_model = SVC(kernel='rbf', probability=True, random_state=42, C=0.1, gamma='scale', max_iter=1000)
    svm_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['SVM'] = svm_model
    
    # 预测并评估 - 智能预测处理
    results = {}
    for name, model in models.items():
        # 验证集预测
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(X_train)[:, 1]
        else:
            val_probs = model.predict(X_train)
            # 对于回归模型，需要将输出转换为概率
            if name in ['Ridge', 'Lasso', 'ElasticNet']:
                val_probs = 1 / (1 + np.exp(-val_probs))  # sigmoid转换
            val_probs = np.clip(val_probs, 0, 1)  # 确保概率在[0,1]范围内
        
        # 测试集预测
        if hasattr(model, 'predict_proba'):
            test_probs = model.predict_proba(X_test)[:, 1]
        else:
            test_probs = model.predict(X_test)
            # 对于回归模型，需要将输出转换为概率
            if name in ['Ridge', 'Lasso', 'ElasticNet']:
                test_probs = 1 / (1 + np.exp(-test_probs))  # sigmoid转换
            test_probs = np.clip(test_probs, 0, 1)  # 确保概率在[0,1]范围内
        
        results[name] = {
            'val_probs': val_probs,
            'test_probs': test_probs,
            'model': model
        }
        
        # 打印模型系数 - 智能系数处理
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim == 2:
                # 对于多特征模型，只显示前两个系数
                if coef.shape[1] >= 2:
                    lgb_coef = float(coef[0, 0])
                    cat_coef = float(coef[0, 1])
                else:
                    lgb_coef = float(coef[0, 0]) if coef.shape[1] > 0 else 0
                    cat_coef = 0
            else:
                # 1D数组
                if len(coef) >= 2:
                    lgb_coef = float(coef[0])
                    cat_coef = float(coef[1])
                else:
                    lgb_coef = float(coef[0]) if len(coef) > 0 else 0
                    cat_coef = 0
            print(f"  {name} - 系数: LightGBM={lgb_coef:.4f}, CatBoost={cat_coef:.4f}")
        elif hasattr(model, 'alpha_'):
            coef = model.coef_
            if coef.ndim == 2:
                lgb_coef = float(coef[0, 0]) if coef.shape[1] > 0 else 0
                cat_coef = float(coef[0, 1]) if coef.shape[1] > 1 else 0
            else:
                lgb_coef = float(coef[0]) if len(coef) > 0 else 0
                cat_coef = float(coef[1]) if len(coef) > 1 else 0
            print(f"  {name} - Alpha: {model.alpha_:.4f}, 系数: LightGBM={lgb_coef:.4f}, CatBoost={cat_coef:.4f}")
        else:
            print(f"  {name} - 非线性模型，无线性系数")
            
    # 选择最佳模型（基于验证集QRT Score）
    best_score = -1
    best_model_name = None
    best_val_probs = None
    best_test_probs = None
    
    for name, result in results.items():
        val_probs = result['val_probs']
        
        # 调优阈值
        best_thresh = 0.5
        best_model_score = -1
        for thresh in np.arange(0.1, 0.9, 0.01):
            val_pred = (val_probs > thresh).astype(int) * 2 - 1
            score = qrt_score(y_train * 2 - 1, val_pred, sample_weights)
            if score > best_model_score:
                best_model_score = score
                best_thresh = thresh
            
        print(f"  {name} - 验证集QRT Score: {best_model_score:.4f}")
            
        if best_model_score > best_score:
            best_score = best_model_score
            best_model_name = name
            best_val_probs = val_probs
            best_test_probs = result['test_probs']
    
    print(f"\n🎯 最佳模型: {best_model_name} (QRT Score: {best_score:.4f})")
    
    # 🚀 创新性集成投票机制
    print(f"\n🚀 创新性集成投票机制")
    
    # 计算所有模型的加权平均
    all_val_probs = []
    all_test_probs = []
    model_scores = []
    
    for name, result in results.items():
        val_probs = result['val_probs']
        
        # 计算每个模型的QRT Score
        best_thresh = 0.5
        best_model_score = -1
        for thresh in np.arange(0.1, 0.9, 0.01):
            val_pred = (val_probs > thresh).astype(int) * 2 - 1
            score = qrt_score(y_train * 2 - 1, val_pred, sample_weights)
            if score > best_model_score:
                best_model_score = score
                best_thresh = thresh
        
        all_val_probs.append(val_probs)
        all_test_probs.append(result['test_probs'])
        model_scores.append(best_model_score)
    
    # 基于性能的加权平均
    model_scores = np.array(model_scores)
    model_weights = model_scores / np.sum(model_scores) if np.sum(model_scores) > 0 else np.ones(len(model_scores)) / len(model_scores)
    
    # 计算集成投票结果
    ensemble_val_probs = np.zeros_like(all_val_probs[0])
    ensemble_test_probs = np.zeros_like(all_test_probs[0])
    
    for i, (val_probs, test_probs, weight) in enumerate(zip(all_val_probs, all_test_probs, model_weights)):
        ensemble_val_probs += weight * val_probs
        ensemble_test_probs += weight * test_probs
    
    # 评估集成投票结果
    best_thresh = 0.5
    best_ensemble_score = -1
    for thresh in np.arange(0.1, 0.9, 0.01):
        ensemble_pred = (ensemble_val_probs > thresh).astype(int) * 2 - 1
        score = qrt_score(y_train * 2 - 1, ensemble_pred, sample_weights)
        if score > best_ensemble_score:
            best_ensemble_score = score
            best_thresh = thresh
    
    print(f"🎯 集成投票QRT Score: {best_ensemble_score:.4f}")
    print(f"📊 模型权重: {dict(zip(results.keys(), model_weights))}")
    
    # 选择最佳结果
    if best_ensemble_score > best_score:
        print(f"✅ 集成投票优于最佳单模型，使用集成结果")
        return ensemble_val_probs, ensemble_test_probs, "EnsembleVoting"
    else:
        print(f"✅ 最佳单模型优于集成投票，使用单模型结果")
        return best_val_probs, best_test_probs, best_model_name

def logistic_regression_stacking(lgb_val, cat_val, lgb_test, cat_test, train_data, model_type='basic'):
    """Logistic Regression Stacking（支持WBCE和基础模型）"""
    print(f"\n=== Logistic Regression Stacking ({model_type.upper()}) ===")
    
    # 根据模型类型自动检测列名
    if model_type == 'wbce':
        lgb_col = 'lgb_wbce_prob' if 'lgb_wbce_prob' in lgb_val.columns else 'lightgbm_wbce_prob'
        cat_col = 'catboost_wbce_prob' if 'catboost_wbce_prob' in cat_val.columns else 'catboost_wbce_prob'
    else:
        lgb_col = 'lightgbm_prob' if 'lightgbm_prob' in lgb_val.columns else 'lightgbm_enhanced_prob'
        cat_col = 'catboost_prob' if 'catboost_prob' in cat_val.columns else 'catboost_enhanced_prob'
    
    # 准备训练数据
    X_train = np.column_stack([
        lgb_val[lgb_col].values,
        cat_val[cat_col].values
    ])
    
    X_test = np.column_stack([
        lgb_test[lgb_col].values,
        cat_test[cat_col].values
    ])
    
    # 准备标签和权重
    y_train = train_data.loc[lgb_val['ID'], 'target_sign'].values
    y_train = (y_train > 0).astype(int)
    sample_weights = train_data.loc[lgb_val['ID'], 'RET_TARGET'].abs().values
    
    # 训练Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 预测
    val_probs = lr_model.predict_proba(X_train)[:, 1]
    test_probs = lr_model.predict_proba(X_test)[:, 1]
    
    print(f"Logistic Regression系数: LightGBM={float(lr_model.coef_[0][0]):.4f}, CatBoost={float(lr_model.coef_[0][1]):.4f}")
    
    return val_probs, test_probs

def evaluate_ensemble(val_probs, train_data, val_indices, method_name):
    """评估集成模型"""
    print(f"\n=== 评估 {method_name} ===")
    
    # 获取验证集标签和权重
    y_val = train_data.loc[val_indices, 'target_sign'].values
    y_val = (y_val > 0).astype(int)
    val_weights = train_data.loc[val_indices, 'RET_TARGET'].abs().values
    
    # 调优阈值
    best_score = -1
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        val_pred = (val_probs > thresh).astype(int) * 2 - 1
        score = qrt_score(y_val * 2 - 1, val_pred, val_weights)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    print(f"最优阈值: {best_thresh:.2f}")
    print(f"验证集QRT Score: {best_score:.4f}")
    
    return best_score, best_thresh
    
def create_submission(test_probs, threshold, method_name, model_type='basic'):
    """创建提交文件"""
    test_pred = (test_probs > threshold).astype(int) * 2 - 1
    
    # 读取测试集ID
    test = pd.read_csv('test_processed.csv')
    
    submit = pd.DataFrame({
        'ID': test['ID'],
        'RET_TARGET': test_pred
    })
    
    # 根据模型类型命名文件
    if model_type == 'wbce':
        filename = f'submit_{method_name}_wbce.csv'
    else:
        filename = f'submit_{method_name}.csv'
    
    submit.to_csv(filename, index=False)
    print(f"提交文件已保存: {filename}")
    
    return filename

def plot_probability_comparison(lgb_val, cat_val, val_probs_weighted, val_probs_stacking, model_type='basic'):
    """绘制概率对比图"""
    try:
        plt.figure(figsize=(15, 5))
        
        # 根据模型类型选择列名
        if model_type == 'wbce':
            lgb_col = 'lgb_wbce_prob' if 'lgb_wbce_prob' in lgb_val.columns else 'lightgbm_wbce_prob'
            cat_col = 'catboost_wbce_prob' if 'catboost_wbce_prob' in cat_val.columns else 'catboost_wbce_prob'
        else:
            lgb_col = 'lightgbm_prob' if 'lightgbm_prob' in lgb_val.columns else 'lightgbm_enhanced_prob'
            cat_col = 'catboost_prob' if 'catboost_prob' in cat_val.columns else 'catboost_enhanced_prob'
        
        # 子图1: 原始模型概率分布
        plt.subplot(1, 3, 1)
        plt.hist(lgb_val[lgb_col], bins=50, alpha=0.7, label='LightGBM', density=True)
        plt.hist(cat_val[cat_col], bins=50, alpha=0.7, label='CatBoost', density=True)
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.title('Original Model Probabilities')
        plt.legend()
        
        # 子图2: 加权平均概率分布
        plt.subplot(1, 3, 2)
        plt.hist(val_probs_weighted, bins=50, alpha=0.7, color='green', density=True)
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.title('Weighted Average Probabilities')
        
        # 子图3: Stacking概率分布
        plt.subplot(1, 3, 3)
        plt.hist(val_probs_stacking, bins=50, alpha=0.7, color='red', density=True)
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.title('Stacking Probabilities')
        
        plt.tight_layout()
        
        # 根据模型类型命名文件
        if model_type == 'wbce':
            filename = 'ensemble_probability_comparison_wbce.png'
        else:
            filename = 'ensemble_probability_comparison.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"概率对比图已保存: {filename}")
        
    except Exception as e:
        print(f"绘图失败: {e}")

def main():
    """主函数"""
    print("🚀 高级Stacking集成学习")
    print("=" * 50)
    
    # 加载概率文件
    lgb_val, cat_val, _, lgb_test, cat_test, _, model_type = load_probabilities()
    
    if lgb_val is None:
        print("❌ 无法加载概率文件，请先训练模型")
        return
    
    # 加载训练数据用于评估
    train_data = pd.read_csv('train_processed.csv')
    val_indices = lgb_val['ID'].values
    
    print(f"\n📊 使用模型类型: {model_type.upper()}")
    
    # 1. 加权平均融合
    val_probs_weighted, test_probs_weighted = weighted_average_ensemble(
        lgb_val, cat_val, lgb_test, cat_test, model_type
    )
    
    weighted_score, weighted_thresh = evaluate_ensemble(
        val_probs_weighted, train_data, val_indices, "加权平均"
    )
    
    create_submission(test_probs_weighted, weighted_thresh, "加权平均", model_type)
    
    # 2. 高级Stacking融合
    val_probs_stacking, test_probs_stacking, best_model_name = advanced_stacking(
        lgb_val, cat_val, lgb_test, cat_test, train_data, model_type
    )
    
    stacking_score, stacking_thresh = evaluate_ensemble(
        val_probs_stacking, train_data, val_indices, f"高级Stacking({best_model_name})"
    )
    
    create_submission(test_probs_stacking, stacking_thresh, "高级stacking", model_type)
    
    # 3. 双模型Logistic Regression Stacking
    val_probs_lr, test_probs_lr = logistic_regression_stacking(
        lgb_val, cat_val, lgb_test, cat_test, train_data, model_type
    )
    
    lr_score, lr_thresh = evaluate_ensemble(
        val_probs_lr, train_data, val_indices, "LogisticRegression"
    )
    
    create_submission(test_probs_lr, lr_thresh, "logistic_regression", model_type)
    
    # 4. 绘制概率对比图
    plot_probability_comparison(lgb_val, cat_val, val_probs_weighted, val_probs_stacking, model_type)
    
    # 5. 总结结果
    print("\n" + "=" * 50)
    print("🎯 集成学习结果总结")
    print("=" * 50)
    
    methods = [
        ("加权平均", weighted_score),
        (f"高级Stacking({best_model_name})", stacking_score),
        ("LogisticRegression", lr_score)
    ]
    
    for method, score in methods:
        print(f"  {method}: QRT Score = {score:.4f}")
    
    best_method, best_score = max(methods, key=lambda x: x[1])
    print(f"\n🏆 最佳方法: {best_method} (QRT Score: {best_score:.4f})")
    
    if model_type == 'wbce':
        print("✅ 使用WBCE优化模型完成")
    else:
        print("✅ 使用基础模型完成")
    
    print("\n🎉 所有集成方法完成！")

if __name__ == "__main__":
    main() 