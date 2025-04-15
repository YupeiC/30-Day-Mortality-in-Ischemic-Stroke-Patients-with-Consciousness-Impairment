import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
import shap
import matplotlib.pyplot as plt
from joblib import dump, load 


def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    X = data.drop('death_within_icu_30days', axis=1)
    y = data['death_within_icu_30days']
    return X, y


def evaluate_model(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'ACC': acc,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
    return metrics


def train_with_cv(X, y, params, n_splits=10):
    cv_results = []
    predictions = np.zeros(len(X))
    probabilities = np.zeros(len(X))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params, verbose=False)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        predictions[val_idx] = y_pred
        probabilities[val_idx] = y_prob
        
        metrics = evaluate_model(y_val, y_pred, y_prob)
        metrics['fold'] = fold
        cv_results.append(metrics)
        
        print(f"Fold {fold}: ACC={metrics['ACC']:.3f}, AUC={metrics['AUC']:.3f}")
    
    return cv_results, predictions, probabilities

def plot_shap_summary(shap_values, features, output_path):
    """绘制SHAP蜂群图"""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_path}/shap_summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_bar(shap_values, features, output_path):
    """绘制SHAP条形图"""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{output_path}/shap_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_heatmap(shap_values, features, output_path):
    """绘制SHAP热力图"""
    plt.figure(figsize=(12, 8))
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=features.values,
        feature_names=features.columns
    )
    shap.plots.heatmap(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_path}/shap_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_force(explainer, shap_values, features, idx, output_path):
    """绘制个案力图"""
    plt.figure(figsize=(12, 4))
    shap.force_plot(explainer.expected_value, 
                   shap_values[idx], 
                   features.iloc[idx],
                   show=False,
                   matplotlib=True)
    plt.tight_layout()
    plt.savefig(f'{output_path}/shap_force_{idx}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_waterfall(explainer, shap_values, features, idx, output_path):
    """绘制个案瀑布图"""
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[idx],
                                       base_values=explainer.expected_value,
                                       data=features.iloc[idx],
                                       feature_names=features.columns),
                       show=False)
    plt.tight_layout()
    plt.savefig(f'{output_path}/shap_waterfall_{idx}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_head_scatter(shap_values, features, output_path):
    """绘制SHAP Head Scatter Plot"""
    # 获取特征重要性排序
    feature_importance = np.abs(shap_values).mean(0)
    feature_names = features.columns
    sorted_idx = np.argsort(feature_importance)
    top_features = feature_names[sorted_idx[-9:]]  
    
    # 为每个重要特征创建head scatter plot
    for i, feature in enumerate(top_features):
        plt.figure(figsize=(6, 4))
        
        # 获取该特征的SHAP值
        feature_shap = shap_values[:, features.columns.get_loc(feature)]
        
        # 排序SHAP值
        sorted_indices = np.argsort(feature_shap)
        sorted_shap = feature_shap[sorted_indices]
        
        # 创建归一化的索引作为x轴
        x = np.linspace(0, 1, len(sorted_shap))
        
        # 绘制散点图
        plt.plot(x, sorted_shap, 'r-', linewidth=1)
        plt.xlabel('SHAP rank (impact on model output)')
        plt.ylabel(f'SHAP value for {feature}')
        plt.title(f'SHAP Head Scatter Plot - {feature}')
        
        plt.text(0.05, plt.ylim()[0], 'Low', verticalalignment='bottom')
        plt.text(0.95, plt.ylim()[1], 'High', verticalalignment='top', horizontalalignment='right')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/shap_head_scatter_{i+1}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # 加载数据  数据集地址！！
    X, y = load_data('TaskData.tsv')
    
    # 设置模型参数
    params = {
        'learning_rate': 0.1,
        'depth': 6,
        'iterations': 300,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC'
    }
    
    # 训练模型
    cv_results, predictions, probabilities = train_with_cv(X, y, params)
    
    # 训练最终模型
    final_model = CatBoostClassifier(**params, verbose=False)
    final_model.fit(X, y)
    
    # 保存模型
    model_path = 'd:/Workspace/xianyu/CatBoost_323_400/web部署/CatBoost.pkl'
    dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # SHAP值计算
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    # 创建输出目录  更改路径！！！
    output_path = 'd:/Workspace/dcode/ML_Classif_Learner_r_185/shap_plots'
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # 绘制各种SHAP图
    plot_shap_summary(shap_values, X, output_path)
    
    plot_shap_bar(shap_values, X, output_path)
    
    plot_shap_heatmap(shap_values, X, output_path)

    plot_shap_head_scatter(shap_values, X, output_path)
    
    top_indices = np.argsort(probabilities)[-3:]
    for idx in top_indices:
        plot_shap_force(explainer, shap_values, X, idx, output_path)
        plot_shap_waterfall(explainer, shap_values, X, idx, output_path)
    
    # 选择预测概率最高的3个样本
    top_indices = np.argsort(probabilities)[-3:]
    for idx in top_indices:
        plot_shap_force(explainer, shap_values, X, idx, output_path)
        plot_shap_waterfall(explainer, shap_values, X, idx, output_path)
    
    # 计算并保存整体评估指标
    overall_metrics = evaluate_model(y, predictions, probabilities)
    
    # 保存结果
    results_df = pd.DataFrame({
        'death_within_icu_30days': y,
        'modelPred': predictions,
        'X0Probmodel': 1 - probabilities,
        'X1Probmodel': probabilities
    })
    results_df.to_csv('actual_prediction.tsv', sep='\t', index=True)
    
    # 2. 交叉验证详细结果
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv('model_resample_measures.tsv', sep='\t', index=False)
    
    # 3. 整体评估指标
    metrics_df = pd.DataFrame([overall_metrics]).T
    metrics_df.columns = ['model']
    metrics_df.index.name = 'Measure'
    metrics_df.to_csv('model_resample_aggregate_measures.tsv', sep='\t')
    
    print("\nOverall Performance:")
    print(f"ACC: {overall_metrics['ACC']:.3f}")
    print(f"AUC: {overall_metrics['AUC']:.3f}")
    print(f"F1: {overall_metrics['F1']:.3f}")
    
    print("\nSHAP plots have been saved to:", output_path)

if __name__ == "__main__":
    main()