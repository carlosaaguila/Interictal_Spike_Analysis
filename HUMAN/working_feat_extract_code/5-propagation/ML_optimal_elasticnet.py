import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils import resample as sklearn_resample
from sklearn.preprocessing import StandardScaler
import os
import sys
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set default output directory
default_output_dir = os.path.join(os.getcwd(), "ML_results", "LR_elastic_optimal_code")
output_dir = os.environ.get("ML_OUTPUT_DIR", default_output_dir)
os.makedirs(output_dir, exist_ok=True)

# Create log file
log_file = os.path.join(output_dir, f"logistic_regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
sys.stdout = open(log_file, 'w')

def load_data():
    pearson_df = pd.read_csv('dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col=0)
    pearson_df['SOZ'] = pearson_df['SOZ'].replace({2: 0, 3: 0})
    pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)

    EI_df = pd.read_csv('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/ML_data/EI_pearson_final1.csv', index_col=0)[['correlation', 'pt_id']]
    EI_df['pt_id'] = EI_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '').astype(int)

    combined_df = pearson_df.merge(EI_df, on='pt_id')
    combined_df = combined_df.dropna(subset='correlation')

    return combined_df, EI_df, pearson_df

def calculate_metrics(y_true, y_pred_proba, threshold):
    """Calculate PPV and NPV at a given threshold"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return ppv, npv

def find_optimal_operating_point(y_true, y_pred_proba):
    """Find optimal operating point using F1 score and calculate metrics"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    ppv, npv = calculate_metrics(y_true, y_pred_proba, optimal_threshold)
    
    # Bootstrap confidence intervals
    n_bootstraps = 1000
    bootstrap_ppvs = []
    bootstrap_npvs = []
    
    for i in range(n_bootstraps):
        indices = sklearn_resample(np.arange(len(y_pred_proba)), random_state=42 + i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        ppv_boot, npv_boot = calculate_metrics(y_true[indices], y_pred_proba[indices], optimal_threshold)
        bootstrap_ppvs.append(ppv_boot)
        bootstrap_npvs.append(npv_boot)
    
    ppv_ci = np.percentile(bootstrap_ppvs, [2.5, 97.5])
    npv_ci = np.percentile(bootstrap_npvs, [2.5, 97.5])
    
    return {
        'optimal_threshold': optimal_threshold,
        'ppv': ppv,
        'npv': npv,
        'ppv_ci': ppv_ci,
        'npv_ci': npv_ci
    }

def bootstrap_roc(y_true, y_pred, n_bootstraps=2000, alpha=0.95):
    bootstrapped_tpr = []
    bootstrapped_fpr = np.linspace(0, 1, 100)
    boot_aucs = []
    
    for i in range(n_bootstraps):
        indices = sklearn_resample(np.arange(len(y_pred)), random_state=42 + i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        fpr_, tpr_, _ = roc_curve(y_true[indices], y_pred[indices])
        boot_aucs.append(auc(fpr_, tpr_))
        tpr_interp = np.interp(bootstrapped_fpr, fpr_, tpr_)
        bootstrapped_tpr.append(tpr_interp)

    bootstrapped_tpr = np.array(bootstrapped_tpr)
    sort_idx = np.argsort(boot_aucs)
    bootstrapped_tpr = bootstrapped_tpr[sort_idx]
    
    lower_percentile = ((1.0 - alpha) / 2.0)
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0))
    
    tpr_lower = bootstrapped_tpr[int(lower_percentile*n_bootstraps)]
    tpr_upper = bootstrapped_tpr[int(upper_percentile*n_bootstraps)]
    mean_tpr = np.mean(bootstrapped_tpr, axis=0)
    
    return bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper

def calculate_auc_ci(y_true, y_pred, n_bootstraps=2000, alpha=0.95):
    bootstrapped_aucs = []
    
    for i in range(n_bootstraps):
        indices = sklearn_resample(np.arange(len(y_pred)), random_state=42 + i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))
    
    sorted_aucs = np.sort(bootstrapped_aucs)
    ci_lower = sorted_aucs[int((1.0 - alpha) / 2.0 * len(sorted_aucs))]
    ci_upper = sorted_aucs[int((alpha + (1.0 - alpha) / 2.0) * len(sorted_aucs))]
    
    return ci_lower, ci_upper

def plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color):
    plt.fill_between(bootstrapped_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

def leave_one_out_cv(X, y):
    unique_ids = X['pt_id'].unique()
    np.random.shuffle(unique_ids)
    
    loo = LeaveOneOut()
    y_true, y_pred, y_predprob = [], [], []
    feature_importances = []
    pt_ids = []
    
    scaler = StandardScaler()
    
    for train_ix, test_ix in loo.split(unique_ids):
        X_train = X[X['pt_id'].isin(unique_ids[train_ix])]
        X_test = X[X['pt_id'].isin(unique_ids[test_ix])]
        y_train = X_train['SOZ']
        y_test = X_test['SOZ']
        
        pt_ids.extend(X_test['pt_id'].values)
        
        X_train = X_train.drop(columns=['SOZ', 'pt_id'])
        X_test = X_test.drop(columns=['SOZ', 'pt_id'])
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegressionCV(
            cv=5,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=np.linspace(0, 1, 10),
            Cs=np.logspace(-4, 4, 20),
            max_iter=10000
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred.append(model.predict(X_test_scaled))
        y_predprob.append(model.predict_proba(X_test_scaled)[:, 1])
        y_true.append(y_test.to_numpy())
        feature_importances.append(np.abs(model.coef_[0]))
    
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_predprob), np.mean(feature_importances, axis=0), pt_ids

def main():
    plt.rcParams['font.family'] = 'Arial'
    combined_data, ictal_data, interictal_data = load_data()
    
    combined_features = combined_data.drop(columns=['SOZ', 'pt_id'])
    interictal_features = interictal_data.drop(columns=['SOZ', 'pt_id'])
    ictal_features = combined_data[['correlation', 'SOZ', 'pt_id']].drop(columns=['SOZ', 'pt_id'])
    
    datasets = [
        ("Combined", combined_features, '#E64B35FF'),
        ("Interictal", interictal_features, '#7E6148FF'),
        ("Ictal", ictal_features, '#00A087FF')
    ]
    
    plt.figure(figsize=(8, 8))
    
    for name, features, color in datasets:
        if (name == "Combined") or (name == "Ictal"):
            data = combined_data
        else: 
            data = interictal_data
            
        X = pd.concat([features, data[['SOZ', 'pt_id']]], axis=1)
        y_true, y_pred, y_predprob, feature_importance, pt_ids = leave_one_out_cv(X, data['SOZ'])
        
        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, y_predprob)
        roc_auc = auc(fpr, tpr)
        ci_lower, ci_upper = calculate_auc_ci(y_true, y_predprob)
        
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        
        bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(y_true, y_predprob)
        plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color)
        
        print(f"\nResults for {name}:")
        print(f"AUC: {roc_auc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
        
        # Calculate operating point metrics for combined model
        if name == "Combined":
            results = find_optimal_operating_point(y_true, y_predprob)
            print("\nOptimal Operating Point Analysis for Combined Model:")
            print(f"Optimal Threshold: {results['optimal_threshold']:.3f}")
            print(f"PPV: {results['ppv']:.3f} (95% CI: {results['ppv_ci'][0]:.3f} - {results['ppv_ci'][1]:.3f})")
            print(f"NPV: {results['npv']:.3f} (95% CI: {results['npv_ci'][0]:.3f} - {results['npv_ci'][1]:.3f})")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Optimal_Threshold', 'PPV', 'NPV'],
                'Value': [results['optimal_threshold'], results['ppv'], results['npv']],
                'CI_Lower': [np.nan, results['ppv_ci'][0], results['npv_ci'][0]],
                'CI_Upper': [np.nan, results['ppv_ci'][1], results['npv_ci'][1]]
            })
            metrics_df.to_csv(os.path.join(output_dir, 'combined_model_metrics.csv'), index=False)
        
        # Save feature importance
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(os.path.join(output_dir, f"{name}_feature_importance.csv"), index=False)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Patient_ID': pt_ids,
            'True_Label': y_true,
            'Predicted_Label': y_pred,
            'Predicted_Probability': y_predprob
        })
        predictions_df.to_csv(os.path.join(output_dir, f"{name}_predictions.csv"), index=False)
    
    # Finalize and save ROC plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curves for Classifying MTLE', fontsize=24, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_weight('bold')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logistic_regression_ROC_curves_v2.pdf'))
    plt.close()

if __name__ == "__main__":
    main()