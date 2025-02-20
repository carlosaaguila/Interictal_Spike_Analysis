import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample as sklearn_resample
from sklearn.preprocessing import StandardScaler
import os
import sys
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set default output directory
default_output_dir = os.path.join(os.getcwd(), "ML_results", "LR_elastic_diagonal")
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

def find_optimal_operating_point(y_true, y_pred_proba):
    """
    Find optimal operating point by maximizing distance from ROC curve to diagonal line.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    distances = np.abs(tpr - fpr) / np.sqrt(2)
    optimal_idx = np.argmax(distances)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    
    # Bootstrap confidence intervals
    n_bootstraps = 1000
    bootstrap_metrics = {
        'ppv': [], 'npv': [], 
        'sensitivity': [], 'specificity': []
    }
    
    for i in range(n_bootstraps):
        indices = sklearn_resample(np.arange(len(y_pred_proba)), random_state=42 + i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        y_pred_boot = (y_pred_proba[indices] >= optimal_threshold).astype(int)
        y_true_boot = y_true[indices]
        
        tp_boot = np.sum((y_pred_boot == 1) & (y_true_boot == 1))
        fp_boot = np.sum((y_pred_boot == 1) & (y_true_boot == 0))
        tn_boot = np.sum((y_pred_boot == 0) & (y_true_boot == 0))
        fn_boot = np.sum((y_pred_boot == 0) & (y_true_boot == 1))
        
        ppv_boot = tp_boot / (tp_boot + fp_boot) if (tp_boot + fp_boot) > 0 else 0
        npv_boot = tn_boot / (tn_boot + fn_boot) if (tn_boot + fn_boot) > 0 else 0
        sens_boot = tp_boot / (tp_boot + fn_boot) if (tp_boot + fn_boot) > 0 else 0
        spec_boot = tn_boot / (tn_boot + fp_boot) if (tn_boot + fp_boot) > 0 else 0
        
        bootstrap_metrics['ppv'].append(ppv_boot)
        bootstrap_metrics['npv'].append(npv_boot)
        bootstrap_metrics['sensitivity'].append(sens_boot)
        bootstrap_metrics['specificity'].append(spec_boot)

    ci_metrics = {}
    for metric in bootstrap_metrics:
        ci_metrics[f'{metric}_ci'] = np.percentile(bootstrap_metrics[metric], [2.5, 97.5])
    
    return {
        'optimal_threshold': optimal_threshold,
        'ppv': ppv,
        'npv': npv,
        'sensitivity': sensitivity,
        'specificity': specificity,
        **ci_metrics,
        'geometric_distance': distances[optimal_idx],
        'optimal_fpr': fpr[optimal_idx],
        'optimal_tpr': tpr[optimal_idx]
    }

def plot_optimal_point(fpr, tpr, results, color):
    """Add optimal operating point to ROC curve"""
    plt.plot([results['optimal_fpr']], [results['optimal_tpr']], 
             'o', color=color, markersize=10, alpha=0.8,
             label=f'Optimal point\n(thresh={results["optimal_threshold"]:.2f})')
    
    # Draw line from diagonal to optimal point
    x = results['optimal_fpr']
    y = results['optimal_tpr']
    plt.plot([x, x], [x, y], '--', color=color, alpha=0.5)

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
    # Set plotting parameters
    plt.rcParams['font.family'] = 'Arial'
    
    # Load all datasets
    combined_data, ictal_data, interictal_data = load_data()
    
    # Prepare feature sets
    combined_features = combined_data.drop(columns=['SOZ', 'pt_id'])
    interictal_features = interictal_data.drop(columns=['SOZ', 'pt_id'])
    ictal_features = combined_data[['correlation', 'SOZ', 'pt_id']].drop(columns=['SOZ', 'pt_id'])
    
    # Define datasets for analysis
    datasets = [
        ("Combined", combined_features, '#E64B35FF'),
        ("Interictal", interictal_features, '#7E6148FF'),
        ("Ictal", ictal_features, '#00A087FF')
    ]
    
    # Create figure for ROC curves
    plt.figure(figsize=(8, 8))
    
    # Process each dataset
    for name, features, color in datasets:
        # Select appropriate data based on dataset type
        if (name == "Combined") or (name == "Ictal"):
            data = combined_data
        else: 
            data = interictal_data
            
        # Prepare features and perform LOOCV
        X = pd.concat([features, data[['SOZ', 'pt_id']]], axis=1)
        y_true, y_pred, y_predprob, feature_importance, pt_ids = leave_one_out_cv(X, data['SOZ'])
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_predprob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC confidence intervals
        ci_lower, ci_upper = calculate_auc_ci(y_true, y_predprob)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        
        # Add confidence intervals to ROC curve
        bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(y_true, y_predprob)
        plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color)
        
        # Print AUC results
        print(f"\nResults for {name}:")
        print(f"AUC: {roc_auc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
        
        # For combined model, calculate and plot optimal operating point
        if name == "Combined":
            # Find optimal operating point using geometric method
            results = find_optimal_operating_point(y_true, y_predprob)
            
            # Add optimal point to plot
            plot_optimal_point(fpr, tpr, results, color)
            
            # Print detailed metrics for optimal point
            print("\nOptimal Operating Point Analysis (Geometric Method):")
            print(f"Optimal Threshold: {results['optimal_threshold']:.3f}")
            print(f"PPV: {results['ppv']:.3f} (95% CI: {results['ppv_ci'][0]:.3f} - {results['ppv_ci'][1]:.3f})")
            print(f"NPV: {results['npv']:.3f} (95% CI: {results['npv_ci'][0]:.3f} - {results['npv_ci'][1]:.3f})")
            print(f"Sensitivity: {results['sensitivity']:.3f} (95% CI: {results['sensitivity_ci'][0]:.3f} - {results['sensitivity_ci'][1]:.3f})")
            print(f"Specificity: {results['specificity']:.3f} (95% CI: {results['specificity_ci'][0]:.3f} - {results['specificity_ci'][1]:.3f})")
            print(f"Geometric Distance from Diagonal: {results['geometric_distance']:.3f}")
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'Metric': ['Optimal_Threshold', 'PPV', 'NPV', 'Sensitivity', 'Specificity'],
                'Value': [results['optimal_threshold'], results['ppv'], results['npv'], 
                         results['sensitivity'], results['specificity']],
                'CI_Lower': [np.nan, results['ppv_ci'][0], results['npv_ci'][0],
                            results['sensitivity_ci'][0], results['specificity_ci'][0]],
                'CI_Upper': [np.nan, results['ppv_ci'][1], results['npv_ci'][1],
                            results['sensitivity_ci'][1], results['specificity_ci'][1]]
            })
            metrics_df.to_csv(os.path.join(output_dir, 'combined_model_metrics_geometric.csv'), index=False)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'Feature': features.columns, 
            'Importance': feature_importance
        })
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
    
    # Finalize plot formatting
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curves for Classifying MTLE', fontsize=24, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Bold tick labels
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_weight('bold')
    
    # Finalize and save plot
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logistic_regression_ROC_curves_v2.pdf'))
    plt.close()

if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {output_dir}")