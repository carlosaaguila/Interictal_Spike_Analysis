#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
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

# Set default output directory to current working directory
default_output_dir = os.path.join(os.getcwd(), "ML_results", "LR_elastic", "test_noloocv")

# Allow custom output directory via environment variable
output_dir = os.environ.get("ML_OUTPUT_DIR", default_output_dir)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create and open log file
log_file = os.path.join(output_dir, f"5fold-cv-LR-24m-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    sys.stdout = open(log_file, 'w')
except IOError as e:
    print(f"Error: Unable to create or write to log file. {e}")
    sys.exit(1)

print(f"Log file created at: {log_file}")

# Load and preprocess data
def load_data():
    outcomes = pd.read_csv('dataset/ML_data/features_and_outcomes.csv', index_col=0)

    Engel_good = ['IA','IB','IC','ID']

    def calculate_outcome(row):
        if pd.isna(row['engel_f1']):
            return row[' ILAE-v1']
        elif str(row['engel_f1']) in Engel_good:
            return 1
        else:
            return 0

    # Apply the function to each row to create the outcome3 column
    outcomes['engel_outcomes_12m'] = outcomes.apply(calculate_outcome, axis=1)

    def calculate_outcome_2yr(row):
        if pd.isna(row['engel_f2']):
            return row[' ILAE-v1']
        elif str(row['engel_f2']) in Engel_good:
            return 1
        else:
            return 0

    outcomes['engel_outcomes_24m'] = outcomes.apply(calculate_outcome_2yr, axis=1)

    combined_df = outcomes.drop(columns=['Patient_ID','Predicted_Label','combined_predprob','rid','hup_id','engel_f1', 'engel_f2', 'ilae_f1', 'ilae_f2',
       ' ILAE-v1', 'ILAE-v2','engel_outcomes_12m','SOZ','True_Label'])

    EI_df = combined_df[['correlation','engel_outcomes_24m','pt_id']]
    pearson_df = outcomes.drop(columns=['Patient_ID','Predicted_Label','combined_predprob','rid','hup_id','engel_f1', 'engel_f2', 'ilae_f1', 'ilae_f2',
       ' ILAE-v1', 'ILAE-v2','engel_outcomes_12m','SOZ','True_Label','correlation'])
    return combined_df, EI_df, pearson_df

# Bootstrap ROC curve
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

# Plot ROC curve with confidence interval
def plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color):
    plt.fill_between(bootstrapped_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

# Perform 5-fold cross-validation with Logistic Regression Elastic Net
def five_fold_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true, y_pred = [], []
    feature_importances = []
    pt_ids = []
    
    scaler = StandardScaler()
    
    for train_ix, test_ix in skf.split(X, y):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        pt_ids.extend(X_test['pt_id'].values)
        
        X_train = X_train.drop(columns=['engel_outcomes_24m', 'pt_id'])
        X_test = X_test.drop(columns=['engel_outcomes_24m', 'pt_id'])
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize Logistic Regression with Elastic Net
        model = LogisticRegressionCV(
            cv=5,  # 5-fold cross-validation for hyperparameter tuning
            random_state=42,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=np.linspace(0, 1, 10),  # This creates an array of 10 values from 0 to 1
            Cs=np.logspace(-4, 4, 20),  # This creates an array of 20 values from 10^-4 to 10^4
            max_iter=10000  # Increase max_iter to ensure convergence
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred.extend(model.predict_proba(X_test_scaled)[:, 1])
        y_true.extend(y_test.to_numpy())
        
        # For Logistic Regression, we use the coefficients as feature importances
        feature_importances.append(np.abs(model.coef_[0]))
    
    return np.array(y_true), np.array(y_pred), np.mean(feature_importances, axis=0), pt_ids

# Main execution
def main():
    plt.rcParams['font.family'] = 'Arial'
    
    combined_data, ictal_data, interictal_data = load_data()
    
    # Prepare datasets
    combined_features = combined_data.drop(columns=['engel_outcomes_24m', 'pt_id'])
    interictal_features = interictal_data.drop(columns=['engel_outcomes_24m', 'pt_id'])
    ictal_features = combined_data[['correlation', 'engel_outcomes_24m', 'pt_id']].drop(columns=['engel_outcomes_24m', 'pt_id'])
    
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
        X = pd.concat([features, data[['engel_outcomes_24m', 'pt_id']]], axis=1)
        y_true, y_pred, feature_importance, pt_ids = five_fold_cv(X, data['engel_outcomes_24m'])
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC confidence interval
        ci_lower, ci_upper = calculate_auc_ci(y_true, y_pred)
        
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        
        bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(y_true, y_pred)
        plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color)
        
        # Print AUC with confidence interval
        print(f"\nResults for {name}:")
        print(f"AUC: {roc_auc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
        
        # Print and save feature importance
        print(f"\nFeature Importance for {name}:")
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print(importance_df)
        importance_df.to_csv(os.path.join(output_dir, f"{name}_feature_importance-24m-outcomes.csv"), index=False)
        
        # Save predictions vs. truth labels
        predictions_df = pd.DataFrame({
            'Patient_ID': pt_ids,
            'True_Label': y_true,
            'Predicted_Probability': y_pred
        })
        predictions_df.to_csv(os.path.join(output_dir, f"{name}_predictions-24m-outcomes.csv"), index=False)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curves for Classifying MTLE', fontsize=24, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Manually set the weight to bold for tick labels
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LR_elastic_predict_24m-outcomes.pdf'))
    plt.close()

if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {output_dir}")


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
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

# Set default output directory to current working directory
default_output_dir = os.path.join(os.getcwd(), "ML_results", "LR_elastic", "test_noloocv")

# Allow custom output directory via environment variable
output_dir = os.environ.get("ML_OUTPUT_DIR", default_output_dir)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create and open log file
log_file = os.path.join(output_dir, f"5fold-cv-LR-12m-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    sys.stdout = open(log_file, 'w')
except IOError as e:
    print(f"Error: Unable to create or write to log file. {e}")
    sys.exit(1)

print(f"Log file created at: {log_file}")

# Load and preprocess data
def load_data():
    outcomes = pd.read_csv('dataset/ML_data/features_and_outcomes.csv', index_col=0)

    Engel_good = ['IA','IB','IC','ID']

    def calculate_outcome(row):
        if pd.isna(row['engel_f1']):
            return row[' ILAE-v1']
        elif str(row['engel_f1']) in Engel_good:
            return 1
        else:
            return 0

    # Apply the function to each row to create the outcome3 column
    outcomes['engel_outcomes_12m'] = outcomes.apply(calculate_outcome, axis=1)

    def calculate_outcome_2yr(row):
        if pd.isna(row['engel_f2']):
            return row[' ILAE-v1']
        elif str(row['engel_f2']) in Engel_good:
            return 1
        else:
            return 0

    outcomes['engel_outcomes_24m'] = outcomes.apply(calculate_outcome_2yr, axis=1)

    combined_df = outcomes.drop(columns=['Patient_ID','Predicted_Label','combined_predprob','rid','hup_id','engel_f1', 'engel_f2', 'ilae_f1', 'ilae_f2',
       ' ILAE-v1', 'ILAE-v2','engel_outcomes_24m','SOZ','True_Label'])

    EI_df = combined_df[['correlation','engel_outcomes_12m','pt_id']]
    pearson_df = outcomes.drop(columns=['Patient_ID','Predicted_Label','combined_predprob','rid','hup_id','engel_f1', 'engel_f2', 'ilae_f1', 'ilae_f2',
       ' ILAE-v1', 'ILAE-v2','engel_outcomes_24m','SOZ','True_Label','correlation'])
    return combined_df, EI_df, pearson_df

# Bootstrap ROC curve
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

# Plot ROC curve with confidence interval
def plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color):
    plt.fill_between(bootstrapped_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

# Perform 5-fold cross-validation with Logistic Regression Elastic Net
def five_fold_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true, y_pred = [], []
    feature_importances = []
    pt_ids = []
    
    scaler = StandardScaler()
    
    for train_ix, test_ix in skf.split(X, y):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        pt_ids.extend(X_test['pt_id'].values)
        
        X_train = X_train.drop(columns=['engel_outcomes_12m', 'pt_id'])
        X_test = X_test.drop(columns=['engel_outcomes_12m', 'pt_id'])
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize Logistic Regression with Elastic Net
        model = LogisticRegressionCV(
            cv=5,  # 5-fold cross-validation for hyperparameter tuning
            random_state=42,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=np.linspace(0, 1, 10),  # This creates an array of 10 values from 0 to 1
            Cs=np.logspace(-4, 4, 20),  # This creates an array of 20 values from 10^-4 to 10^4
            max_iter=10000  # Increase max_iter to ensure convergence
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred.extend(model.predict_proba(X_test_scaled)[:, 1])
        y_true.extend(y_test.to_numpy())
        
        # For Logistic Regression, we use the coefficients as feature importances
        feature_importances.append(np.abs(model.coef_[0]))
    
    return np.array(y_true), np.array(y_pred), np.mean(feature_importances, axis=0), pt_ids

# Main execution
def main():
    plt.rcParams['font.family'] = 'Arial'
    
    combined_data, ictal_data, interictal_data = load_data()
    
    # Prepare datasets
    combined_features = combined_data.drop(columns=['engel_outcomes_12m', 'pt_id'])
    interictal_features = interictal_data.drop(columns=['engel_outcomes_12m', 'pt_id'])
    ictal_features = combined_data[['correlation', 'engel_outcomes_12m', 'pt_id']].drop(columns=['engel_outcomes_12m', 'pt_id'])
    
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
        X = pd.concat([features, data[['engel_outcomes_12m', 'pt_id']]], axis=1)
        y_true, y_pred, feature_importance, pt_ids = five_fold_cv(X, data['engel_outcomes_12m'])
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC confidence interval
        ci_lower, ci_upper = calculate_auc_ci(y_true, y_pred)
        
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        
        bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(y_true, y_pred)
        plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color)
        
        # Print AUC with confidence interval
        print(f"\nResults for {name}:")
        print(f"AUC: {roc_auc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
        
        # Print and save feature importance
        print(f"\nFeature Importance for {name}:")
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print(importance_df)
        importance_df.to_csv(os.path.join(output_dir, f"{name}_feature_importance-12m-outcomes.csv"), index=False)
        
        # Save predictions vs. truth labels
        predictions_df = pd.DataFrame({
            'Patient_ID': pt_ids,
            'True_Label': y_true,
            'Predicted_Probability': y_pred
        })
        predictions_df.to_csv(os.path.join(output_dir, f"{name}_predictions-12m-outcomes.csv"), index=False)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curves for Classifying MTLE', fontsize=24, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Manually set the weight to bold for tick labels
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LR_elastic_predict_12m-outcomes.pdf'))
    plt.close()

if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {output_dir}")

# %%
