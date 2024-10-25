import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample as sklearn_resample
from sklearn.preprocessing import StandardScaler
import os
import sys
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set up output directory
default_output_dir = os.path.join(os.getcwd(), "ML_results", "LR_elastic_erin_suggest")
output_dir = os.environ.get("ML_OUTPUT_DIR", default_output_dir)
os.makedirs(output_dir, exist_ok=True)

def load_data():
    pearson_df = pd.read_csv('dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col=0)
    pearson_df['SOZ'] = pearson_df['SOZ'].replace({2: 0, 3: 0})
    pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)

    EI_df = pd.read_csv('dataset/ML_data/EI_pearson_final1.csv', index_col=0)[['correlation', 'pt_id']]
    EI_df['pt_id'] = EI_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '').astype(int)

    combined_df = pearson_df.merge(EI_df, on='pt_id')
    combined_df = combined_df.dropna(subset='correlation')

    return combined_df, EI_df, pearson_df

def train_test_single_split(X, data, random_state):
    """Perform a single 70/30 split and return AUC"""
    # Get unique patient IDs
    unique_ids = X['pt_id'].unique()
    
    # Split at patient level
    train_ids, test_ids = train_test_split(
        unique_ids, 
        test_size=0.3, 
        random_state=random_state,
        stratify=data[data['pt_id'].isin(unique_ids)].groupby('pt_id')['SOZ'].first()
    )
    
    # Create train and test sets
    X_train = X[X['pt_id'].isin(train_ids)]
    X_test = X[X['pt_id'].isin(test_ids)]
    y_train = X_train['SOZ']
    y_test = X_test['SOZ']
    
    # Drop non-feature columns
    X_train = X_train.drop(columns=['SOZ', 'pt_id'])
    X_test = X_test.drop(columns=['SOZ', 'pt_id'])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    model = LogisticRegressionCV(
        cv=5,
        random_state=42,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=np.linspace(0, 1, 10),
        Cs=np.logspace(-4, 4, 20),
        max_iter=10000
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_predprob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate AUC
    return roc_auc_score(y_test, y_predprob)

def compute_model_differences(combined_data, n_iterations=1000):
    """Compute AUC differences between combined and ictal models over multiple iterations"""
    # Prepare datasets
    combined_features = combined_data.drop(columns=['SOZ', 'pt_id'])
    ictal_features = combined_data[['correlation', 'SOZ', 'pt_id']].drop(columns=['SOZ', 'pt_id'])
    
    # Initialize arrays to store results
    auc_differences = np.zeros(n_iterations)
    combined_aucs = np.zeros(n_iterations)
    ictal_aucs = np.zeros(n_iterations)
    
    # Run iterations
    for i in range(n_iterations):
        # Create combined model data
        X_combined = pd.concat([combined_features, combined_data[['SOZ', 'pt_id']]], axis=1)
        
        # Create ictal model data
        X_ictal = pd.concat([ictal_features, combined_data[['SOZ', 'pt_id']]], axis=1)
        
        # Get AUCs for both models using same random state for fair comparison
        combined_auc = train_test_single_split(X_combined, combined_data, random_state=i)
        ictal_auc = train_test_single_split(X_ictal, combined_data, random_state=i)
        
        # Store results
        auc_differences[i] = combined_auc - ictal_auc
        combined_aucs[i] = combined_auc
        ictal_aucs[i] = ictal_auc
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Completed iteration {i+1}/{n_iterations}")
    
    # Calculate p-value (one-sided test)
    p_value = np.mean(auc_differences <= 0)
    
    # Calculate mean AUCs and confidence intervals
    combined_mean = np.mean(combined_aucs)
    ictal_mean = np.mean(ictal_aucs)
    combined_ci = np.percentile(combined_aucs, [2.5, 97.5])
    ictal_ci = np.percentile(ictal_aucs, [2.5, 97.5])
    
    print("\nResults:")
    print(f"Combined Model Mean AUC: {combined_mean:.3f} (95% CI: [{combined_ci[0]:.3f}, {combined_ci[1]:.3f}])")
    print(f"Ictal Model Mean AUC: {ictal_mean:.3f} (95% CI: [{ictal_ci[0]:.3f}, {ictal_ci[1]:.3f}])")
    print(f"Mean AUC Difference: {np.mean(auc_differences):.3f}")
    print(f"One-sided p-value: {p_value:.4f}")
    
    # Save detailed results to a text file
    with open(os.path.join(output_dir, 'statistical_results.txt'), 'w') as f:
        f.write("Statistical Analysis Results\n")
        f.write("===========================\n\n")
        f.write(f"Number of iterations: {n_iterations}\n\n")
        f.write("Model Performance:\n")
        f.write(f"Combined Model Mean AUC: {combined_mean:.3f}\n")
        f.write(f"Combined Model 95% CI: [{combined_ci[0]:.3f}, {combined_ci[1]:.3f}]\n")
        f.write(f"Ictal Model Mean AUC: {ictal_mean:.3f}\n")
        f.write(f"Ictal Model 95% CI: [{ictal_ci[0]:.3f}, {ictal_ci[1]:.3f}]\n\n")
        f.write("Difference Analysis:\n")
        f.write(f"Mean AUC Difference: {np.mean(auc_differences):.3f}\n")
        f.write(f"One-sided p-value: {p_value:.4f}\n")
    
    return auc_differences, combined_aucs, ictal_aucs

def main():
    plt.rcParams['font.family'] = 'Arial'

    # Load data
    print("Loading data...")
    combined_data, ictal_data, interictal_data = load_data()
    
    # Compute model differences
    print("\nStarting model comparison...")
    auc_differences, combined_aucs, ictal_aucs = compute_model_differences(combined_data)
    
    # Plot distribution of AUC differences
    plt.figure(figsize=(10, 6))
    plt.hist(auc_differences, bins=50, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='No Difference')
    plt.title('Distribution of AUC Differences (Combined - Ictal)', fontsize=16, fontweight='bold')
    plt.xlabel('AUC Difference', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'auc_differences_distribution.pdf'))
    plt.close()
    
    # Save the results
    results_df = pd.DataFrame({
        'Combined_AUC': combined_aucs,
        'Ictal_AUC': ictal_aucs,
        'AUC_Difference': auc_differences
    })
    results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)

if __name__ == "__main__":
    print(f"Results will be saved to: {output_dir}")
    main()
    print("\nAnalysis complete!")