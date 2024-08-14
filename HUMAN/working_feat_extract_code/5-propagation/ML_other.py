#%%
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.utils import resample as sklearn_resample
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Import custom functions
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *
from delongs_test import *

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

def bootstrap_roc(y_true, y_pred, n_bootstraps=1000, alpha=0.95):

    rng_seed = 42  # reproducibility
    bootstrapped_tpr = []
    bootstrapped_fpr = np.linspace(0, 1, 100)  
    boot_aucs = []
    
    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = sklearn_resample(np.arange(len(y_pred)), random_state=rng_seed + i)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue

        score = y_pred[indices]
        true = y_true[indices]
        fpr_, tpr_, _ = roc_curve(true, score)
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


def plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color):
    plt.fill_between(bootstrapped_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

# %%
# pearson_df = pd.read_csv('dataset/ML_data/pearson_ML_v3.csv', index_col=0) #GIVES AUC = 0.82
# pearson_df['pt_id'] = pearson_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '')
# pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)

pearson_df = pd.read_csv('dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col = 0) #THIS GIVES AUC = 0.8 
# pearson_df = pd.read_csv('dataset/ML_data/MUSC/pooled_spearman_all_norm.csv', index_col=0)
# pearson_df['SOZ'] = pearson_df['SOZ'].replace(2, 0)
# pearson_df['SOZ'] = pearson_df['SOZ'].replace(3, 0)
pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)



# musc_pearson_df = pd.read_csv('dataset/ML_data/musc_ML_2.csv')

# spearman_df = pd.read_csv('dataset/ML_data/spearman_ML_v2.csv', index_col=0)
EI_df = pd.read_csv('dataset/ML_data/EI_pooled_corr.csv', index_col=0)[['correlation','pt_id']] #using HFER
EI_df['pt_id'] = EI_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '')
EI_df['pt_id'] = EI_df['pt_id'].astype(int)

combine_pearson = pearson_df.merge(EI_df, on='pt_id')#, how = 'left')
#so you should have NaN's in the correlation for some of the MUSC patients

# %%
