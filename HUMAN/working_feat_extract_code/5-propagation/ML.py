#%% required packages
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
pearson_df['SOZ'] = pearson_df['SOZ'].replace(2, 0)
pearson_df['SOZ'] = pearson_df['SOZ'].replace(3, 0)
pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)



# musc_pearson_df = pd.read_csv('dataset/ML_data/musc_ML_2.csv')

# spearman_df = pd.read_csv('dataset/ML_data/spearman_ML_v2.csv', index_col=0)
EI_df = pd.read_csv('dataset/ML_data/EI_pooled_corr.csv', index_col=0)[['correlation','pt_id']] #using HFER
EI_df['pt_id'] = EI_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '')
EI_df['pt_id'] = EI_df['pt_id'].astype(int)

combine_pearson = pearson_df.merge(EI_df, on='pt_id')#, how = 'left')
#so you should have NaN's in the correlation for some of the MUSC patients
#%%
########################
# LEAVE ONE OUT - Logistic Regression
# ########################

###############################################
# ALL FEATURES
###############################################


# all_feats = pearson_df.merge(spearman_df, on=['SOZ', 'pt_id'])
all_feats = combine_pearson
all_feats = all_feats.dropna(subset = 'correlation')

#Split the data according to IDs 
#from all_feats dataframe, get the unique id's
unique_ids = all_feats['pt_id'].unique()
#split into two lists of unique ids in a random order
np.random.shuffle(unique_ids)

#create LeaveOneOut model
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()

# Initialize the model and fit it on the training set
# enumerate splits
y_true, y_pred = list(), list()
y_predprob = list()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

feature_importances_TEST = list()
for train_ix, test_ix in LOO.split(unique_ids):
    #get data
    X_train = all_feats[all_feats['pt_id'].isin(unique_ids[train_ix])]
    X_test = all_feats[all_feats['pt_id'].isin(unique_ids[test_ix])]
    y_train = X_train[['SOZ']]
    y_test = X_test[['SOZ']]
    #drop columns 'isSOZ' and 'id'
    X_train = X_train.drop(columns = ['SOZ', 'pt_id'])
    X_test = X_test.drop(columns = ['SOZ', 'pt_id'])

    # fit model
    # rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = None).fit(X_train, y_train)
    rfc = LogisticRegression().fit(X_train, y_train)

    # evaluate model
    yhat = rfc.predict(X_test)
    y_pred_prob = rfc.predict_proba(X_test)[:,1]

    # store
    y_predprob.append(y_pred_prob)
    y_true.append(y_test['SOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy
    # for random forest, feature_importances_ is the feature importance
    # feature_importances_TEST.append(rfc.feature_importances_)
    #for logistic regression, feature_importances_ is the coefficients
    feature_importances_TEST.append(rfc.coef_[0])

################ evaluate predictions
from sklearn.metrics import accuracy_score
y_true_clean = [x for x in y_true for x in x]
y_pred_clean = [x for x in y_pred for x in x]
y_predprob_clean = [x for x in y_predprob for x in x]

combined_ids = unique_ids
combined_y = y_predprob_clean
combined_true = y_true_clean

acc = accuracy_score(y_true_clean, y_pred_clean)
print('Accuracy: %.3f' % acc)

################ AUC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

plt.figure(figsize = (8,8))
plt.rcParams['font.family'] = 'Arial'
fpr1, tpr1,_ = roc_curve(y_true_clean, y_predprob_clean)
roc_auc1 = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, color='#E64B35FF', lw=3, label='Combined (AUC = %0.2f)' % roc_auc1)
# plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), lw = 1, '--', color='black')
# plt.grid()
plt.title('ROC Curves for Classifying MTLE', fontsize = 24, fontweight = 'bold')
plt.xlabel('False Positive Rate', fontsize = 20,fontweight = 'bold')
plt.ylabel('True Positive Rate', fontsize = 20,fontweight = 'bold')

# Bootstrap ROC to get 95% confidence intervals
bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(np.array(y_true_clean), np.array(y_predprob_clean))

# Plot ROC curve with confidence intervals
plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color='#E64B35FF')

# rfc_confusion = C_M(y_true_clean, y_pred_clean)
# rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
# plt.figure(figsize=(8,8))
# sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
# plt.title("Confusion Matrix for LR test set predictions (all features)")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.savefig('figures/ML/all_feats_confusion.pdf')


# TP = rfc_confusion[1,1]
# TN = rfc_confusion[0,0]
# FN = rfc_confusion[1,0]
# FP = rfc_confusion[0,1]

# sensitivity= TP / (TP + FN) 
# specificity = TN / (TN + FP) 
# bal_accuracy = (sensitivity + specificity) / 2
# print("Balanced Accuracy:", bal_accuracy)

###############################################
#INTERICTAL FEATURES
###############################################


# all_feats = pd.read_csv('dataset/ML_data/pearson_ML_v3.csv', index_col = 0)
# all_feats['SOZ'] = all_feats['SOZ'].replace(2, 0)

pearson_df = pd.read_csv('dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col = 0) #AUC = 0.8
pearson_df['SOZ'] = pearson_df['SOZ'].replace(2, 0)
pearson_df['SOZ'] = pearson_df['SOZ'].replace(3, 0)

all_feats = pearson_df

#Split the data according to IDs 
#from all_feats dataframe, get the unique id's
unique_ids = all_feats['pt_id'].unique()
#split into two lists of unique ids in a random order
np.random.shuffle(unique_ids)

#create LeaveOneOut model
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()

# Initialize the model and fit it on the training set
# enumerate splits
y_true, y_pred = list(), list()
y_predprob = list()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

feature_importances_TEST = list()
for train_ix, test_ix in LOO.split(unique_ids):

    #get data
    X_train = all_feats[all_feats['pt_id'].isin(unique_ids[train_ix])]
    X_test = all_feats[all_feats['pt_id'].isin(unique_ids[test_ix])]
    y_train = X_train[['SOZ']]
    y_test = X_test[['SOZ']]
    #drop columns 'isSOZ' and 'id'
    X_train = X_train.drop(columns = ['SOZ', 'pt_id'])
    X_test = X_test.drop(columns = ['SOZ', 'pt_id'])

    # fit model
    # rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = None).fit(X_train, y_train)
    # rfc = LogisticRegression(penalty = 'l2', solver = 'liblinear', l1_ratio = 0.5).fit(X_train, y_train)
    rfc = LogisticRegression().fit(X_train, y_train)

    # evaluate model
    yhat = rfc.predict(X_test)
    y_pred_prob = rfc.predict_proba(X_test)[:,1]
    # store
    y_predprob.append(y_pred_prob)
    y_true.append(y_test['SOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy
    # for random forest, feature_importances_ is the feature importance
    # feature_importances_TEST.append(rfc.feature_importances_)
    #for logistic regression, feature_importances_ is the coefficients
    feature_importances_TEST.append(rfc.coef_[0])

################ evaluate predictions
from sklearn.metrics import accuracy_score
y_true_clean = [x for x in y_true for x in x]
y_pred_clean = [x for x in y_pred for x in x]
y_predprob_clean = [x for x in y_predprob for x in x]

inter_ids = unique_ids
inter_y = y_predprob_clean
inter_true = y_true_clean

acc = accuracy_score(y_true_clean, y_pred_clean)
print('Accuracy: %.3f' % acc)

################ AUC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support

# plt.figure(figsize = (8,8))
# RocCurveDisplay.from_predictions(y_true_clean, y_predprob_clean)
fpr2, tpr2,_ = roc_curve(y_true_clean, y_predprob_clean)
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, color='#7E6148FF', lw=3, label='Interictal (AUC = %0.2f)' % roc_auc2)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')

# Bootstrap ROC to get 95% confidence intervals
bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(np.array(y_true_clean), np.array(y_predprob_clean))

# Plot ROC curve with confidence intervals
plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color='#7E6148FF')


# plt.grid()
# plt.title('FPR vs. TPR ROC Curve of LR Testing Performance (spike rate)')
# plt.savefig('figures/ML/rate_only_roc.pdf')

# ################ Confusion Matrix
# from sklearn.metrics import confusion_matrix as C_M
# import seaborn as sns

# rfc_confusion = C_M(y_true_clean, y_pred_clean)
# rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
# plt.figure(figsize=(8,8))
# sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
# plt.title("Confusion Matrix for LR test set predictions (spike rate)")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.savefig('figures/ML/rate_only_confusion.pdf')

# TP = rfc_confusion[1,1]
# TN = rfc_confusion[0,0]
# FN = rfc_confusion[1,0]
# FP = rfc_confusion[0,1]

# sensitivity= TP / (TP + FN) 
# specificity = TN / (TN + FP) 
# bal_accuracy = (sensitivity + specificity) / 2
# print("Balanced Accuracy:", bal_accuracy)


########################
# #ONLY ICTAL DATA
# ########################


all_feats = combine_pearson[['correlation','pt_id','SOZ']]
all_feats = all_feats.dropna(subset = 'correlation')


#Split the data according to IDs 
#from all_feats dataframe, get the unique id's
unique_ids = all_feats['pt_id'].unique()
#split into two lists of unique ids in a random order
np.random.shuffle(unique_ids)

#create LeaveOneOut model
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()

# Initialize the model and fit it on the training set
# enumerate splits
y_true, y_pred = list(), list()
y_predprob = list()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

feature_importances_TEST = list()
for train_ix, test_ix in LOO.split(unique_ids):

    #get data
    X_train = all_feats[all_feats['pt_id'].isin(unique_ids[train_ix])]
    X_test = all_feats[all_feats['pt_id'].isin(unique_ids[test_ix])]
    y_train = X_train[['SOZ']]
    y_test = X_test[['SOZ']]
    #drop columns 'isSOZ' and 'id'
    X_train = X_train.drop(columns = ['SOZ', 'pt_id'])
    X_test = X_test.drop(columns = ['SOZ', 'pt_id'])

    # fit model
    # rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = None).fit(X_train, y_train)
    # rfc = LogisticRegression(penalty = 'l2', solver = 'liblinear', l1_ratio = 0.5).fit(X_train, y_train)
    rfc = LogisticRegression().fit(X_train, y_train)

    # evaluate model
    yhat = rfc.predict(X_test)
    y_pred_prob = rfc.predict_proba(X_test)[:,1]
    # store
    y_predprob.append(y_pred_prob)
    y_true.append(y_test['SOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy
    # for random forest, feature_importances_ is the feature importance
    # feature_importances_TEST.append(rfc.feature_importances_)
    #for logistic regression, feature_importances_ is the coefficients
    feature_importances_TEST.append(rfc.coef_[0])

################ evaluate predictions
from sklearn.metrics import accuracy_score
y_true_clean = [x for x in y_true for x in x]
y_pred_clean = [x for x in y_pred for x in x]
y_predprob_clean = [x for x in y_predprob for x in x]

ictal_ids = unique_ids
ictal_y = y_predprob_clean
ictal_true = y_true_clean

acc = accuracy_score(y_true_clean, y_pred_clean)
print('Accuracy: %.3f' % acc)

################ AUC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support

fpr3, tpr3,_ = roc_curve(y_true_clean, y_predprob_clean)
roc_auc3 = auc(fpr3, tpr3)
plt.plot(fpr3, tpr3, color='#00A087FF', lw=3, label='Ictal (AUC = %0.2f)' % roc_auc3)

# Bootstrap ROC to get 95% confidence intervals
bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(np.array(y_true_clean), np.array(y_predprob_clean))

# Plot ROC curve with confidence intervals
plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color='#00A087FF')

plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
sns.despine()
# plt.figure(figsize = (8,8))
# RocCurveDisplay.from_predictions(y_true_clean, y_predprob_clean)
# plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
# plt.grid()
# plt.title('FPR vs. TPR ROC Curve of LR Testing Performance (spike rate)')
# plt.savefig('figures/ML/rate_only_roc.pdf')

# ################ Confusion Matrix
# from sklearn.metrics import confusion_matrix as C_M
# import seaborn as sns

# rfc_confusion = C_M(y_true_clean, y_pred_clean)
# rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
# plt.figure(figsize=(8,8))
# sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
# plt.title("Confusion Matrix for LR test set predictions (spike rate)")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# Alternatively, use tick_params for setting font size
plt.tick_params(axis='both', which='major', labelsize=16)

# Manually set the weight to bold for tick labels
for label in plt.gca().get_xticklabels():
    label.set_weight('bold')
for label in plt.gca().get_yticklabels():
    label.set_weight('bold')

# plt.savefig('figures/ML/all_ROCS_w_CI.pdf')

plt.show()

# TP = rfc_confusion[1,1]
# TN = rfc_confusion[0,0]
# FN = rfc_confusion[1,0]
# FP = rfc_confusion[0,1]

# sensitivity= TP / (TP + FN) 
# specificity = TN / (TN + FP) 
# bal_accuracy = (sensitivity + specificity) / 2
# print("Balanced Accuracy:", bal_accuracy)


# %%
def rearrange_order(combined_ids, target_ids, target_true, target_y):
    # Create a dictionary to map ids to their true and predicted values
    id_to_true = {id_: true for id_, true in zip(target_ids, target_true)}
    id_to_y = {id_: y for id_, y in zip(target_ids, target_y)}
    
    # Rearrange the target lists to match the order of combined_ids
    rearranged_ids = [id_ for id_ in combined_ids]
    rearranged_true = [id_to_true[id_] for id_ in combined_ids]
    rearranged_y = [id_to_y[id_] for id_ in combined_ids]
    
    return rearranged_ids, rearranged_true, rearranged_y


# Rearrange ictal lists
ictal_ids_rearranged, ictal_true_rearranged, ictal_y_rearranged = rearrange_order(combined_ids, ictal_ids, ictal_true, ictal_y)

# Rearrange inter lists
inter_ids_rearranged, inter_true_rearranged, inter_y_rearranged = rearrange_order(combined_ids, inter_ids, inter_true, inter_y)

#%%
print(ictal_ids_rearranged)
print(inter_ids_rearranged)
print(combined_ids)
# %%

#Run the delong test on the pairs

print('Difference between COMBINED vs. ICTAL-ONLY:')
auc1, cov1, p1 = (delong_roc_test(np.array(combined_true), np.array(combined_y), np.array(ictal_y_rearranged)))
print("AUC:",auc1)
print("COV:",cov1)
print("p-value:",np.exp(np.log(10)*p1)) #needed extra math to calculate the ture p-val, this was implemented from the github

print('Difference between INTER-ONLY vs. ICTAL-ONLY:')
auc2, cov2, p2 = (delong_roc_test(np.array(combined_true), np.array(inter_y_rearranged), np.array(ictal_y_rearranged)))
print("AUC:",auc2)
print("COV:",cov2)
print("p-value:",np.exp(np.log(10)*p2))

print('Difference between INTER-ONLY vs. COMBINED:')
auc3, cov3, p3 = (delong_roc_test(np.array(combined_true), np.array(inter_y_rearranged), np.array(combined_y)))
print("AUC:",auc3)
print("COV:",cov3)
print("p-value:",np.exp(np.log(10)*p3)) 


# %%

p_values =  [p1,p2,p3]
p_values = [np.exp(np.log(10)*p) for p in p_values]
desired_fdr = 0.05

def benjamini_hochberg(p_values, fdr):
    m = len(p_values)
    sorted_p_values = np.sort(p_values)
    sorted_index = np.argsort(p_values)
    bh_critical_values = np.arange(1, m + 1) / m * fdr

    # Determine the largest p-value that is less than or equal to its BH critical value
    significant = sorted_p_values <= bh_critical_values
    if significant.any():
        max_significant_index = np.where(significant)[0][-1]
        threshold_p_value = sorted_p_values[max_significant_index]
    else:
        threshold_p_value = None

    # Determine which p-values are significant
    significant_p_values = p_values <= threshold_p_value if threshold_p_value is not None else np.zeros_like(p_values, dtype=bool)

    return significant_p_values, threshold_p_value

significant_p_values, threshold_p_value = benjamini_hochberg(p_values, desired_fdr)

print("P-values:", p_values)
print("Significant p-values:", significant_p_values)
print("Threshold p-value:", threshold_p_value)
