#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Import custom functions
import sys, os
code_v2_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/')
sys.path.append(code_v2_path)
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *

code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

# %%
subset = True


pearson_df = pd.read_csv('dataset/ML_data/pearson_ML_v3.csv', index_col=0)
pearson_df['SOZ'] = pearson_df['SOZ'].replace(2, 0)
spearman_df = pd.read_csv('dataset/ML_data/spearman_ML_v2.csv', index_col=0)
if subset == True:
    EI_df = pd.read_csv('dataset/ML_data/EI_corr_subset_pearson.csv', index_col = 0)[['EI_corr','pt_id']]
elif subset == False:
    EI_df = pd.read_csv('dataset/ML_data/EI_corr_full_pearson.csv', index_col = 0)[['correlation','pt_id']]

combine_pearson = pearson_df.merge(EI_df, on='pt_id')

#%%
########################
# LEAVE ONE OUT - Logistic Regression
# ########################

# all_feats = pearson_df.merge(spearman_df, on=['SOZ', 'pt_id'])
all_feats = combine_pearson

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

# plt.savefig('figures/ML/all_feats_ROC.pdf')

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

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

########################
# LEAVE ONE OUT - Logistic Regression
# ########################

#ONLY SPIKE RATE

all_feats = pd.read_csv('dataset/ML_data/pearson_ML_v2.csv', index_col = 0)

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

if subset == False: 
    all_feats = combine_pearson[['correlation','pt_id','SOZ']]
else:
    all_feats = combine_pearson[['EI_corr','pt_id','SOZ']]


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

# plt.savefig('figures/ML/all_ROCS_AES.pdf')

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
