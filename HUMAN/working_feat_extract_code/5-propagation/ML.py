#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats

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

#only using the same side electrodes as the SOZ laterality

# ADD BILATERAL PATIENTS
# LOOK AT MESIAL TEMPORAL VS. OTHER
"""
Feat_of_interest = ['spike_rate','slow_width', 'slow_max','rise_amp','decay_amp','linelen','sharpness']
take_spike_leads = False

pearson_feat_df = list()
spearman_feat_df = list()

for feat in Feat_of_interest:

    ####################
    # 1. Load in data  #
    ####################
    #load spikes from dataset   
    if ('rate' in feat) | ('latency' in feat) | (feat == 'seq_spike_time_diff'):
        all_spikes = pd.read_csv('dataset/spikes_bySOZ_T-R.csv', index_col=0)
        bilateral_spikes = pd.read_csv('dataset/bilateral_spikes_bySOZ_T-R.csv', index_col=0)
    else:
        all_spikes = pd.read_csv('dataset/spikes_bySOZ.csv')
        bilateral_spikes = pd.read_csv('dataset/bilateral_MTLE_all_spikes.csv')
        bilateral_spikes = bilateral_spikes.drop(['engel','hup_id','name','spike_rate'], axis=1)

    #rename 'clinic_SOZ' to 'SOZ'
    bilateral_spikes = bilateral_spikes.rename(columns={'clinic_SOZ':'SOZ'})

    all_spikes = pd.concat([all_spikes, bilateral_spikes], axis=0).reset_index(drop=True)

    #flag that says we want spike leaders only
    if take_spike_leads == True:
        all_spikes = all_spikes[all_spikes['is_spike_leader'] == 1]

    #remove patients with 'SOZ' containing other
    # all_spikes = all_spikes[~all_spikes['SOZ'].str.contains('other')].reset_index(drop=True)

    #channels to keep 
    chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']

    #if channel_label contains any of the strings in chs_tokeep, keep it
    all_spikes = all_spikes[all_spikes['channel_label'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

    #only take the electrode channels that are in the same side
    left_spikes = all_spikes[all_spikes['SOZ'].str.contains('left')].reset_index(drop=True)
    left_spikes_tokeep = left_spikes[~left_spikes['channel_label'].str.contains('R')].reset_index(drop=True)

    right_spikes = all_spikes[all_spikes['SOZ'].str.contains('right')].reset_index(drop=True)
    right_spikes_tokeep = right_spikes[~right_spikes['channel_label'].str.contains('L')].reset_index(drop=True)

    bilateral_spikes = all_spikes[all_spikes['SOZ'].str.contains('bilateral')].reset_index(drop=True)

    #concat them back into all_spikes
    all_spikes = pd.concat([left_spikes_tokeep, right_spikes_tokeep, bilateral_spikes], axis =0).reset_index(drop=True)
    #alternative without bilateral
    # all_spikes = pd.concat([left_spikes_tokeep, right_spikes_tokeep], axis =0).reset_index(drop=True)



    #get only the spikes that contain 'mesial temporal' in the SOZ column
    mesial_temp_spikes = all_spikes[all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

    # grab the remaining spikes that aren't in mesial_temp_spikes
    non_mesial_temp_spikes = all_spikes[~all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

    #remove any 'channel_label' that contains the letter T or F
    mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)
    non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)

    ########################################
    # 2. Filter Elecs, Group, and Analysis #
    ########################################

    #strip the letters from the channel_label column and keep only the numerical portion
    mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')
    non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')

    #replace "sharpness" with the absolute value of it
    mesial_temp_spikes[feat] = abs(mesial_temp_spikes[feat])
    non_mesial_temp_spikes[feat] = abs(non_mesial_temp_spikes[feat])

    #group by patient and channel_label and get the average spike rate for each patient and channel
    mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[feat].mean().reset_index()
    #for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
    non_mesial_temp_spikes['SOZ'] = non_mesial_temp_spikes['SOZ'].str.split('_').str[1]
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'SOZ'])[feat].mean().reset_index()

    # for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
    mesial_temp_spikes_avg['SOZ'] = 1

    #concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
    all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
    all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=feat)
    all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

    #reorder all_spikes_avg, so that is_mesial is decesending
    all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])


    #create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
    #the values are the average spike rate for each patient and channel
    mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=feat)
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=feat)

    #reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
    mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

    #remove 'HUP215' from all_spikes_avg
    if ('latency' in feat) | (feat == 'seq_spike_time_diff'):
        all_spikes_avg = all_spikes_avg.drop('HUP215')
        all_spikes_avg = all_spikes_avg.drop('HUP099')

    channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
    channel_labels = [int(x) for x in channel_labels]
    spearman_corr = []
    label = []
    for row in range(len(all_spikes_avg)):
        spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
        label.append(all_spikes_avg.index[row])

    corr_df = pd.DataFrame(spearman_corr, columns=[f'{feat}_correlation', f'{feat}_p-value'])
    corr_df['SOZ'] = [x[1] for x in label]
    corr_df['pt_id'] = [x[0] for x in label]

    # find the pearson correlation of each row in all_spikes_avg
    # initialize a list to store the spearman correlation
    pearson_corr = []
    p_label = []
    for row in range(len(all_spikes_avg)):
        gradient = all_spikes_avg.iloc[row].to_list()
        channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
        channel_labels = [int(x) for x in channel_labels]
        # for each nan in the graident list, remove the corresponding channel_labels
        list_to_remove = []
        for i in range(len(channel_labels)):
            if np.isnan(gradient[i]):
                list_to_remove.append(i)

        #remove list_to_remove from channel_labels and gradient
        channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
        gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]

        pearson_corr.append(stats.pearsonr(channel_labels,gradient))
        p_label.append(all_spikes_avg.index[row])

    pearson_df = pd.DataFrame(pearson_corr, columns=[f'{feat}_correlation', f'{feat}_p-value'])
    pearson_df['SOZ'] = [x[1] for x in label]
    pearson_df['pt_id'] = [x[0] for x in label]


    # append the dataframe with spearman/pearson correlation values
    pearson_feat_df.append(pearson_df)
    spearman_feat_df.append(corr_df)


pearson_df = pd.concat(pearson_feat_df, axis=1)
spearman_df = pd.concat(spearman_feat_df, axis=1)

#keep only one SOZ and pt_id column in pearson_df and spearman_df
pearson_df = pearson_df.loc[:,~pearson_df.columns.duplicated()]
spearman_df = spearman_df.loc[:,~spearman_df.columns.duplicated()]

#remove any column that contains 'p-value'
pearson_df = pearson_df.loc[:,~pearson_df.columns.str.contains('p-value')]
spearman_df = spearman_df.loc[:,~spearman_df.columns.str.contains('p-value')]

#reorder columns so that SOZ and pt_id are first
pearson_df = pearson_df[['SOZ', 'pt_id'] + [c for c in pearson_df if c not in ['SOZ', 'pt_id']]]
spearman_df = spearman_df[['SOZ', 'pt_id'] + [c for c in spearman_df if c not in ['SOZ', 'pt_id']]]

#change patients with SOZ == 1 to 'mesial'
pearson_df.loc[pearson_df['SOZ'] == 1, 'SOZ'] = 'mesial'
spearman_df.loc[spearman_df['SOZ'] == 1, 'SOZ'] = 'mesial'

#keep patients with either 'mesial' or 'neocortical' in SOZ
pearson_df = pearson_df[pearson_df['SOZ'].str.contains('mesial|neocortical|cortex')].reset_index(drop=True)
spearman_df = spearman_df[spearman_df['SOZ'].str.contains('mesial|neocortical|cortex')].reset_index(drop=True)

#change 'mesial' to 1 and 'neocortical' to 0
pearson_df.loc[pearson_df['SOZ'] == 'mesial', 'SOZ'] = 1
pearson_df.loc[pearson_df['SOZ'] == 'temporal neocortical', 'SOZ'] = 0
pearson_df.loc[pearson_df['SOZ'] == 'other cortex', 'SOZ'] = 0

spearman_df.loc[spearman_df['SOZ'] == 'mesial', 'SOZ'] = 1
spearman_df.loc[spearman_df['SOZ'] == 'temporal neocortical', 'SOZ'] = 0
spearman_df.loc[spearman_df['SOZ'] == 'other cortex', 'SOZ'] = 0

#make sure SOZ is an integer
pearson_df['SOZ'] = pearson_df['SOZ'].astype(int)
spearman_df['SOZ'] = spearman_df['SOZ'].astype(int)

pearson_df.to_csv('dataset/ML_data/pearson_ML_v2.csv')
spearman_df.to_csv('dataset/ML_data/spearman_ML_v2.csv')

"""
# %%

pearson_df = pd.read_csv('dataset/ML_data/pearson_ML_v2.csv', index_col=0)
spearman_df = pd.read_csv('dataset/ML_data/spearman_ML_v2.csv', index_col=0)

########################
# LEAVE ONE OUT - Logistic Regression
# ########################

all_feats = pearson_df.merge(spearman_df, on=['SOZ', 'pt_id'])

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

RocCurveDisplay.from_predictions(y_true_clean, y_predprob_clean)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
plt.grid()
plt.title('FPR vs. TPR ROC Curve of LR Testing Performance (all features)')

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

rfc_confusion = C_M(y_true_clean, y_pred_clean)
rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
plt.figure(figsize=(6,4))
sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
plt.title("Confusion Matrix for LR test set predictions (all features)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


TP = rfc_confusion[1,1]
TN = rfc_confusion[0,0]
FN = rfc_confusion[1,0]
FP = rfc_confusion[0,1]

sensitivity= TP / (TP + FN) 
specificity = TN / (TN + FP) 
bal_accuracy = (sensitivity + specificity) / 2
print("Balanced Accuracy:", bal_accuracy)




# %%

########################
# LEAVE ONE OUT - Logistic Regression
# ########################

#ONLY SPIKE RATE

all_feats = all_feats[['SOZ','pt_id','spike_rate_correlation_x', 'spike_rate_correlation_y']]

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
    rfc = LogisticRegression(penalty = 'l2', solver = 'liblinear', l1_ratio = 0.5).fit(X_train, y_train)

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

RocCurveDisplay.from_predictions(y_true_clean, y_predprob_clean)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
plt.grid()
plt.title('FPR vs. TPR ROC Curve of LR Testing Performance (spike rate)')

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

rfc_confusion = C_M(y_true_clean, y_pred_clean)
rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
plt.figure(figsize=(6,4))
sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
plt.title("Confusion Matrix for LR test set predictions (spike rate)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

TP = rfc_confusion[1,1]
TN = rfc_confusion[0,0]
FN = rfc_confusion[1,0]
FP = rfc_confusion[0,1]

sensitivity= TP / (TP + FN) 
specificity = TN / (TN + FP) 
bal_accuracy = (sensitivity + specificity) / 2
print("Balanced Accuracy:", bal_accuracy)


# %%
