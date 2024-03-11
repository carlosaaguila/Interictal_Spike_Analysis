"""
Goal of this script: generate ML model to predict mTLE vs. other, using all features vs. just spike rate. 
"""

# required packages
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

"""
## load the spike data
MUSC_spikes = pd.read_csv('../dataset/MUSC_allspikes_v2.csv', index_col=0)

#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]

#fix SOZ and laterality
MUSC_spikes = MUSC_spikes.merge(MUSC_sozs, left_on = 'pt_id', right_on = 'ParticipantID', how = 'inner')
MUSC_spikes = MUSC_spikes.drop(columns=['ParticipantID','Site_1MUSC_2Emory','IfNeocortical_Location','Correction Notes','lateralization_left','lateralization_right','region'])

# ADD MUSC PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

list_of_feats = ['spike_rate', 'rise_amp','decay_amp','sharpness','linelen','recruiment_latency','spike_width','slow_width','slow_amp']
pearson_feat_df = list()
spearman_feat_df = list()

for Feat_of_interest in list_of_feats:

    take_spike_leads = False

    #########################
    # 1. Organize the data  #
    #########################

    all_spikes = MUSC_spikes

    #flag that says we want spike leaders only
    if take_spike_leads == True:
        all_spikes = all_spikes[all_spikes['is_spike_leader'] == 1]

    #remove patients with 'SOZ' containing other
    # all_spikes = all_spikes[~all_spikes['SOZ'].str.contains('other')].reset_index(drop=True)

    #channels to keep 
    chs_tokeep = ['RA','LA','LPH','RPH','LAH','RAH']

    #if channel_label contains any of the strings in chs_tokeep, keep it
    all_spikes = all_spikes[all_spikes['channel_label'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

    #remove any channels that contains letters that shouldn't be there
    all_spikes = all_spikes[~all_spikes['channel_label'].str.contains('I|LAP|T|S|C')].reset_index(drop=True)

    ## fixes to only have same-side spikes
    #only take the electrode channels that are in the same side
    left_spikes = all_spikes[((all_spikes['Left'] == 1) & (all_spikes['Right'] == 0))].reset_index(drop=True)
    left_spikes_tokeep = left_spikes[~left_spikes['channel_label'].str.contains('R')].reset_index(drop=True)

    right_spikes = all_spikes[((all_spikes['Left'] == 0) & (all_spikes['Right'] == 1))].reset_index(drop=True)
    right_spikes_tokeep = right_spikes[~right_spikes['channel_label'].str.contains('L')].reset_index(drop=True)

    bilateral_spikes = all_spikes[((all_spikes['Left'] == 1) & (all_spikes['Right'] == 1))].reset_index(drop=True)

    #concat them back into all_spikes
    all_spikes = pd.concat([left_spikes_tokeep, right_spikes_tokeep, bilateral_spikes], axis =0).reset_index(drop=True)

    def soz_assigner(row):
        if row['MTL'] == 1:
            return 1
        elif row['Neo'] == 1:
            return 2
        elif row['Temporal'] == 1:
            return 4
        elif row['Other'] == 1:
            return 3
        else:
            return None

    all_spikes['region'] = all_spikes.apply(soz_assigner, axis = 1)

    #get only the spikes that contain 'mesial temporal' in the SOZ column
    mesial_temp_spikes = all_spikes[all_spikes['region'] == 1].reset_index(drop=True)

    # grab the remaining spikes that aren't in mesial_temp_spikes
    non_mesial_temp_spikes = all_spikes[~(all_spikes['region'] == 1)].reset_index(drop=True)

    ########################################
    # 2. Filter Elecs, Group, and Analysis #
    ########################################

    #strip the letters from the channel_label column and keep only the numerical portion
    mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|P', '')
    non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|P', '')

    #replace "sharpness" with the absolute value of it
    mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
    non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

    #group by patient and channel_label and get the average spike rate for each patient and channel
    mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
    mesial_temp_spikes_avg['region'] = 1

    #for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'region'])[Feat_of_interest].mean().reset_index()


    #concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
    all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
    all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','region'], columns='channel_label', values=Feat_of_interest)
    all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10'])

    #reorder all_spikes_avg, so that is_mesial is decesending
    all_spikes_avg = all_spikes_avg.sort_values(by=['region', 'pt_id'], ascending=[True, True])

    #########################
    # Generate Correlations #
    #########################

    #find the spearman correlation of each row in all_spikes_avg
    #initialize a list to store the spearman correlation
    channel_labels = ['1','2','3','4','5','6','7','8','9','10']
    channel_labels = [int(x) for x in channel_labels]
    spearman_corr = []
    label = []
    for row in range(len(all_spikes_avg)):
        #if the row has less than 8 channels, omit from analysis
        if len(all_spikes_avg.iloc[row].dropna()) < 8:
            continue
        spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
        label.append(all_spikes_avg.index[row]) 

    corr_df = pd.DataFrame(spearman_corr, columns=[f'{Feat_of_interest}_correlation', f'{Feat_of_interest}_p-value'])
    corr_df['SOZ'] = [x[1] for x in label]
    corr_df['pt_id'] = [x[0] for x in label]

    # find the pearson correlation of each row in all_spikes_avg
    # initialize a list to store the spearman correlation
    pearson_corr = []
    p_label = []
    for row in range(len(all_spikes_avg)):
        #if the row has less than 8 channels, omit from analysis
        if len(all_spikes_avg.iloc[row].dropna()) < 8:
            continue
        gradient = all_spikes_avg.iloc[row].to_list()
        channel_labels = ['1','2','3','4','5','6','7','8','9','10']
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

    pearson_df = pd.DataFrame(pearson_corr, columns=[f'{Feat_of_interest}_correlation', f'{Feat_of_interest}_p-value'])
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

#change SOZ's 2 and 3, to 0 -> to indicate that they have a "other" epilepsy designation
pearson_df.loc[pearson_df['SOZ'] == 2, 'SOZ'] = 0
pearson_df.loc[pearson_df['SOZ'] == 3, 'SOZ'] = 0

spearman_df.loc[spearman_df['SOZ'] == 2, 'SOZ'] = 0
spearman_df.loc[spearman_df['SOZ'] == 3, 'SOZ'] = 0

#make sure SOZ is an integer
pearson_df['SOZ'] = pearson_df['SOZ'].astype(int)
spearman_df['SOZ'] = spearman_df['SOZ'].astype(int)

pearson_df.to_csv('../dataset/ML_data/MUSC/pearson_ML.csv')
spearman_df.to_csv('../dataset/ML_data/MUSC/spearman_ML.csv')
"""

# Load in the dataframes:
pearson_df = pd.read_csv('../dataset/ML_data/MUSC/pearson_ML.csv', index_col=0)
spearman_df = pd.read_csv('../dataset/ML_data/MUSC/spearman_ML.csv', index_col=0)

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

