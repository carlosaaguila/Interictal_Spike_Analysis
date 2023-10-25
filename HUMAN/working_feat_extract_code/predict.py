#%% import requirements
import pickle as pkl
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *
#pd.set_option('display.max_rows', None)

#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

#%% 
# load the data
SOZ_feats_new = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v3/SOZ_feats.csv', index_col = 0)
nonSOZ_feats_new = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v3/nonSOZ_feats.csv', index_col = 0)
 
# clean the data
SOZ_feats_new['isSOZ'] = 1
nonSOZ_feats_new['isSOZ'] = 0

#%% add some extra features
#for SOZ
SOZ_feats_new['spike_width'] = SOZ_feats_new['right_point']-SOZ_feats_new['left_point']
#calculate the sharpness of a spike, by subtracting decay_slope from rise_slope
SOZ_feats_new['sharpness'] = np.abs(SOZ_feats_new['rise_slope']-SOZ_feats_new['decay_slope'])
#calculate rise_duration of a spike, by subtracting left_point from peak
SOZ_feats_new['rise_duration'] = SOZ_feats_new['peak']-SOZ_feats_new['left_point']
#calculate decay_duration of a spike, by subtracting peak from right_point
SOZ_feats_new['decay_duration'] = SOZ_feats_new['right_point']-SOZ_feats_new['peak']

#repeat for nonSOZ
nonSOZ_feats_new['spike_width'] = nonSOZ_feats_new['right_point']-nonSOZ_feats_new['left_point']
nonSOZ_feats_new['sharpness'] = np.abs(nonSOZ_feats_new['rise_slope']-nonSOZ_feats_new['decay_slope'])
nonSOZ_feats_new['rise_duration'] = nonSOZ_feats_new['peak']-nonSOZ_feats_new['left_point']
nonSOZ_feats_new['decay_duration'] = nonSOZ_feats_new['right_point']-nonSOZ_feats_new['peak']

#%%
# concatenate all the data vertically
all_feats = pd.concat([SOZ_feats_new, nonSOZ_feats_new], axis = 0)

#remove the '[' and ']' from the spike_rate column
all_feats['spike_rate'] = all_feats['spike_rate'].str.replace('[', '')
all_feats['spike_rate'] = all_feats['spike_rate'].str.replace(']', '')

# convert spike_rate to a float
all_feats['spike_rate'] = all_feats['spike_rate'].astype(float)

all_feats = all_feats.dropna()

#%%
#group by 'name' and 'id' and take the mean of each column
all_feats_eleclevel = all_feats.groupby(['name','id']).median().reset_index()
# all_feats_eleclevel = all_feats.groupby(['id','isSOZ']).median().reset_index()

#shuffle and reset the index of all_feats
all_feats_eleclevel = all_feats_eleclevel.sample(frac=1).reset_index(drop=True)

#make 'isSOZ' a integer
all_feats_eleclevel['isSOZ'] = all_feats_eleclevel['isSOZ'].astype(int)

# clean dataframe
all_feats_eleclevel = all_feats_eleclevel.drop(columns = ['peak','left_point','average_amp','right_point','slow_end','slow_max','name'])

# for each row in 'id', remove 'HUP' and convert to int
all_feats_eleclevel['id'] = all_feats_eleclevel['id'].str.replace('HUP', '')
all_feats_eleclevel['id'] = all_feats_eleclevel['id'].astype(int)

#%%
"""
import seaborn as sns

#run correlation matrices on features
ids_in_study = all_feats_eleclevel['id'].unique()
#for each id in ids_in_study calculate a correlation matrix
corr_matrices = np.zeros(len(ids_in_study), dtype = object)
for i,pt in enumerate(ids_in_study):
    print(pt)
    #get the data for that id
    id_data = all_feats_eleclevel[all_feats_eleclevel['id'] == pt]
    #drop id column
    id_data = id_data.drop(columns = ['id'])
    #get the correlation matrix for that id
    corr_matrix = id_data.corr().abs()
    #plt.figure(figsize=(10,10))
    #plt.title(f'Correlation Matrix for {ids_in_study[i]}')
    #sns.heatmap(corr_matrix.abs(), cmap='coolwarm', annot=True, fmt='.2f')
    #plt.show()
    corr_matrices[i] = corr_matrix

#find the mean across all the matrices in corr_matrices for each element in the matrix, keeping the same shape
mean_corr_matrix = np.mean(corr_matrices, axis = 0)
"""

#%% RUN PCA ON ALL FEATURES
"""
#Import necessary libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# TO-DO: Get transformed set of principal components on x_test (17)
# 1. Refit and transform on training with parameter n (as deduced from the last step) 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# TO-DO: Instantiate and Fit PCA
pca = PCA(random_state = seed, n_components = 17)
pca.fit(X_train_s)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_test_pca = pca.transform(X_test_s)
X_train_pca = pca.transform(X_train_s)

explained_variance_ratios = PCmodel.explained_variance_ratio_
cum_evr = np.cumsum(explained_variance_ratios)
"""
#%% split train/test
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_feats.drop(columns = ['isSOZ']), all_feats[['isSOZ']], test_size=0.25, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
 
print("Training size:\t\t", len(x_train))
print("Validation size:\t", len(x_val))
print("Test size:\t\t", len(x_test))
print('\n')
print("Training, label 0:\t", len(y_train[y_train['isSOZ'] == 0]))
print("Training, label 1:\t", len(y_train[y_train['isSOZ'] == 1]))
print('\n')
print("Validation, label 0:\t", len(y_val[y_val['isSOZ'] == 0]))
print("Validation, label 1:\t", len(y_val[y_val['isSOZ'] == 1]))
print('\n')
print("Test, label 0:\t\t", len(y_test[y_test['isSOZ'] == 0]))
print("Test, label 1:\t\t", len(y_test[y_test['isSOZ'] == 1]))
"""
# %% train a Random Forest Classifier
"""

########################
# RANDOM FOREST CLASSIFIER
########################

from sklearn.ensemble import RandomForestClassifier

#Initialize the model and fit it on the training set
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = None).fit(x_train, y_train)

#use the model to predict on the test set and save these predictions as 'rfc_y_pred'
rfc_y_pred = rfc.predict(x_test)

#Find the R-squared score for the test set and the validation set and store the value in 'rfc_score_test' and 'rfc_score_val'
rfc_score_test = rfc.score(x_test, y_test)
rfc_score_val = rfc.score(x_val, y_val)

print("Random Forest Classifier R-squared score for test set:\t", rfc_score_test)
print("Random Forest Classifier R-squared score for validation set:\t", rfc_score_val)

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

rfc_confusion = C_M(y_test, rfc_y_pred)
rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
plt.figure(figsize=(6,4))
sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
plt.title("Confusion Matrix for Random Forest Classifier test set predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

"""
# Random Forest Classifier R-squared score for    test set:	         0.6379378197560016
# Random Forest Classifier R-squared score for    validation set:	     0.6194411648957103
"""

############### Precision, Recall, F1 Score
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, _ = precision_recall_fscore_support(y_test, rfc_y_pred, average='binary')
print('Precision: ', prec)
print('Recall: ', rec)
print('F1 Score: ', f1)

"""
# TP = 1126
# TN = 495
# FP = 579
# FN = 341

# Given our confusion matrix, we see 579 False Positives and 341 False Negatives using our Random Forest Classifier given our test set.

# Given this, we calculated our precision or the positive predictive value to understand the fraction 
# of true postitives from all relevant positive predictions (true and false positives). Our precision from this model was 0.661.

# We also measured recall or sensitivity, to understand the ratio of relevant instances that were predicted.
# Our recall score for this model was 0.768.

# Additiaonlly, we calculated an F1 score to display the model's accuracy as a harmonic mean of both the
# precision and the recall in one metric. Our F1 score for this model was 0.710.
"""

############### AUC Curves
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

RocCurveDisplay.from_predictions(y_test, rfc_y_pred)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
plt.grid()
plt.title('FPR vs. TPR ROC Curve of RFC Testing Performance')

"""
# An ROC curve (reciever oparting characteristic curve) shows the performance of a classification model 
# at different classifications thresholds by showing the ratio of the True Positive Rate by the
# False Positive Rate. To further understand the ROC curve, we calculate the area under the curve (AUC).
# This provides an aggregate measurement of performace that is equal to the probability that the model 
# ranks a random positive example more highly than a random negative example.

# For our Random Forest Classifier, we get a AUC of 0.61, signifying that our chances of 
# predicting the correct value are a bit higher than chance. (50/50)
"""

############### Feature Importances

# Set names for columns (word number)
col = x_test.columns

# Define vector for feature importances
y = rfc.feature_importances_

# Plot the feature importances 
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(y)) # the x locations for the groups
ax.barh(ind, y, width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False)
plt.title("Feature importance in RandomForest Classifier")
plt.xlabel("Relative importance")
plt.ylabel("Morphology Feature") 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
plt.show()

"""
#%% Train a Support Vector Machine Classifier
"""

########################
#SUPPORT VECTOR MACHINE
########################

from sklearn.svm import SVC

#initialize the model and fit it on the training set
svm_model = SVC(kernel = 'sigmoid', random_state = 42).fit(x_train, y_train)

#use the model to predict on the test set and save these predictions as 'y_pred_svm'
y_pred_svm = svm_model.predict(x_test)

#mean-accuracy score for test and validation sets and store value as 'svm_score_test' and 'svm_score_val'
svm_score_test = svm_model.score(x_test, y_test)
svm_score_val = svm_model.score(x_val, y_val)
print('Testing set accuracy:\t', svm_score_test)
print('Validation set accuracy:\t', svm_score_val)

################ AUC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

RocCurveDisplay.from_predictions(y_test, y_pred_svm)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
plt.grid()
plt.title('FPR vs. TPR ROC Curve of SVM Testing Performance')

"""
# Performance was not very good 
# AUC = 0.47
"""
"""
# %% Train Linear Regression Models
"""

########################
#LOGISTIC REGRESSION CLASSIFIER
########################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

LR = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2', 'elasticnet'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_LR_acc = GridSearchCV(LR, param_grid = grid_values, scoring = 'recall')
grid_LR_acc.fit(x_train, y_train)

#Predict values based on new parameters
y_pred_LR = grid_LR_acc.predict(x_test)

#Logisitic Regression scores for test and validation sets and store value as 'LR_score_test' and 'LR_score_val' 
LR_score_test = grid_LR_acc.score(x_test, y_test)
LR_score_val = grid_LR_acc.score(x_val, y_val)
print('Testing set accuracy:\t', LR_score_test)
print('Validation set accuracy:\t', LR_score_val)

################ AUC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

RocCurveDisplay.from_predictions(y_test, y_pred_LR)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
plt.grid()
plt.title('FPR vs. TPR ROC Curve of LR Testing Performance')

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

rfc_confusion = C_M(y_test, y_pred_LR)
rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
plt.figure(figsize=(6,4))
sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
plt.title("Confusion Matrix for Logistic Reg. test set predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

"""
# Random Forest Classifier R-squared score for    test set:	         0.6379378197560016
# Random Forest Classifier R-squared score for    validation set:	     0.6194411648957103
"""

############### Precision, Recall, F1 Score
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_LR, average='binary')
print('Precision: ', prec)
print('Recall: ', rec)
print('F1 Score: ', f1)

############### Feature Importances

# Set names for columns (word number)
col = x_test.columns

# Define vector for feature importances
y = rfc.feature_importances_

# Plot the feature importances 
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(y)) # the x locations for the groups
ax.barh(ind, y, width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False)
plt.title("Feature importance in Logistic Regression")
plt.xlabel("Relative importance")
plt.ylabel("Morphology Feature") 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
plt.show()
"""
#%%

all_feats = all_feats_eleclevel

# %%
"""
########################
# LEAVE ONE OUT - RANDOM FOREST CLASSIFIER
# ########################

#Split the data according to IDs 

#from all_feats dataframe, get the unique id's
unique_ids = all_feats['id'].unique()
#split into two lists of unique ids in a random order
np.random.shuffle(unique_ids)

#create LeaveOneOut model
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()

# Initialize the model and fit it on the training set
# enumerate splits
y_true, y_pred = list(), list()
y_predprob = list()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

for train_ix, test_ix in LOO.split(unique_ids):

    #get data
    X_train = all_feats[all_feats['id'].isin(unique_ids[train_ix])]
    X_test = all_feats[all_feats['id'].isin(unique_ids[test_ix])]
    y_train = X_train[['isSOZ']]
    y_test = X_test[['isSOZ']]
    #drop columns 'isSOZ' and 'id'
    X_train = X_train.drop(columns = ['isSOZ', 'id'])
    X_test = X_test.drop(columns = ['isSOZ', 'id'])

    # fit model
    rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = None).fit(X_train, y_train)
    #rfc = LogisticRegression().fit(X_train, y_train)

    # evaluate model
    yhat = rfc.predict(X_test)
    y_pred_prob = rfc.predict_proba(X_test)[:,1]
    # store
    y_predprob.append(y_pred_prob)
    y_true.append(y_test['isSOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy

filename = '/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/models/basic_loocv_rfc.sav' 
pkl.dump(rfc, open(filename, 'wb'))
"""
#%% 
"""
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
plt.title('FPR vs. TPR ROC Curve of RFC Testing Performance')
plt.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ML/basic_loocv_rfc/auc.png', dpi = 300)

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

rfc_confusion = C_M(y_true_clean, y_pred_clean)
rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
plt.figure(figsize=(6,4))
sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
plt.title("Confusion Matrix for RFC test set predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ML/basic_loocv_rfc/confusion.png', dpi = 300)
"""
# %%
########################
# LEAVE ONE OUT + PCA + GRID SEARCH - RANDOM FOREST CLASSIFIER
# ########################

#Split the data according to IDs 
#from all_feats dataframe, get the unique id's
unique_ids = all_feats['id'].unique()
#split into two lists of unique ids in a random order
np.random.shuffle(unique_ids)

#create LeaveOneOut model
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()

# Initialize the model and fit it on the training set
# enumerate splits
y_true, y_pred = list(), list()
y_predprob = list()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

for train_ix, test_ix in LOO.split(unique_ids):
    print('startng training')
    #get data
    X_train = all_feats[all_feats['id'].isin(unique_ids[train_ix])]
    X_test = all_feats[all_feats['id'].isin(unique_ids[test_ix])]
    y_train = X_train[['isSOZ']]
    y_test = X_test[['isSOZ']]
    #drop columns 'isSOZ' and 'id'
    X_train = X_train.drop(columns = ['isSOZ', 'id'])
    X_test = X_test.drop(columns = ['isSOZ', 'id'])

    #PCA
    # 1. Refit and transform on training with parameter n (as deduced from the last step)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 2. Transform on Testing Set and store it as `X_test_pca`
    pca = PCA(random_state = 42, n_components = 6) #if we are using standard scaler... n_components = 6, if not we use 2.
    pca.fit(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    X_train_pca = pca.transform(X_train_s)

    # fit model
    rfc = RandomForestClassifier()
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8, None],
        'criterion' :['gini', 'entropy']
    }
    grid_RFC = GridSearchCV(rfc, param_grid = param_grid, scoring = 'recall')
    grid_RFC.fit(X_train_pca, y_train)

    #Predict values based on new parameters
    yhat = grid_RFC.predict(X_test_pca)
    y_pred_prob = grid_RFC.predict_proba(X_test_pca)[:,1]
    # store
    y_predprob.append(y_pred_prob)
    y_true.append(y_test['isSOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy

#%% 

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
plt.title('FPR vs. TPR ROC Curve of RFC Testing Performance')
plt.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ML/LOO_PCA_RFC_auc.png', dpi = 300)

################ Confusion Matrix
from sklearn.metrics import confusion_matrix as C_M
import seaborn as sns

rfc_confusion = C_M(y_true_clean, y_pred_clean)
rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
plt.figure(figsize=(6,4))
sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
plt.title("Confusion Matrix for RFC test set predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ML/LOO_PCA_RFC_confusion.png', dpi = 300)

#finally save the model & outputs, did this last in case any thing else crashes
pkl.dump(y_true_clean, open('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/models/gridsearch/y_true.pkl', 'wb'))
pkl.dump(y_pred_clean, open('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/models/gridsearch/y_pred.pkl', 'wb'))
pkl.dump(y_predprob_clean, open('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/models/gridsearch/y_predprob.pkl', 'wb'))

filename = '/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/models/gridsearch/gridsearch_loocv_rfc.sav' 
pkl.dump(grid_RFC, open(filename, 'wb'))


 # %%
"""
########################
# LEAVE ONE OUT - RANDOM FOREST CLASSIFIER (NULL MODEL)
# ########################

FEATURE = 'rise_amp'

#Split the data according to IDs 

#from all_feats dataframe, get the unique id's
unique_ids = all_feats['id'].unique()
#split into two lists of unique ids in a random order
np.random.shuffle(unique_ids)

#create LeaveOneOut model
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()

# Initialize the model and fit it on the training set
# enumerate splits
y_true, y_pred = list(), list()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

for train_ix, test_ix in LOO.split(unique_ids):

    #get data
    X_train = all_feats[all_feats['id'].isin(unique_ids[train_ix])]
    X_test = all_feats[all_feats['id'].isin(unique_ids[test_ix])]
    y_train = X_train[['isSOZ']]
    y_test = X_test[['isSOZ']]
    #drop columns 'isSOZ' and 'id'
    X_train = X_train[[FEATURE]]
    X_test = X_test[[FEATURE]]
    # fit model
    rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = None).fit(X_train, y_train)
    #rfc = LogisticRegression().fit(X_train, y_train)
    # evaluate model
    yhat = rfc.predict_proba(X_test)[:,1]
    # store
    y_true.append(y_test['isSOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy
    
#%% 
################# evaluate predictions
from sklearn.metrics import accuracy_score
y_true_clean = [x for x in y_true for x in x]
y_pred_clean = [x for x in y_pred for x in x]

# acc = accuracy_score(y_true_clean, y_pred_clean)
# print('Accuracy: %.3f' % acc)

################ AUC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

RocCurveDisplay.from_predictions(y_true_clean, y_pred_clean)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--', color='black')
plt.grid()
plt.title(f'FPR vs. TPR ROC Curve LOO-RFC ({FEATURE})')

################ Confusion Matrix
# from sklearn.metrics import confusion_matrix as C_M
# import seaborn as sns

# rfc_confusion = C_M(y_true_clean, y_pred_clean)
# rfc_conf_mat_df = pd.DataFrame(rfc_confusion)
# plt.figure(figsize=(6,4))
# sns.heatmap(rfc_conf_mat_df, cmap='GnBu', annot=True, fmt = "g")
# plt.title("Confusion Matrix for LOO-RFC (spikerate)")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
"""