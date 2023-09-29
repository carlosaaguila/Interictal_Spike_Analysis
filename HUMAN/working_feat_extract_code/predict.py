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

#%% load the data

SOZ_feats_new = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_feats.csv', index_col = 0)
nonSOZ_feats_new = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_feats.csv', index_col = 0)
 
# clean the data
SOZ_feats_new['isSOZ'] = 1
nonSOZ_feats_new['isSOZ'] = 0

# concatenate all the data vertically
all_feats = pd.concat([SOZ_feats_new, nonSOZ_feats_new], axis = 0)

#remove the 'id' column
# all_feats = all_feats.drop(columns = ['id'])

#shuffle and reset the index of all_feats
all_feats = all_feats.sample(frac=1).reset_index(drop=True)

#%% split train/test
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


# %% train a Random Forest Classifier

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
Random Forest Classifier R-squared score for    test set:	         0.6379378197560016
Random Forest Classifier R-squared score for    validation set:	     0.6194411648957103
"""

############### Precision, Recall, F1 Score
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, _ = precision_recall_fscore_support(y_test, rfc_y_pred, average='binary')
print('Precision: ', prec)
print('Recall: ', rec)
print('F1 Score: ', f1)

"""
TP = 1126
TN = 495
FP = 579
FN = 341

Given our confusion matrix, we see 579 False Positives and 341 False Negatives using our Random Forest Classifier given our test set.

Given this, we calculated our precision or the positive predictive value to understand the fraction 
of true postitives from all relevant positive predictions (true and false positives). Our precision from this model was 0.661.

We also measured recall or sensitivity, to understand the ratio of relevant instances that were predicted.
Our recall score for this model was 0.768.

Additiaonlly, we calculated an F1 score to display the model's accuracy as a harmonic mean of both the
precision and the recall in one metric. Our F1 score for this model was 0.710.
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
An ROC curve (reciever oparting characteristic curve) shows the performance of a classification model 
at different classifications thresholds by showing the ratio of the True Positive Rate by the
False Positive Rate. To further understand the ROC curve, we calculate the area under the curve (AUC).
This provides an aggregate measurement of performace that is equal to the probability that the model 
ranks a random positive example more highly than a random negative example.

For our Random Forest Classifier, we get a AUC of 0.61, signifying that our chances of 
predicting the correct value are a bit higher than chance. (50/50)
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

#%% Train a Support Vector Machine Classifier

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
Performance was not very good 
AUC = 0.47
"""
# %% Train Linear Regression Models

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
Random Forest Classifier R-squared score for    test set:	         0.6379378197560016
Random Forest Classifier R-squared score for    validation set:	     0.6194411648957103
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



# %%
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
from sklearn.ensemble import RandomForestClassifier
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
    # evaluate model
    yhat = rfc.predict(X_test)
    # store
    y_true.append(y_test['isSOZ'].to_numpy())
    y_pred.append(yhat)
    # calculate accuracy
    
#%% 
# evaluate predictions
from sklearn.metrics import accuracy_score
y_true_clean = [x for x in y_true for x in x]
y_pred_clean = [x for x in y_pred for x in x]

acc = accuracy_score(y_true_clean, y_pred_clean)
print('Accuracy: %.3f' % acc)
# %%
