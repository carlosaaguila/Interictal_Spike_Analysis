#%% 
#set up environment
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import scipy.stats as stats
import warnings
#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

#establish ROI's
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
emptylabel = ['EmptyLabel','NaN']
L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']
roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]

#%% get ROI's from akash's list:
mni_dkt = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/dkt_mni_v2.xlsx', sheet_name = 'Sheet1')
mni_dkt = mni_dkt[['Label 0: background', 'name']].dropna().reset_index(drop = True).rename(columns = {"Label 0: background":"DKT", "name":"MNI"})
mni_dkt["DKT"] = mni_dkt['DKT'].apply(lambda x: x.split(':')[-1] + ' ')
mni_labels = np.unique(mni_dkt['MNI'].to_numpy())
#get the DKT labels for each MNI labels
dkt_labels = []
for i in mni_labels:
    subdf = mni_dkt.loc[mni_dkt['MNI'] == i]
    dkt_labels.append(subdf['DKT'].to_list())
#pass dkt_labels as roilist if you want 40 ROIs, that correspond to the names in the mni_labels

# %% load all tables w/ new features

# load the tables V1
df_riseamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseamp.csv")
df_decayamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayamp.csv")
df_slowwidth = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowwidth.csv")
df_slowamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowamp.csv")
df_riseslope = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseslope.csv")
df_decayslope = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayslope.csv")
df_averageamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/averageamp.csv")
df_LL = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/LL.csv")

#load the saved tables V2
df_riseamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseamp_v2.csv")
df_decayamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayamp_v2.csv")
df_slowwidth_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowwidth_v2.csv")
df_slowamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowamp_v2.csv")
df_riseslope_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseslope_v2.csv")
df_decayslope_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayslope_v2.csv")
df_averageamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/averageamp_v2.csv")
df_LL_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/LL_v2.csv")


# %% clean up the tables to remove bilateral and rename SOZ

#remove bilateral tags
df_riseamp_v2_clean = df_riseamp_v2[df_riseamp_v2['soz'] != 'bilateral']
df_decayamp_v2_clean = df_decayamp_v2[df_decayamp_v2['soz'] != 'bilateral']
df_slowwidth_v2_clean = df_slowwidth_v2[df_slowwidth_v2['soz'] != 'bilateral']
df_slowamp_v2_clean = df_slowamp_v2[df_slowamp_v2['soz'] != 'bilateral']
df_riseslope_v2_clean = df_riseslope_v2[df_riseslope_v2['soz'] != 'bilateral']
df_decayslope_v2_clean = df_decayslope_v2[df_decayslope_v2['soz'] != 'bilateral']
df_averageamp_v2_clean = df_averageamp_v2[df_averageamp_v2['soz'] != 'bilateral']
df_LL_v2_clean = df_LL_v2[df_LL_v2['soz'] != 'bilateral']

#dictionary containing the SOZ and their corresponding ROI
dict_soz = {'bilateral - mesial temporal':'R_Mesial', 'bilateral - mesial temporal':'L_Mesial',
             'bilateral - temporal neocortical':'R_Lateral', 'bilateral - temporal neocortical':'L_Lateral',
             'right - temporal neocortical':'R_Lateral', 'right - mesial temporal':'R_Mesial', 
             'left - temporal neocortical':'L_Lateral', 'left - mesial temporal':'L_Mesial',
             'right - other cortex':'R_OtherCortex', 'left - other cortex':'L_OtherCortex'}

#match SOZ to ROI using 1/0 to denote ROI is in the SOZ or not
def convertTF(feature_matrix):
    """
    Function to add 1/0 to match SOZ to ROI
    """
    df_riseamp_v3_clean = pd.DataFrame()
    for soz, roi in dict_soz.items():
        #riseamp v2 table
        subdf = feature_matrix.loc[feature_matrix['soz'] == soz]
        subdf = subdf.replace(to_replace = dict_soz)
        #change elements in soz to 1 or 0 if they match elements in roi
        subdf['soz2'] = subdf['soz'] == subdf['roi']
        subdf['soz2'] = subdf['soz2'].astype(int) #convert to int
        subdf['soz2'] = subdf['soz2'].astype('category')
        df_riseamp_v3_clean = pd.concat([df_riseamp_v3_clean, subdf], axis = 0)

    return df_riseamp_v3_clean

df_riseamp_v3_clean = convertTF(df_riseamp_v2_clean)
df_decayamp_v3_clean = convertTF(df_decayamp_v2_clean)
df_slowwidth_v3_clean = convertTF(df_slowwidth_v2_clean)
df_slowamp_v3_clean = convertTF(df_slowamp_v2_clean)
df_riseslope_v3_clean = convertTF(df_riseslope_v2_clean)
df_decayslope_v3_clean = convertTF(df_decayslope_v2_clean)
df_averageamp_v3_clean = convertTF(df_averageamp_v2_clean)
df_LL_v3_clean = convertTF(df_LL_v2_clean)

    #decayamp v2 table

#drop soz and rename soz 2
df_riseamp_v3_clean = df_riseamp_v3_clean.drop(columns = ['soz'])
df_riseamp_v3_clean = df_riseamp_v3_clean.rename(columns = {'soz2':'soz'})

df_decayamp_v3_clean = df_decayamp_v3_clean.drop(columns = ['soz'])
df_decayamp_v3_clean = df_decayamp_v3_clean.rename(columns = {'soz2':'soz'})

df_slowwidth_v3_clean = df_slowwidth_v3_clean.drop(columns = ['soz'])
df_slowwidth_v3_clean = df_slowwidth_v3_clean.rename(columns = {'soz2':'soz'})

df_slowamp_v3_clean = df_slowamp_v3_clean.drop(columns = ['soz'])
df_slowamp_v3_clean = df_slowamp_v3_clean.rename(columns = {'soz2':'soz'})  

df_riseslope_v3_clean = df_riseslope_v3_clean.drop(columns = ['soz'])
df_riseslope_v3_clean = df_riseslope_v3_clean.rename(columns = {'soz2':'soz'})

df_decayslope_v3_clean = df_decayslope_v3_clean.drop(columns = ['soz'])
df_decayslope_v3_clean = df_decayslope_v3_clean.rename(columns = {'soz2':'soz'})

df_averageamp_v3_clean = df_averageamp_v3_clean.drop(columns = ['soz'])
df_averageamp_v3_clean = df_averageamp_v3_clean.rename(columns = {'soz2':'soz'})

df_LL_v3_clean = df_LL_v3_clean.drop(columns = ['soz'])
df_LL_v3_clean = df_LL_v3_clean.rename(columns = {'soz2':'soz'})


#%%find the average feature for each pt for each roi

#riseamp
df_riseamp_avg_clean = df_riseamp_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_decayamp_avg_clean = df_decayamp_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_slowwidth_avg_clean = df_slowwidth_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_slowamp_avg_clean = df_slowamp_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_riseslope_avg_clean = df_riseslope_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_decayslope_avg_clean = df_decayslope_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_averageamp_avg_clean = df_averageamp_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_LL_avg_clean = df_LL_v3_clean.groupby(['pt','roi','soz']).median().reset_index().dropna()


#%% put dataframes into lists for for loops (mixed LMS)
feature_avg_list = [df_riseamp_avg_clean, df_decayamp_avg_clean, df_slowwidth_avg_clean, df_slowamp_avg_clean, df_riseslope_avg_clean, df_decayslope_avg_clean, df_averageamp_avg_clean, df_LL_avg_clean]
feature_list = [df_riseamp_v3_clean, df_decayamp_v3_clean, df_slowwidth_v3_clean, df_slowamp_v3_clean, df_riseslope_v3_clean, df_decayslope_v3_clean, df_averageamp_v3_clean, df_LL_v3_clean]
feature = ['amp', 'amp', 'width', 'amp', 'slope', 'slope', 'amp', 'LL']
titles = ['Rise Amp', 'Decay Amp', 'Slow Width', 'Slow Amp', 'Rise Slope', 'Decay Slope', 'Average Amp', 'LL']

#%% mixed LM - average feature
for i, (feat, list) in enumerate(zip(feature, feature_avg_list)):
    md = smf.mixedlm("{} ~ C(soz) + C(roi) + C(soz):C(roi)".format(feat), list, groups="pt")
    mdf = md.fit()
    print("{} --- MIXED LM RESULTS".format(titles[i]))
    print(mdf.summary())
    print(mdf.pvalues)

#%% mixed LM - all features
for i, (feat, list) in enumerate(zip(feature, feature_list)):
    md = smf.mixedlm("{} ~ C(soz) + C(roi) + C(soz):C(roi)".format(feat), list, groups="pt")
    mdf = md.fit()
    print("{} --- MIXED LM RESULTS".format(titles[i]))
    print(mdf.summary())

#%% 2-WAY ANOVA MODEL  
# Performing two-way ANOVA
for i,(feat, list) in enumerate(zip(feature, feature_avg_list)):
    model = ols('{} ~ C(soz) + C(roi) + C(soz):C(roi)'.format(feat), data=list).fit()
    result = sm.stats.anova_lm(model, type=2)
    print("{} --- 2-WAY ANOVA RESULTS".format(titles[i]))
    print(result)
    print('====================================================================================================')


#%% 2-WAY ANOVA MODEL - all features
# Performing two-way ANOVA
for i,(feat, list) in enumerate(zip(feature, feature_list)):
    model = ols('{} ~ C(soz) + C(roi) + C(soz):C(roi)'.format(feat), data=list).fit()
    result = sm.stats.anova_lm(model, type=2)
    print("{} --- 2-WAY ANOVA RESULTS".format(titles[i]))
    print(result)
    print('====================================================================================================')


# %% try to flip the feature and soz
for i, (feat, list) in enumerate(zip(feature, feature_avg_list)):
    md = smf.mixedlm("C(soz) ~ {}".format(feat), list, groups="pt")
    mdf = md.fit()
    print("{} --- MIXED LM RESULTS".format(titles[i]))
    print(mdf.summary())
    print(mdf.pvalues)
# %%
