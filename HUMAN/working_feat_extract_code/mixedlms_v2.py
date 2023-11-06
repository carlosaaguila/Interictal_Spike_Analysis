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

# load the tables V1 - basic combination of all bilaterals
df_riseamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseamp.csv")
df_decayamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayamp.csv")
df_slowwidth = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowwidth.csv")
df_slowamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowamp.csv")
df_riseslope = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseslope.csv")
df_decayslope = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayslope.csv")
df_averageamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/averageamp.csv")
df_LL = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/LL.csv")

#load the saved tables V2 - seperated the bilateral tags into bilateral, bilateral MT (mesialtemp), or bilateral T (temporal)
df_riseamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseamp_v2.csv")
df_decayamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayamp_v2.csv")
df_slowwidth_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowwidth_v2.csv")
df_slowamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/slowamp_v2.csv")
df_riseslope_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/riseslope_v2.csv")
df_decayslope_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/decayslope_v2.csv")
df_averageamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/averageamp_v2.csv")
df_LL_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tablesv2/LL_v2.csv")

#load the mni parcellation tables:
df_riseamp_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/riseamp.csv")
df_decayamp_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/decayamp.csv")
df_slowwidth_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/slowwidth.csv")
df_slowamp_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/slowamp.csv")
df_riseslope_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/riseslope.csv")
df_decayslope_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/decayslope.csv")
df_averageamp_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/averageamp.csv")
df_LL_mni = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mni_atlas_LM_tables/LL.csv")

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
def convertTF(feature_matrix, dict_soz):
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

df_riseamp_v3_clean = convertTF(df_riseamp_v2_clean, dict_soz)
df_decayamp_v3_clean = convertTF(df_decayamp_v2_clean, dict_soz)
df_slowwidth_v3_clean = convertTF(df_slowwidth_v2_clean, dict_soz)
df_slowamp_v3_clean = convertTF(df_slowamp_v2_clean, dict_soz)
df_riseslope_v3_clean = convertTF(df_riseslope_v2_clean, dict_soz)
df_decayslope_v3_clean = convertTF(df_decayslope_v2_clean, dict_soz)
df_averageamp_v3_clean = convertTF(df_averageamp_v2_clean, dict_soz)
df_LL_v3_clean = convertTF(df_LL_v2_clean, dict_soz)

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

#%% Reepat the same process for the MNI datasets:
#remove bilaterals:
df_riseamp_mni_clean = df_riseamp_mni[df_riseamp_mni['soz'] != 'bilateral']
df_decayamp_mni_clean = df_decayamp_mni[df_decayamp_mni['soz'] != 'bilateral']
df_slowwidth_mni_clean = df_slowwidth_mni[df_slowwidth_mni['soz'] != 'bilateral']
df_slowamp_mni_clean = df_slowamp_mni[df_slowamp_mni['soz'] != 'bilateral']
df_riseslope_mni_clean = df_riseslope_mni[df_riseslope_mni['soz'] != 'bilateral']
df_decayslope_mni_clean = df_decayslope_mni[df_decayslope_mni['soz'] != 'bilateral']
df_averageamp_mni_clean = df_averageamp_mni[df_averageamp_mni['soz'] != 'bilateral']
df_LL_mni_clean = df_LL_mni[df_LL_mni['soz'] != 'bilateral']

#dictionary containing the SOZ and their corresponding ROI
dict_soz2 = {'bilateral - mesial temporal':'Amyg_Hipp_L', 'bilateral - mesial temporal':'Amyg_Hipp_R',
             'bilateral - mesial temporal':'ParaHippocampal_R', 'bilateral - mesial temporal':'ParaHippocampal_L',
             'left - mesial temporal':'Amyg_Hipp_L', 'left - mesial temporal':'ParaHippocampal_L',
             'right - mesial temporal':'Amyg_Hipp_R', 'right - mesial temporal':'ParaHippocampal_R',
             'bilateral - temporal neocortical':'Temporal_Inf_L', 'bilateral - temporal neocortical':'Temporal_Inf_R',
             'bilateral - temporal neocortical':'Temporal_Mid_L', 'bilateral - temporal neocortical':'Temporal_Mid_R',
             'bilateral - temporal neocortical':'Temporal_Sup_L', 'bilateral - temporal neocortical':'Temporal_Sup_R',
             'bilateral - temporal neocortical':'Fusiform_L', 'bilateral - temporal neocortical':'Fusiform_R', 
             'left - temporal neocortical':'Temporal_Inf_L', 'left - temporal neocortical':'Temporal_Mid_L', 
             'left - temporal neocortical':'Temporal_Sup_L', 'left - temporal neocortical':'Fusiform_L',
             'right - temporal neocortical':'Temporal_Inf_R', 'right - temporal neocortical':'Temporal_Mid_R', 
             'right - temporal neocortical':'Temporal_Sup_R', 'right - temporal neocortical':'Fusiform_R',
             'left - other cortex':'Cingulum_L', 'left - other cortex':'FMO_Rect_L', 'left - other cortex':'Frontal_Mid_All_L', 
             'left - other cortex':'Frontal_Sup_All_L', 'left - other cortex':'Frontal_inf_All_L', 'left - other cortex':'Insula_L', 
             'left - other cortex':'Occipital_Lat_L', 'left - other cortex':'Occipital_Med_L', 'left - other cortex':'Parietal_Sup_Inf_L', 
             'left - other cortex':'Postcentral_L', 'left - other cortex':'Precentral_L', 'left - other cortex':'Precuneus_PCL_L', 
             'left - other cortex':'SupraMarginal_Angular_L', 'left - other cortex':'thalam_limbic_L',
             'right - other cortex':'Cingulum_R', 'right - other cortex':'FMO_Rect_R', 'right - other cortex':'Frontal_Mid_All_R', 
             'right - other cortex':'Frontal_Sup_All_R', 'right - other cortex':'Frontal_inf_All_R', 
             'right - other cortex':'Insula_R', 'right - other cortex':'Occipital_Lat_R', 'right - other cortex':'Occipital_Med_R', 
             'right - other cortex':'Parietal_Sup_Inf_R', 'right - other cortex':'Postcentral_R', 'right - other cortex':'Precentral_R', 
             'right - other cortex':'Precuneus_PCL_R', 'right - other cortex':'SupraMarginal_Angular_R', 'right - other cortex':'thalam_limbic_R'
             }

df_riseamp_mni_convert = convertTF(df_riseamp_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_decayamp_mni_convert = convertTF(df_decayamp_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_slowwidth_mni_convert = convertTF(df_slowwidth_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_slowamp_mni_convert = convertTF(df_slowamp_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_riseslope_mni_convert = convertTF(df_riseslope_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_decayslope_mni_convert = convertTF(df_decayslope_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_averageamp_mni_convert = convertTF(df_averageamp_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})
df_LL_mni_convert = convertTF(df_LL_mni_clean, dict_soz2).drop(columns = ['soz']).rename(columns = {'soz2':'soz'})

#get the average
df_riseamp_mni_avg= df_riseamp_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_decayamp_mni_avg = df_decayamp_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_slowwidth_mni_avg = df_slowwidth_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_slowamp_mni_avg = df_slowamp_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_riseslope_mni_avg = df_riseslope_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_decayslope_mni_avg = df_decayslope_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_averageamp_mni_avg = df_averageamp_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()
df_LL_mni_avg = df_LL_mni_convert.groupby(['pt','roi','soz']).median().reset_index().dropna()

#%% put dataframes into lists for for loops (mixed LMS)
feature_avg_list = [df_riseamp_avg_clean, df_decayamp_avg_clean, df_slowwidth_avg_clean, df_slowamp_avg_clean, df_riseslope_avg_clean, df_decayslope_avg_clean, df_averageamp_avg_clean, df_LL_avg_clean]
feature_list = [df_riseamp_v3_clean, df_decayamp_v3_clean, df_slowwidth_v3_clean, df_slowamp_v3_clean, df_riseslope_v3_clean, df_decayslope_v3_clean, df_averageamp_v3_clean, df_LL_v3_clean]
mni_list = [df_riseamp_mni_convert, df_decayamp_mni_convert, df_slowwidth_mni_convert, df_slowamp_mni_convert, df_riseslope_mni_convert, df_decayslope_mni_convert, df_averageamp_mni_convert, df_LL_mni_convert]
mni_avg_list = [df_riseamp_mni_avg, df_decayamp_mni_avg, df_slowwidth_mni_avg, df_slowamp_mni_avg, df_riseslope_mni_avg, df_decayslope_mni_avg, df_averageamp_mni_avg, df_LL_mni_avg]
feature = ['amp', 'amp', 'width', 'amp', 'slope', 'slope', 'amp', 'LL']
titles = ['Rise Amp', 'Decay Amp', 'Slow Width', 'Slow Amp', 'Rise Slope', 'Decay Slope', 'Average Amp', 'LL']

#%% Forest Plots function
def forest_plots(model, title):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12
    params = model.params
    conf = model.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['2.5%', '97.5%', 'Odds Ratio']# convert log odds to ORs
    odds = pd.DataFrame((conf))# check if pvalues are significant
    odds['pvalues'] = model.pvalues
    odds['significant?'] = ['significant' if pval <= 0.05 else 'not significant' for pval in model.pvalues]

    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(10, 10), dpi=300)
    for idx, row in odds.iloc[::-1].iterrows():
        ci = [[row['Odds Ratio'] - row[::-1]['2.5%']], [row['97.5%'] - row['Odds Ratio']]]
        if row['significant?'] == 'significant':
            plt.errorbar(x=[row['Odds Ratio']], y=[row.name], xerr=ci,
                ecolor='tab:red', capsize=3, linestyle='None', linewidth=1, marker="o", 
                        markersize=5, mfc="tab:red", mec="tab:red")
        else:
            plt.errorbar(x=[row['Odds Ratio']], y=[row.name], xerr=ci,
                ecolor='tab:gray', capsize=3, linestyle='None', linewidth=1, marker="o", 
                        markersize=5, mfc="tab:gray", mec="tab:gray")
        plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Odds Ratio and 95% Confidence Interval', fontsize=10)
    plt.tight_layout()
    plt.title('Forest Plot of {}'.format(title), fontsize=12)
    # plt.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/forest_plots/MNI/{}.png'.format(title), dpi=300)
    plt.show()
    return odds, fig
#%% mixed LM - average feature
for i, (feat, list) in enumerate(zip(feature, feature_avg_list)):
    md = smf.mixedlm("{} ~ C(soz) + C(roi)".format(feat), list, groups="pt")
    mdf = md.fit()
    print("{} --- MIXED LM RESULTS".format(titles[i]))
    print(mdf.summary())
    print(mdf.pvalues)
    odds, fig = forest_plots(mdf, titles[i])


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


# %% mixed LM  - MNI atlas table (convert tables)
for i, (feat, list) in enumerate(zip(feature, mni_avg_list)):
    print(titles[i])
    md = smf.mixedlm('{} ~ C(soz) + C(roi, Treatment(reference = "Temporal_Mid_L"))'.format(feat), list, groups="pt")
    mdf = md.fit()
    print("{} --- MIXED LM RESULTS".format(titles[i]))
    print(mdf.summary())
    print(mdf.pvalues)
    odds, fig = forest_plots(mdf, titles[i])

# %% 2-WAY ANOVA MODEL - MNI atlas table

for i,(feat, list) in enumerate(zip(feature, mni_avg_list)):
    model = ols('{} ~ C(soz) + C(roi) + C(soz):C(roi)'.format(feat), data=list, groups = 'pt').fit()
    result = sm.stats.anova_lm(model, type=2)
    print("{} --- 2-WAY ANOVA RESULTS".format(titles[i]))
    print(result)
    print('====================================================================================================')

# %%
