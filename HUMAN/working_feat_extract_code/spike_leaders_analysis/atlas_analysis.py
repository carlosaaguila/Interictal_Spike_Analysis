#%% 
#set up environment
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings
#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *
from morphology_pipeline import *

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

# get ROI's from akash's list:
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

"""
#code to replace the atlas labels with the mni labels

#load the spikes leaders
atlas_spikes = pd.read_csv('../working features/clean_spikeleads/clean_atlas_leaders.csv')

# if final_label is in the dkt_labels, replace it with the mni_label
for i in range(len(atlas_spikes)):
    for j in range(len(mni_labels)):
        if atlas_spikes['final_label'][i] in dkt_labels[j]:
            atlas_spikes['final_label'][i] = atlas_spikes['final_label'][i].replace(atlas_spikes['final_label'][i], mni_labels[j])
            break
"""

atlas_spikes = pd.read_csv('../working features/clean_spikeleads/mni_corrected_leads.csv', index_col = 0)
#remove any nan final_label 
atlas_spikes = atlas_spikes.dropna(subset = ['final_label']).reset_index(drop = True)
#remove the labels that are not in mni_labels
atlas_spikes = atlas_spikes.loc[atlas_spikes['final_label'].isin(mni_labels)].reset_index(drop = True)

#replace ' right insula' with 'Insula_R' in final_label
atlas_spikes['final_label'] = atlas_spikes['final_label'].apply(lambda x: x.replace(' right insula', 'Insula_R'))

#find the median value for each final_label and each patient
median_vals = atlas_spikes.groupby(['final_label', 'pt_id']).median().reset_index()

#%%
#merge the SOZ designation for each patient
SOZ_list = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/soz_locations.csv', index_col = 0)

#merge the SOZ designation for each patient
median_vals = median_vals.merge(SOZ_list, left_on = 'pt_id', right_on = 'name', how = 'left')

# function to match SOZ and ROI
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

#%%
#dictionary containing the SOZ and their corresponding ROI
dict_soz = {'bilateral - mesial temporal':'R_Mesial', 'bilateral - mesial temporal':'L_Mesial',
             'bilateral - temporal neocortical':'R_Lateral', 'bilateral - temporal neocortical':'L_Lateral',
             'right - temporal neocortical':'R_Lateral', 'right - mesial temporal':'R_Mesial', 
             'left - temporal neocortical':'L_Lateral', 'left - mesial temporal':'L_Mesial',
             'right - other cortex':'R_OtherCortex', 'left - other cortex':'L_OtherCortex',
             'left - temporal':'L_Lateral', 'right - temporal':'R_Lateral',
             'left - temporal':'L_Mesial', 'right - temporal':'R_Mesial'
             }


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
             'left - temporal':'Temporal_Inf_L', 'left - temporal':'Temporal_Mid_L', 
             'left - temporal':'Temporal_Sup_L', 'left - temporal':'Fusiform_L',
             'left - temporal':'Amyg_Hipp_L', 'left - temporal':'ParaHippocampal_L',
            'right - temporal':'Temporal_Inf_R', 'right - temporal':'Temporal_Mid_R',
            'right - temporal':'Temporal_Sup_R', 'right - temporal':'Fusiform_R',
            'right - temporal':'Amyg_Hipp_R', 'right - temporal':'ParaHippocampal_R'
            'bilateral - temporal':'Temporal_Inf_L', 'bilateral - temporal':'Temporal_Mid_L', 'bilateral - temporal':'Temporal_Sup_L', 'bilateral - temporal':'Fusiform_L',
            'bilateral - temporal':'Temporal_Inf_R', 'bilateral - temporal':'Temporal_Mid_R', 'bilateral - temporal':'Temporal_Sup_R', 'bilateral - temporal':'Fusiform_R',
            'bilateral - temporal':'Amyg_Hipp_L', 'bilateral - temporal':'Amyg_Hipp_R', 'bilateral - temporal':'ParaHippocampal_L', 'bilateral - temporal':'ParaHippocampal_R'
             }

#CLASS LIST COMBINATION
to_combine = ['bilateral - diffuse', 'bilateral - mesial temporal', 'bilateral - multifocal' , 'bilateral - temporal multifocal','diffuse - diffuse', 'left - diffuse' ,'left - multifocal', 'right - multifocal']
to_remove = ['frontal']

#remove SOZ's that are weird (frontal and temporal)
for l in to_remove:
    median_vals = median_vals[median_vals['region'] != l]

#combine the SOZ and region labels
median_vals['soz'] = median_vals['lateralization'] + " - " + median_vals['region'] 
#combine the bilateral and diffuse labels
median_vals['soz'] = median_vals['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#WHEN I PICK UP TOMORROW: HERE YOU WANT TO USE CONVERT TF TO AGGREGATE THE NEW 1/0 SOZ LABELS
# WORK ON A DRAFT OF THE POWERPOINT


# %%
#show the distribution of the labels
plt.figure(figsize = (10,10))
sns.countplot(y = atlas_spikes['final_label'], order = atlas_spikes['final_label'].value_counts().index)
plt.show()

# %%
