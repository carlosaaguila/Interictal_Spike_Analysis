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
#show all rows in dataframe
pd.set_option('display.max_rows', None)

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

#merge the SOZ designation for each patient
SOZ_list = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/soz_locations.csv', index_col = 0)

#merge the SOZ designation for each patient
median_vals = median_vals.merge(SOZ_list, left_on = 'pt_id', right_on = 'name', how = 'left')

#dictionary containing the SOZ and their corresponding ROI
dict_soz = {'bilateral - mesial temporal':'R_Mesial', 'bilateral - mesial temporal':'L_Mesial',
             'bilateral - temporal neocortical':'R_Lateral', 'bilateral - temporal neocortical':'L_Lateral',
             'right - temporal neocortical':'R_Lateral', 'right - mesial temporal':'R_Mesial', 
             'left - temporal neocortical':'L_Lateral', 'left - mesial temporal':'L_Mesial',
             'right - other cortex':'R_OtherCortex', 'left - other cortex':'L_OtherCortex',
             'left - temporal':'L_Lateral', 'right - temporal':'R_Lateral',
             'left - temporal':'L_Mesial', 'right - temporal':'R_Mesial'
             }

SOZs = ['bilateral - mesial temporal', 'left - mesial temporal', 'right - mesial temporal', 
         'bilateral - temporal neocortical', 'left - temporal neocortical', 'right - temporal neocortical',
         'left - other cortex', 'right - other cortex',
         'left - temporal', 'right - temporal', 'bilateral - temporal']
regions = [
         ['Amyg_Hipp_L', 'Amyg_Hipp_R', 'ParaHippocampal_R', 'ParaHippocampal_L'],
         ['Amyg_Hipp_L', 'ParaHippocampal_L'],
         ['Amyg_Hipp_R', 'ParaHippocampal_R'],
         ['Temporal_Inf_L', 'Temporal_Mid_L', 'Temporal_Sup_L', 'Fusiform_L', 'Temporal_Inf_R', 'Temporal_Mid_R', 'Temporal_Sup_R', 'Fusiform_R'],
         ['Temporal_Inf_L', 'Temporal_Mid_L', 'Temporal_Sup_L', 'Fusiform_L'],
         ['Temporal_Inf_R', 'Temporal_Mid_R', 'Temporal_Sup_R', 'Fusiform_R'],
         ['Cingulum_L', 'FMO_Rect_L','Frontal_Mid_All_L','Frontal_Sup_All_L','Frontal_inf_All_L','Insula_L','Occipital_Lat_L','Occipital_Med_L','Parietal_Sup_Inf_L','Postcentral_L','Precentral_L','Precuneus_PCL_L','SupraMarginal_Angular_L','thalam_limbic_L'],
         ['Cingulum_R', 'FMO_Rect_R','Frontal_Mid_All_R','Frontal_Sup_All_R','Frontal_inf_All_R','Insula_R','Occipital_Lat_R','Occipital_Med_R','Parietal_Sup_Inf_R','Postcentral_R','Precentral_R','Precuneus_PCL_R','SupraMarginal_Angular_R','thalam_limbic_R'],
         ['Temporal_Inf_L', 'Temporal_Mid_L', 'Temporal_Sup_L', 'Fusiform_L','Amyg_Hipp_L', 'ParaHippocampal_L'],
         ['Temporal_Inf_R', 'Temporal_Mid_R', 'Temporal_Sup_R', 'Fusiform_R','Amyg_Hipp_R', 'ParaHippocampal_R'],
         ['Temporal_Inf_L', 'Temporal_Mid_L', 'Temporal_Sup_L', 'Fusiform_L','Temporal_Inf_R', 'Temporal_Mid_R', 'Temporal_Sup_R', 'Fusiform_R','Amyg_Hipp_L', 'Amyg_Hipp_R', 'ParaHippocampal_R', 'ParaHippocampal_L']
         ]

#CLASS LIST COMBINATION
to_combine = ['bilateral - diffuse', 'bilateral - mesial temporal', 'bilateral - multifocal' , 'bilateral - temporal multifocal','diffuse - diffuse', 'left - diffuse' ,'left - multifocal', 'right - multifocal']
to_remove = ['frontal']

#remove SOZ's that are weird (frontal and temporal)
for l in to_remove:
    median_vals = median_vals[median_vals['region'] != l]

#combine the SOZ and region labels
median_vals['soz'] = median_vals['lateralization'] + " - " + median_vals['region'] 

#remove nans in the soz
median_vals = median_vals.dropna(subset = ['soz']).reset_index(drop = True)

#create new column
median_vals['soz2'] = 0
#if the soz is in SOZs, then check if the regions is in the corresponding values, add a 1 to the soz2 column
for i in range(len(median_vals)):
    for j in range(len(SOZs)):
        if median_vals['soz'][i] == SOZs[j]:
            if median_vals['final_label'][i] in regions[j]:
                median_vals['soz2'][i] = 1
                break
            else:
                median_vals['soz2'][i] = 0
                break
        else:
            median_vals['soz2'][i] = 0

# %%
#show the distribution of the labels
plt.figure(figsize = (10,10))
sns.countplot(y = atlas_spikes['final_label'], order = atlas_spikes['final_label'].value_counts().index)
plt.show()

# %%
#load master elec file
master_elec = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/master_elecs.csv')
# %%
resected = master_elec[master_elec['resected'] == True]
#load 
allpt_ids = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/all_ptids.csv', index_col=0)

#merge the resected electrodes with the allpt_ids
resected = resected.merge(allpt_ids, left_on = 'rid', right_on = 'r_id', how = 'left')

#drop whichPts, r_id, and ptname
resected = resected[['rid','ptname','name','resected','label']]

# %%
resec_notes = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/resection_notes.csv')
resec_notes = resec_notes[['name','Surg','Loc']].dropna()

#remove the patients in resec_notes from resected
resected = resected[~resected['ptname'].isin(resec_notes['name'])].reset_index(drop = True)

# %%
#in resected, grab any label that says "temporal" or "hippocampus"
resected_slice = resected[resected['label'].str.contains('temporal|hippocampus|amygdala|hippocampal', case = False)].reset_index(drop = True)
# %%
#give me the patients in resected_slice in resected
resected_slice2 = resected[resected['ptname'].isin(resected_slice['ptname'])].reset_index(drop = True)
# %%
