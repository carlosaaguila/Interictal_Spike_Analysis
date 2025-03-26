#%%
#Grab the HUP DATASET#
######################
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

# Import custom functions
import sys, os
code_v2_path = os.path.dirname('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/')
sys.path.append(code_v2_path)
from get_iEEG_data2 import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *
import pywt
from tqdm import tqdm
from datetime import datetime

code_path = os.path.dirname('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

data_directory = ['/mnt/sauce/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/sauce/littlab/data/Human_Data']
drop_pts = ['HUP093','HUP108','HUP113','HUP114','HUP116','HUP123','HUP087','HUP099','HUP111','HUP121','HUP105','HUP106','HUP107','HUP159'] #These are the patients with less than 8 contacts.
#load in both spike dataframes for HUP
spikes_thresh = pd.read_csv('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
spikes_thresh = spikes_thresh[~spikes_thresh['pt_id'].isin(drop_pts)]

all_spikes = spikes_thresh
#WHAT DO YOU WANT TO REMOVE FROM THE CORE PLOT (CHOICES: 'frontal','mesial temporal','other cortex', 'temporal neocortical','temporal')
soz_to_remove = ['temporal']

#channels to keep 

# chs_tokeep = ['RA','LA','RDA','LDA','LH', 'RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']
chs_tokeep = ['RA','LA','RDA','LDA','LDH','RDH','LHD', 'RHD','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']

#if channel_label contains any of the strings in chs_tokeep, keep it
all_spikes = all_spikes[all_spikes['channel_label'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

#only take the electrode channels that are in the same side
left_spikes = all_spikes[all_spikes['lateralization'].str.contains('left')].reset_index(drop=True)
left_spikes_tokeep = left_spikes[~left_spikes['channel_label'].str.contains('R')].reset_index(drop=True)

right_spikes = all_spikes[all_spikes['lateralization'].str.contains('right')].reset_index(drop=True)
right_spikes_tokeep = right_spikes[~right_spikes['channel_label'].str.contains('L')].reset_index(drop=True)

bilateral_spikes = all_spikes[all_spikes['lateralization'].str.contains('bilateral')].reset_index(drop=True)

#concat them back into all_spikes
all_spikes = pd.concat([left_spikes_tokeep, right_spikes_tokeep, bilateral_spikes], axis =0).reset_index(drop=True)

#get only the spikes that contain 'mesial temporal' in the SOZ column
mesial_temp_spikes = all_spikes[all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

# grab the remaining spikes that aren't in mesial_temp_spikes
non_mesial_temp_spikes = all_spikes[~all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

#remove any 'channel_label' that contains the letter T or F
# mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)
# non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)

mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RCB|Z')].reset_index(drop=True)
non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RCB|Z')].reset_index(drop=True)

all_spikes = pd.concat([mesial_temp_spikes, non_mesial_temp_spikes], axis=0).reset_index(drop=True)

def soz_assigner(row):
    if row['SOZ'] == 'mesial temporal':
        return 1
    elif row['SOZ'] == 'temporal neocortical':
        return 2
    elif row['SOZ'] == 'other cortex':
        return 3
    elif row['SOZ'] == "frontal":
        return 3
    else:
        return None

all_spikes['SOZ'] = all_spikes.apply(soz_assigner, axis = 1)
HUP_SPIKES = all_spikes

#%%

#Grab the MUSC DATASET#
#######################

## load the spike data
MUSC_spikes = pd.read_csv('../dataset/complete_dfs/MUSC_thresholded.csv', index_col=0)

#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]
MUSC_sozs = MUSC_sozs.drop(columns=['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'])

#fix SOZ and laterality
MUSC_spikes = MUSC_spikes.merge(MUSC_sozs, left_on = 'pt_id', right_on = 'ParticipantID', how = 'inner')
MUSC_spikes = MUSC_spikes.drop(columns=['ParticipantID','Site_1MUSC_2Emory','IfNeocortical_Location','Correction Notes','lateralization_left','lateralization_right','region'])

#find the patients that should be null, and remove them for the full dataset
nonnan_mask = MUSC_sozs.dropna()
pts_to_remove = nonnan_mask[nonnan_mask['Correction Notes'].str.contains('null')]['ParticipantID'].array
MUSC_spikes = MUSC_spikes[~MUSC_spikes['pt_id'].isin(pts_to_remove)]
MUSC_full = MUSC_spikes

all_spikes = MUSC_full
#########################
# 1. Organize the data  #
#########################

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
        return None
    elif row['Other'] == 1:
        return 3
    else:
        return None


all_spikes['SOZ'] = all_spikes.apply(soz_assigner, axis = 1)

MUSC_SPIKES = all_spikes

#%%
# to check localizations
rid = pd.read_csv('/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pt_ids.csv')

pt_ids = np.array([' 3T_MP0015', ' 3T_MP0020', '3T_MP0001', '3T_MP0003', '3T_MP0004',
       '3T_MP0006', '3T_MP0010', '3T_MP0013', '3T_MP0014', '3T_MP0018',
       '3T_MP0019', '3T_MP0022', '3T_MP0026', '3T_MP0031', '3T_MP0035',
       '3T_MP0036', '3T_MP0037', '3T_MP0038', '3T_MP0042', '3T_MP0043',
       '3T_MP0044', '3T_MP0045', 'HUP088', 'HUP126', 'HUP127', 'HUP133',
       'HUP135', 'HUP138', 'HUP140', 'HUP141', 'HUP142', 'HUP162',
       'HUP163', 'HUP164', 'HUP165', 'HUP181', 'HUP185', 'HUP187',
       'HUP190', 'HUP191', 'HUP192', 'HUP199', 'HUP202', 'HUP204',
       'HUP205', 'HUP218', 'HUP219', 'HUP221', 'HUP223', 'HUP224',
       'HUP130', 'HUP131', 'HUP134', 'HUP139', 'HUP145', 'HUP146',
       'HUP150', 'HUP151', 'HUP158', 'HUP161', 'HUP166', 'HUP170',
       'HUP171', 'HUP172', 'HUP179', 'HUP180', 'HUP188', 'HUP196',
       '3T_MP0011', '3T_MP0028', 'HUP128', 'HUP189', 'HUP207', 'HUP209',
       'HUP215', 'HUP225'], dtype=object)

# %%

#Strip the numerical portion of the electrode (channel_label)
# Strip numbers from channel labels using regex
# Strip numbers from end of channel labels
MUSC_SPIKES['channel_label'] = MUSC_SPIKES['channel_label'].str.extract('([A-Za-z]+)', expand=False)
HUP_SPIKES['channel_label'] = HUP_SPIKES['channel_label'].str.extract('([A-Za-z]+)', expand=False)

# Remove L and R from channel labels
MUSC_SPIKES['channel_label'] = MUSC_SPIKES['channel_label'].str.replace('L', '').str.replace('R', '')
HUP_SPIKES['channel_label'] = HUP_SPIKES['channel_label'].str.replace('L', '').str.replace('R', '')

HUP_SPIKES = HUP_SPIKES[HUP_SPIKES['pt_id'].isin(pt_ids)]
MUSC_SPIKES = MUSC_SPIKES[MUSC_SPIKES['pt_id'].isin(pt_ids)]

#%% 
MUSC_SPIKES['channel_label'] = MUSC_SPIKES['channel_label'].str.replace('AH', 'B').str.replace('PH', 'C')
HUP_SPIKES['channel_label'] = HUP_SPIKES['channel_label'].str.replace('DA', 'A').str.replace('DB', 'B').str.replace('DC', 'C')

HUP_SPIKES = HUP_SPIKES[~HUP_SPIKES['channel_label'].isin(['AH','HD','AD'])]

combined = pd.concat([MUSC_SPIKES, HUP_SPIKES], axis=0)
channel_counts = combined.groupby(['SOZ', 'channel_label'])['pt_id'].nunique().unstack(fill_value=0)

from scipy.stats import chi2_contingency
data = np.array(channel_counts)
chi2, p_value, dof, expected = chi2_contingency(data)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

row_labels = ['MTL', 'Temporal Neocortical', 'Other Cortex']
col_labels = ['Electrode A', 'Electrode B', 'Electrode C']

# Create figure with a specific size and style
plt.figure(figsize=(5, 5))
# Create heatmap with lighter colors and annotations
ax = sns.heatmap(data, 
            annot=True,  # Show numbers in cells
            fmt='d',     # Format as integers
            cmap='Blues',  # Using a lighter color palette
            xticklabels=col_labels,
            yticklabels=row_labels,
            cbar_kws={'label': 'Count'},
            center=np.mean(data),  # Center the colormap
            vmin=0,  # Set minimum value for better color scaling
            annot_kws={'size': 16, 
                       'color':'red'})  # Make annotations larger

plt.title('Distribution of Counts Across SOZ Groups', pad=20, fontsize=14)
plt.xlabel('Channel', fontsize=12)
plt.ylabel('SOZ Group', fontsize=12)

# Adjust layout
plt.tight_layout()
# plt.savefig('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/figures/elec_counts/dist-elec.pdf')

# #%%
# # Count unique channels per patient
# musc_counts = MUSC_SPIKES.groupby('pt_id')['channel_label'].nunique().reset_index()
# hup_counts = HUP_SPIKES.groupby('pt_id')['channel_label'].nunique().reset_index()

# # Add SOZ info
# musc_soz = MUSC_SPIKES[['pt_id', 'SOZ']].drop_duplicates().dropna()
# hup_soz = HUP_SPIKES[['pt_id', 'SOZ']].drop_duplicates().dropna()

# musc_counts = musc_counts.merge(musc_soz, on='pt_id', how='left').dropna()
# hup_counts = hup_counts.merge(hup_soz, on='pt_id', how='left').dropna()

# # Combine datasets
# all_counts = pd.concat([musc_counts, hup_counts], axis=0)

# # Create SOZ_type column
# def soz_type_assigner(row):
#     if row['SOZ'] == 1:
#         return 'MTL'
#     elif row['SOZ'] == 2:
#         return 'Temporal Neocortical' 
#     elif row['SOZ'] == 3:
#         return 'Other Cortex'
#     else:
#         return None

# all_counts['SOZ_type'] = all_counts.apply(soz_type_assigner, axis=1)

# # Calculate Kruskal-Wallis H-test
# mtl_data = all_counts[all_counts['SOZ_type']=='MTL']['channel_label']
# neo_data = all_counts[all_counts['SOZ_type']=='Temporal Neocortical']['channel_label']
# other_data = all_counts[all_counts['SOZ_type']=='Other Cortex']['channel_label']

# h_stat, p_val = stats.kruskal(mtl_data, neo_data, other_data)
# print(f"Kruskal-Wallis test results:")
# print(f"H-statistic: {h_stat:.2f}")
# print(f"p-value: {p_val:.4f}")

# # Calculate median and IQR for each group
# for soz_type in ['MTL', 'Temporal Neocortical', 'Other Cortex']:
#     data = all_counts[all_counts['SOZ_type']==soz_type]['channel_label']
#     median = np.median(data)
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     iqr = q3 - q1
#     print(f"\n{soz_type}:")
#     print(f"Median: {median:.1f}")
#     print(f"IQR: {iqr:.1f} ({q1:.1f} - {q3:.1f})")

# # Create violin plot
# plt.figure(figsize=(8,6))

# my_palette = {'MTL':'#E64B35FF', 'Other Cortex':'#7E6148FF', 'Temporal Neocortical':'#3C5488FF'}
# pairs=[('MTL', 'Temporal Neocortical'), ('Temporal Neocortical', 'Other Cortex'), ('MTL', 'Other Cortex')]
# order = ['MTL', 'Temporal Neocortical', 'Other Cortex']
# ax = sns.violinplot(x='SOZ_type', y='channel_label', data=all_counts, palette=my_palette, order=order)
# plt.title('Number of Unique Electrodes by SOZ Type')
# plt.xlabel('SOZ Type')
# plt.ylabel('Number of Unique Channels')

