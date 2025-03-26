#%%
import os
from os.path import join as ospj
from glob import glob
import sys

import numpy as np
import pandas as pd

#%%
hup_rids = pd.read_csv(ospj('/users','aguilac','CNTSurgicalRepositor-CarlosUpdatedOutcome_DATA_LABELS_2024-09-04_1518.csv'))[['Record ID','HUP Number']].dropna().rename(columns = {'HUP Number':'pt_id'})
# musc_rids = pd.read_csv(ospj('/users', 'aguilac','MUSCdems_cleaned.csv'))[['Record ID','IEEG Portal Number for data shared with Penn']].dropna().rename(columns = {'IEEG Portal Number for data shared with Penn':'pt_id'})
# musc_rids['pt_id'] = musc_rids['pt_id'].str.replace("3T_MP00","").str.replace("MP00",'').str.replace("_D01-D02","").str.replace("_D01","").astype(int)
# rids = pd.concat([hup_rids, musc_rids], axis = 0)

ids_we_have = pd.read_csv("/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/data/mtl_all_feats.csv", index_col=0).pt_id.str.replace('HUP','').str.replace(" 3T_MP00","").str.replace('3T_MP00','').astype(int).unique()
rids_we_want = hup_rids[hup_rids.pt_id.isin(ids_we_have)]

collected_elecs = "/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/collected_electrode2ROI/"
# for rid in rids_we_want['Record ID'].unique():

atlas_dkt = glob(ospj(collected_elecs, '*'))

# %%
atlas_dkt_filtered = []
kept_rids = []

for rid in rids_we_want['Record ID'].unique():
    rid_str = f"RID{str(rid).zfill(4)}"
    matching_files = [f for f in atlas_dkt if rid_str in f]
    
    # If we have both ieeg_recon and ieeg_recon_old, keep only ieeg_recon
    if len(matching_files) > 1:
        recon_files = [f for f in matching_files if 'ieeg_recon/' in f]
        recon_old_files = [f for f in matching_files if 'ieeg_recon_old/' in f]
        if recon_files and recon_old_files:
            matching_files = recon_files
            
    if matching_files:
        kept_rids.append(rid)
        atlas_dkt_filtered.extend(matching_files)

atlas_dkt = atlas_dkt_filtered
print(f"Kept {len(kept_rids)} RIDs: {sorted(kept_rids)}")
# %%
# %%
data_directory = ['/mnt/sauce/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/sauce/littlab/data/Human_Data']
drop_pts = ['HUP093','HUP108','HUP113','HUP114','HUP116','HUP123','HUP087','HUP099','HUP111','HUP121','HUP105','HUP106','HUP107','HUP159'] #These are the patients with less than 8 contacts.
#load in both spike dataframes for HUP
spikes_thresh = pd.read_csv('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
spikes_thresh = spikes_thresh[~spikes_thresh['pt_id'].isin(drop_pts)]

all_spikes = spikes_thresh
#WHAT DO YOU WANT TO REMOVE FROM THE CORE PLOT (CHOICES: 'frontal','mesial temporal','other cortex', 'temporal neocortical','temporal')
soz_to_remove = ['temporal']

#channels to keep 
# chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']
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

HUP_SPIKES = all_spikes
HUP_SPIKES.pt_id = HUP_SPIKES.pt_id.str.replace('HUP','').astype(int)

hup_spikes = HUP_SPIKES.groupby(['pt_id','channel_label','SOZ'])['spike_rate'].median().reset_index()

#merge the record ID with hup_spikes
hup_spikes = pd.merge(hup_spikes, hup_rids[['pt_id','Record ID']], on='pt_id', how='left')


#%%
big_table = pd.DataFrame()
for i, atlas_file in enumerate(atlas_dkt):
    # Load atlas csv
    atlas_df = pd.read_csv(atlas_file)[['labels','roi']]
    rid_oi = kept_rids[i]
    atlas_df['Record ID'] = rid_oi
    big_table = pd.concat([big_table, atlas_df], axis =0)


# %%
merged_hup_spikes = hup_spikes.merge(big_table, left_on = ['channel_label','Record ID'], right_on=['labels','Record ID'], how = 'inner')

#%%
mesial = ['Hippocampus','Amygdala','parahippocampal','entorhinal']
# Use str.contains with | to join multiple search terms
mesial_channels = merged_hup_spikes[merged_hup_spikes['roi'].str.contains('|'.join(mesial), case=False)].reset_index(drop=True)
print(mesial_channels.roi.sort_values().unique())

def soz_assigner(row):
    if row['SOZ'] == 'temporal neocortical':
        return int(2)
    elif row['SOZ'] == 'other cortex':
        return int(3)
    elif row['SOZ'] == 'frontal':
        return int(3)
    elif row['SOZ'] == 'mesial temporal':
        return int(1)
    else:
        return None

mesial_channels['SOZ'] = mesial_channels.apply(soz_assigner, axis = 1)

# %%
from plot_soz_correlations import *

plt.rcParams['font.family'] = 'Arial'
# cohen, cliff = plot_soz_correlations(
#     mesial_channels,
#     ['spike_rate'],
#     x_labels=[''],
#     title='Spike Rates for MTL-labeled channels',
#     figsize = (10,6),
#     comparisons_correction= None,
# )

# print("Effect Sizes for spike rate:\n", cohen)
# print("\nCliff's delta effect sizes for spike rate:\n", cliff)


yo = mesial_channels.groupby(['pt_id','SOZ'])['spike_rate'].median().reset_index()
cohen, cliff = plot_soz_correlations(
    yo,
    ['spike_rate'],
    x_labels=[''],
    title='Spike Rates for MTL-labeled channels',
    figsize = (10,6),
    comparisons_correction= "Benjamini-Hochberg",
    path = '/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/figures/mtl-spikes/spikerate-pearson.pdf'
)

print("Effect Sizes for spike rate:\n", cohen)
print("\nCliff's delta effect sizes for spike rate:\n", cliff)

#%%
from scipy.stats import f_oneway, levene, shapiro, kruskal
_, p_levene = levene(yo[yo['SOZ'] == 1]['spike_rate'], yo[yo['SOZ'] == 2]['spike_rate'], yo[yo['SOZ'] == 3]['spike_rate'])

#%%
# Get count of channels per patient
channel_counts = mesial_channels.groupby(['pt_id','SOZ'])['channel_label'].count().reset_index()
channel_counts = channel_counts.rename(columns={'channel_label': 'channel_count'})

# print("\nNumber of mesial channels per patient:")
# print(channel_counts.sort_values('channel_count', ascending=False))

print("\nSummary statistics of mesial channels per patient:")
print(channel_counts.groupby('SOZ')['channel_count'].describe())

print('analysis for # of mtl contacts:\n')
h_stat, p_val = kruskal(channel_counts[channel_counts['SOZ'] == 1]['channel_count'], 
                        channel_counts[channel_counts['SOZ'] == 2]['channel_count'],
                        channel_counts[channel_counts['SOZ'] == 3]['channel_count'])

print("\nKruskal-Wallis test results:")
print(f"H-statistic: {h_stat:.3f}")
print(f"p-value: {p_val:.3e}")

# %%
# Perform Kruskal-Wallis H-test
h_stat, p_val = kruskal(yo[yo['SOZ'] == 1]['spike_rate'], 
                        yo[yo['SOZ'] == 2]['spike_rate'],
                        yo[yo['SOZ'] == 3]['spike_rate'])

print("\nKruskal-Wallis test results:")
print(f"H-statistic: {h_stat:.3f}")
print(f"p-value: {p_val:.3e}")

# %%
