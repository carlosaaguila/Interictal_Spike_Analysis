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

#%% 
# Analysis to look at the average sharpness of spikes from MTLE vs. TLE patients

####################
# 1. Load in data  #
####################

#load spikes from dataset
all_spikes = pd.read_csv('dataset/spikes_bySOZ.csv')
#remove patients with 'SOZ' containing other
all_spikes = all_spikes[~all_spikes['SOZ'].str.contains('other')].reset_index(drop=True)

#channels to keep 
chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']

#if channel_label contains any of the strings in chs_tokeep, keep it
all_spikes = all_spikes[all_spikes['channel_label'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

#get only the spikes that contain 'mesial temporal' in the SOZ column
mesial_temp_spikes = all_spikes[all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

# grab the remaining spikes that aren't in mesial_temp_spikes
non_mesial_temp_spikes = all_spikes[~all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

#remove any 'channel_label' that contains the letter T or F
mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB')].reset_index(drop=True)
non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB')].reset_index(drop=True)

########################################
# 2. Filter Elecs, Group, and Analysis #
########################################

#strip the letters from the channel_label column and keep only the numerical portion
mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')
non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')

#Initialize the feature of interest, this will generate a heatmap on this feature.
Feat_of_interest = 'avg_latency'

#replace "sharpness" with the absolute value of it
mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

#group by patient and channel_label and get the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()

# for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
mesial_temp_spikes_avg['is_mesial'] = 1
# for non_mesial_temp_spikes_avg, add a column called 'mesial' and set it to 0
non_mesial_temp_spikes_avg['is_mesial'] = 0

#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','is_mesial'], columns='channel_label', values=Feat_of_interest)
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['is_mesial', 'pt_id'], ascending=[False, True])


#create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
#the values are the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)

#reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

####################
# 3. Plot Heatmaps #
####################

#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# sns.heatmap(mesial_temp_spikes_avg, cmap='viridis')
# plt.show()
# plt.figure(figsize=(10,10))
# sns.heatmap(non_mesial_temp_spikes_avg, cmap='viridis')
# plt.show()

#color in all the mesial temporal channels
plt.figure(figsize=(15,15))
sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)
plt.xlabel('Channel Number', fontsize=16)
plt.ylabel('Patient ID', fontsize=16)
plt.title(f'Average {Feat_of_interest} by Channel and Patient', fontsize=20)
plt.axhline(25, color='r', linewidth=2)
#change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0))
#create a list of 48 colors
colors = ['r']*25 + ['k']*23
for ytick, color in zip(plt.gca().get_yticklabels(), colors):
    ytick.set_color(color)

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Mesial Temporal')
black_patch = mpatches.Patch(color='black', label='Non-Mesial Temporal')
plt.legend(handles=[red_patch, black_patch], loc='upper right')

plt.savefig(f'figures/{Feat_of_interest}_allptsbySOZ.png.png', dpi = 300)
plt.show()


#%%
# REDO the analysis but this time add color to the plot for each SOZ type

#Initialize the feature of interest, this will generate a heatmap on this feature.
#Feat_of_interest = ''
take_spike_leads = False

####################
# 1. Load in data  #
####################

if Feat_of_interest == 'spike_rate':
    all_spikes = pd.read_csv('dataset/spikes_bySOZ_v2.csv')
else:
    #load spikes from dataset
    all_spikes = pd.read_csv('dataset/spikes_bySOZ.csv')

#flag that says we want spike leaders only
if take_spike_leads == True:
    all_spikes = all_spikes[all_spikes['is_spike_leader'] == 1]

#remove patients with 'SOZ' containing other
# all_spikes = all_spikes[~all_spikes['SOZ'].str.contains('other')].reset_index(drop=True)

#channels to keep 
chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']

#if channel_label contains any of the strings in chs_tokeep, keep it
all_spikes = all_spikes[all_spikes['channel_label'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

#get only the spikes that contain 'mesial temporal' in the SOZ column
mesial_temp_spikes = all_spikes[all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

# grab the remaining spikes that aren't in mesial_temp_spikes
non_mesial_temp_spikes = all_spikes[~all_spikes['SOZ'].str.contains('mesial')].reset_index(drop=True)

#remove any 'channel_label' that contains the letter T or F
mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB')].reset_index(drop=True)
non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB')].reset_index(drop=True)

########################################
# 2. Filter Elecs, Group, and Analysis #
########################################

#strip the letters from the channel_label column and keep only the numerical portion
mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')
non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')

#replace "sharpness" with the absolute value of it
mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

#group by patient and channel_label and get the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
#for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
non_mesial_temp_spikes['SOZ'] = non_mesial_temp_spikes['SOZ'].str.split('_').str[1]
non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'SOZ'])[Feat_of_interest].mean().reset_index()

# for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
mesial_temp_spikes_avg['SOZ'] = 1

#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=Feat_of_interest)
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])


#create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
#the values are the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)

#reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

####################
# 3. Plot Heatmaps #
####################

#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# sns.heatmap(mesial_temp_spikes_avg, cmap='viridis')
# plt.show()
# plt.figure(figsize=(10,10))
# sns.heatmap(non_mesial_temp_spikes_avg, cmap='viridis')
# plt.show()

#color in all the mesial temporal channels
plt.figure(figsize=(20,20))
sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)
plt.xlabel('Channel Number', fontsize=20)
plt.ylabel('Patient ID', fontsize=20)
plt.title(f'Average {Feat_of_interest} by Channel and Patient', fontsize=24)
plt.axhline(25, color='k', linewidth=2.5)
plt.axhline(25+22, color='k', linewidth=1.5, linestyle = '--')
plt.axhline(25+22+13, color='k', linewidth=1.5, linestyle = '--')
#change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0), fontsize=13)
#create a list of 48 colors
colors = ['#E64B35FF']*25 + ['#7E6148FF']*22 + ['#00A087FF']*13 + ['#3C5488FF']*10 
for ytick, color in zip(plt.gca().get_yticklabels(), colors):
    ytick.set_color(color)

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
temporal_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#00A087FF', label='Temporal Neocortical Patients')
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
plt.legend(handles=[mesial_patch, temporal_patch, neocort_patch, other_patch], loc='upper right')

if take_spike_leads == True:
    plt.savefig(f'figures/perSOZ_leads/{Feat_of_interest}_allptsbySOZ.png.png', dpi = 300)
else: 
    plt.savefig(f'figures/perSOZ/add_OC/{Feat_of_interest}_allptsbySOZ.png.png', dpi = 300)

plt.show()

# %%
#only using the same side electrodes as the SOZ laterality

Feat_of_interest = 'decay_amp'
take_spike_leads = False

####################
# 1. Load in data  #
####################

#load spikes from dataset
if ('rate' in Feat_of_interest) | ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    all_spikes = pd.read_csv('dataset/spikes_bySOZ_T-R.csv', index_col=0)
else:
    all_spikes = pd.read_csv('dataset/spikes_bySOZ.csv')

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

#concat them back into all_spikes
all_spikes = pd.concat([left_spikes_tokeep, right_spikes_tokeep], axis =0).reset_index(drop=True)

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
mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

#group by patient and channel_label and get the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
#for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
non_mesial_temp_spikes['SOZ'] = non_mesial_temp_spikes['SOZ'].str.split('_').str[1]
non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'SOZ'])[Feat_of_interest].mean().reset_index()

# for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
mesial_temp_spikes_avg['SOZ'] = 1

#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=Feat_of_interest)
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])


#create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
#the values are the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)

#reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#remove 'HUP215' from all_spikes_avg
if ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    all_spikes_avg = all_spikes_avg.drop('HUP215')
    all_spikes_avg = all_spikes_avg.drop('HUP099')

####################
# 3. Plot Heatmaps #
####################

#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# sns.heatmap(mesial_temp_spikes_avg, cmap='viridis')
# plt.show()
# plt.figure(figsize=(10,10))
# sns.heatmap(non_mesial_temp_spikes_avg, cmap='viridis')
# plt.show()

#color in all the mesial temporal channels
plt.figure(figsize=(20,20))
if Feat_of_interest == 'spike_rate':
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=10)
else:
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)

plt.xlabel('Channel Number', fontsize=20)
plt.ylabel('Patient ID', fontsize=20)
plt.title(f'Average {Feat_of_interest} by Channel and Patient', fontsize=24)
#change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0), fontsize=13)

#in all_spikes_avg, get the number of 'temporal neocortical' patients
temp_neocort_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal neocortical'])
#in all_spikes_avg, get the number of 'temporal' patients
temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal'])
#same for other cortex
other_cortex_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'other cortex'])
#same for mesial temporal
mesial_temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 1])

plt.axhline(mesial_temp_pts, color='k', linewidth=2.5)
plt.axhline(mesial_temp_pts+other_cortex_pts, color='k', linewidth=1.5, linestyle = '--')
plt.axhline(mesial_temp_pts+other_cortex_pts+temp_pts, color='k', linewidth=1.5, linestyle = '--')
#create a list of 48 colors
colors = ['#E64B35FF']*mesial_temp_pts + ['#7E6148FF']*other_cortex_pts + ['#00A087FF']*temp_pts + ['#3C5488FF']*temp_neocort_pts
for ytick, color in zip(plt.gca().get_yticklabels(), colors):
    ytick.set_color(color)

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
temporal_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#00A087FF', label='Temporal Neocortical Patients')
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
plt.legend(handles=[mesial_patch, temporal_patch, neocort_patch, other_patch], loc='upper right')

# plt.savefig(f'figures/sameside_perSOZ/{Feat_of_interest}_allptsbySOZ.png.png', dpi = 300)
plt.show()

# all_spikes_avg.to_csv(f'/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/gradient_data/{Feat_of_interest}.csv')

# %%
# ADD BILATERAL PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

Feat_of_interest = 'slow_width'
take_spike_leads = False

####################
# 1. Load in data  #
####################

#load spikes from dataset
if ('rate' in Feat_of_interest) | ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    all_spikes = pd.read_csv('dataset/spikes_bySOZ_T-R.csv', index_col=0)
else:
    all_spikes = pd.read_csv('dataset/spikes_bySOZ.csv')

bilateral_spikes = pd.read_csv('dataset/bilateral_MTLE_all_spikes.csv')
#remove 'engel','hup_id','name','spike_rate' columns
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
mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

#group by patient and channel_label and get the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
#for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
non_mesial_temp_spikes['SOZ'] = non_mesial_temp_spikes['SOZ'].str.split('_').str[1]
non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'SOZ'])[Feat_of_interest].mean().reset_index()

# for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
mesial_temp_spikes_avg['SOZ'] = 1

#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=Feat_of_interest)
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])


#create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
#the values are the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)

#reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#remove 'HUP215' from all_spikes_avg
if ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    all_spikes_avg = all_spikes_avg.drop('HUP215')
    all_spikes_avg = all_spikes_avg.drop('HUP099')
#%%
####################
# 3. Plot Heatmaps #
####################

#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# sns.heatmap(mesial_temp_spikes_avg, cmap='viridis')
# plt.show()
# plt.figure(figsize=(10,10))
# sns.heatmap(non_mesial_temp_spikes_avg, cmap='viridis')
# plt.show()

#color in all the mesial temporal channels
plt.figure(figsize=(20,20))
if Feat_of_interest == 'spike_rate':
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=10)
if Feat_of_interest == 'sharpness':
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=200)
if Feat_of_interest == 'linelen':
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=4300)
if Feat_of_interest == 'slow_max':
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=900)
else:
    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)

plt.xlabel('Channel Number', fontsize=20)
plt.ylabel('Patient ID', fontsize=20)
plt.title(f'Average {Feat_of_interest} by Channel and Patient', fontsize=24)
#change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0), fontsize=13)

#in all_spikes_avg, get the number of 'temporal neocortical' patients
temp_neocort_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal neocortical'])
#in all_spikes_avg, get the number of 'temporal' patients
temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal'])
#same for other cortex
other_cortex_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'other cortex'])
#same for mesial temporal
mesial_temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 1])

plt.axhline(mesial_temp_pts, color='k', linewidth=2.5)
plt.axhline(mesial_temp_pts+other_cortex_pts, color='k', linewidth=1.5, linestyle = '--')
plt.axhline(mesial_temp_pts+other_cortex_pts+temp_pts, color='k', linewidth=1.5, linestyle = '--')
#create a list of 48 colors
colors = ['#E64B35FF']*mesial_temp_pts + ['#7E6148FF']*other_cortex_pts + ['#00A087FF']*temp_pts + ['#3C5488FF']*temp_neocort_pts
for ytick, color in zip(plt.gca().get_yticklabels(), colors):
    ytick.set_color(color)

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
temporal_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#00A087FF', label='Temporal Neocortical Patients')
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
plt.legend(handles=[mesial_patch, temporal_patch, neocort_patch, other_patch], loc='upper right')

plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_allptsbySOZ.png.png', dpi = 300)
plt.show()

# %%

########################
# MANUAL PLOTS (STATS) #
########################
"""
#Correlation plot of all_spikes_avg (spearman)
plt.figure(figsize = (10,10))
sns.heatmap(all_spikes_avg.corr(method='spearman'), annot=True, cmap='Blues')
plt.show()

#now do the same thing but with pearson correlation
plt.figure(figsize = (10,10))
sns.heatmap(all_spikes_avg.corr(method='pearson'), annot=True, cmap='Blues')
plt.show()
"""

# %%
#########################
# Generate Correlations #
#########################

#find the spearman correlation of each row in all_spikes_avg
#initialize a list to store the spearman correlation
channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
channel_labels = [int(x) for x in channel_labels]
spearman_corr = []
label = []
for row in range(len(all_spikes_avg)):
    spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
    label.append(all_spikes_avg.index[row])

corr_df = pd.DataFrame(spearman_corr, columns=['correlation', 'p-value'])
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

pearson_df = pd.DataFrame(pearson_corr, columns=['correlation', 'p-value'])
pearson_df['SOZ'] = [x[1] for x in label]
pearson_df['pt_id'] = [x[0] for x in label]

#%%
"""
row = 80-17
gradient = all_spikes_avg.iloc[row].to_list()
#grab index from gradient
ptname = all_spikes_avg.index[row][0]
channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
channel_labels = [int(x) for x in channel_labels]
list_to_remove = []
for i in range(len(channel_labels)):
    if np.isnan(gradient[i]):
        list_to_remove.append(i)

#remove list_to_remove from channel_labels and gradient
channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]

#subplot with 2 plots
fig, ax = plt.subplots(1,2, figsize=(10,5))
#plot an scatter plot of channel_label vs. gradient, then fit an R squared line
sns.regplot(x=channel_labels, y=gradient, ci=95, line_kws={'color':'r'}, robust = True, ax=ax[0])
ax[0].set_title('robust = True - Spearman-like')
sns.regplot(x=channel_labels, y=gradient, ci=95, line_kws={'color':'r'}, robust = False, ax=ax[1])
ax[1].set_title('robust = False - Pearson-like')
plt.show()

#display the spearman correlation and p-value
print(ptname)
print('Spearman ---- {}'.format(stats.spearmanr(channel_labels,gradient)))
print('Pearson ----- {}'.format(stats.pearsonr(channel_labels,gradient)))
"""

# %%
#Spearman Correlation STATS
#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
from scipy.stats import ranksums
mesial_temp = corr_df[corr_df['SOZ'] == 1]['correlation']
other_cortex = corr_df[corr_df['SOZ'] == 'other cortex']['correlation']
temporal = corr_df[corr_df['SOZ'] == 'temporal']['correlation']
neocortical = corr_df[corr_df['SOZ'] == 'temporal neocortical']['correlation']

#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
print('Mesial Temporal vs. Other Cortex')
print(ranksums(mesial_temp, other_cortex, nan_policy='omit'))
print('Mesial Temporal vs. Temporal')
print(ranksums(mesial_temp, temporal, nan_policy='omit'))
print('Mesial Temporal vs. Temporal Neocortical')
print(ranksums(mesial_temp, neocortical, nan_policy='omit'))
print('Other Cortex vs. Temporal')
print(ranksums(other_cortex, temporal, nan_policy='omit'))
print('Other Cortex vs. Temporal Neocortical')
print(ranksums(other_cortex, neocortical, nan_policy='omit'))
print('Temporal vs. Temporal Neocortical')
print(ranksums(temporal, neocortical, nan_policy='omit'))

# %%
#SPEARMAN CORRELATION PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(10,10))
my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal':'#3C5488FF', 'temporal neocortical':'#00A087FF'}
#change font to arial
plt.rcParams['font.family'] = 'Arial'
sns.boxplot(x='SOZ', y='correlation', data=corr_df, palette=my_palette)
plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Spearman Correlation', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(4), ['Mesial Temporal', 'Other Cortex', 'Temporal', 'Neocortical'], fontsize = 12)
plt.yticks(fontsize = 12)

#############################################################################################
#part to change
plt.title('Distribution of Spearman Correlation by SOZ Type (Feature = Slow Wave Width)', fontsize=16)

#add a significance bar between -
# # Mesial and Other C
# plt.plot([0, 0, 1, 1], [1.5, 1.6, 1.6, 1.5], lw=1.5, c='k')
# plt.text((0+1)*.5, 1.65, "***", ha='center', va='bottom', color='k')

# # add a signficance bar between -
# # mesial temporal and temporal 
# plt.plot([0, 0, 2, 2], [1.75,1.85,1.85,1.75], lw=1.5, c='k')
# plt.text((0+2)*.5, 1.9, "***", ha='center', va='bottom', color='k')

# # add a signficance bar between -
# # mesial temporal and temporal neocorical
# plt.plot([0, 0, 3, 3], [2,2.1,2.1,2], lw=1.5, c='k')
# plt.text((0+3)*.5, 2.15, "***", ha='center', va='bottom', color='k')

plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/spearman/{Feat_of_interest}-ranksum.png', dpi = 300, bbox_inches='tight')

#############################################################################################

plt.show()

#%%
#Pearson Correlation STATS
#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
from scipy.stats import ranksums
mesial_temp = pearson_df[pearson_df['SOZ'] == 1]['correlation']
other_cortex = pearson_df[pearson_df['SOZ'] == 'other cortex']['correlation']
temporal = pearson_df[pearson_df['SOZ'] == 'temporal']['correlation']
neocortical = pearson_df[pearson_df['SOZ'] == 'temporal neocortical']['correlation']

print(Feat_of_interest)
#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
print('Mesial Temporal vs. Other Cortex')
print(ranksums(mesial_temp, other_cortex, nan_policy='omit'))
print('Mesial Temporal vs. Temporal')
print(ranksums(mesial_temp, temporal, nan_policy='omit'))

print('Mesial Temporal vs. Temporal Neocortical')
print(ranksums(mesial_temp, neocortical, nan_policy='omit'))
print('Other Cortex vs. Temporal')
print(ranksums(other_cortex, temporal, nan_policy='omit'))
print('Other Cortex vs. Temporal Neocortical')
print(ranksums(other_cortex, neocortical, nan_policy='omit'))
print('Temporal vs. Temporal Neocortical')
print(ranksums(temporal, neocortical, nan_policy='omit'))


# %%
#Pearson Correlation PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(10,10))
my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal':'#3C5488FF', 'temporal neocortical':'#00A087FF'}
#change font to arial
plt.rcParams['font.family'] = 'Arial'
sns.boxplot(x='SOZ', y='correlation', data=pearson_df, palette=my_palette)
plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Pearson Correlation', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(4), ['Mesial Temporal', 'Other Cortex', 'Temporal', 'Neocortical'], fontsize = 12)
plt.yticks(fontsize = 12)

#############################################################################################
#part to change
plt.title('Distribution of Pearson Correlation by SOZ Type (Feature = Slow Wave Width)', fontsize=16)

# add a significance bar between -
# # Mesial and Other C
# plt.plot([0, 0, 1, 1], [1.5, 1.6, 1.6, 1.5], lw=1.5, c='k')
# plt.text((0+1)*.5, 1.65, "***", ha='center', va='bottom', color='k')

# # add a signficance bar between -
# # mesial temporal and temporal 
# plt.plot([0, 0, 2, 2], [1.75,1.85,1.85,1.75], lw=1.5, c='k')
# plt.text((0+2)*.5, 1.9, "***", ha='center', va='bottom', color='k')

# # add a signficance bar between -
# # mesial temporal and temporal neocorical
# plt.plot([0, 0, 3, 3], [2,2.1,2.1,2], lw=1.5, c='k')
# plt.text((0+3)*.5, 2.15, "***", ha='center', va='bottom', color='k')

plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/pearson/{Feat_of_interest}-ranksum.png', dpi = 300, bbox_inches='tight')

#############################################################################################

plt.show()
# %%
