#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re

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
Feat_of_interest = 'sharpness'
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
