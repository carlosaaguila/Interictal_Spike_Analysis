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

print(mesial_temp_spikes['pt_id'].nunique())
print(non_mesial_temp_spikes['pt_id'].nunique())

# %%
#strip the letters from the channel_label column and keep only the numerical portion
mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')
non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')

#replace "sharpness" with the absolute value of it
mesial_temp_spikes['sharpness'] = abs(mesial_temp_spikes['sharpness'])
non_mesial_temp_spikes['sharpness'] = abs(non_mesial_temp_spikes['sharpness'])

#group by patient and channel_label and get the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])['sharpness'].mean().reset_index()
non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label'])['sharpness'].mean().reset_index()

# for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
mesial_temp_spikes_avg['is_mesial'] = 1
# for non_mesial_temp_spikes_avg, add a column called 'mesial' and set it to 0
non_mesial_temp_spikes_avg['is_mesial'] = 0

#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','is_mesial'], columns='channel_label', values='sharpness')
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['is_mesial', 'pt_id'], ascending=[False, True])

# %%
#create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
#the values are the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values='sharpness')
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values='sharpness')

#reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

#%%
#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(mesial_temp_spikes_avg, cmap='viridis')
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(non_mesial_temp_spikes_avg, cmap='viridis')
plt.show()

#%%
#color in all the mesial temporal channels
plt.figure(figsize=(15,15))
sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)
plt.xlabel('Channel Number', fontsize=16)
plt.ylabel('Patient ID', fontsize=16)
plt.title('Average Sharpness by Channel and Patient', fontsize=20)
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

# plt.savefig('figures/allpts_bySOZ_mesial.png', dpi = 300)
plt.show()


# %%
