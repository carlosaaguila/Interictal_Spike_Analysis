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

## load the spike data
MUSC_spikes = pd.read_csv('../dataset/MUSC_allspikes_v2.csv', index_col=0)

#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]

#fix SOZ and laterality
MUSC_spikes = MUSC_spikes.merge(MUSC_sozs, left_on = 'pt_id', right_on = 'ParticipantID', how = 'inner')
MUSC_spikes = MUSC_spikes.drop(columns=['ParticipantID','Site_1MUSC_2Emory','IfNeocortical_Location','Correction Notes','lateralization_left','lateralization_right','region'])

# ADD MUSC PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

Feat_of_interest = 'spike_rate'
take_spike_leads = False

#########################
# 1. Organize the data  #
#########################

all_spikes = MUSC_spikes

#flag that says we want spike leaders only
if take_spike_leads == True:
    all_spikes = all_spikes[all_spikes['is_spike_leader'] == 1]

#remove patients with 'SOZ' containing other
# all_spikes = all_spikes[~all_spikes['SOZ'].str.contains('other')].reset_index(drop=True)

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
        return 4
    elif row['Other'] == 1:
        return 3
    else:
        return None

all_spikes['region'] = all_spikes.apply(soz_assigner, axis = 1)

#get only the spikes that contain 'mesial temporal' in the SOZ column
mesial_temp_spikes = all_spikes[all_spikes['region'] == 1].reset_index(drop=True)

# grab the remaining spikes that aren't in mesial_temp_spikes
non_mesial_temp_spikes = all_spikes[~(all_spikes['region'] == 1)].reset_index(drop=True)

########################################
# 2. Filter Elecs, Group, and Analysis #
########################################

#strip the letters from the channel_label column and keep only the numerical portion
mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|P', '')
non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|P', '')

#replace "sharpness" with the absolute value of it
mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

#group by patient and channel_label and get the average spike rate for each patient and channel
mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
mesial_temp_spikes_avg['region'] = 1

#for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'region'])[Feat_of_interest].mean().reset_index()


#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','region'], columns='channel_label', values=Feat_of_interest)
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10'])

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['region', 'pt_id'], ascending=[True, True])

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

plt.clf()
#color in all the mesial temporal channels
plt.figure(figsize=(20,20))

sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)

plt.xlabel('Channel Number', fontsize=20)
plt.ylabel('Patient ID', fontsize=20)
plt.title(f'Average {Feat_of_interest} by Channel and Patient', fontsize=24)
#change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0), fontsize=13)

#in all_spikes_avg, get the number of 'temporal neocortical' patients
temp_neocort_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 2])
#in all_spikes_avg, get the number of 'temporal' patients
temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 4])
#same for other cortex
other_cortex_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 3])
#same for mesial temporal
mesial_temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 1])

plt.axhline(mesial_temp_pts, color='k', linewidth=2.5)
plt.axhline(mesial_temp_pts+temp_neocort_pts, color='k', linewidth=1.5, linestyle = '--')
#create a list of 31 colors
colors = ['#E64B35FF']*mesial_temp_pts + ['#3C5488FF']*temp_neocort_pts + ['#7E6148FF']*other_cortex_pts
for ytick, color in zip(plt.gca().get_yticklabels(), colors):
    ytick.set_color(color)

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')

plt.legend(handles=[mesial_patch, neocort_patch, other_patch], loc='upper right')

# plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_allptsbySOZ.png', dpi = 300)
plt.show()