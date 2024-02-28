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

# %%  ADD BILATERAL PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

Feat_of_interest = 'rise_amp' #recruiment_latency
take_spike_leads = False
remove_other_pts = True

####################
# 1. Load in data  #
####################

#load spikes from dataset
if ('rate' in Feat_of_interest) | ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    all_spikes = pd.read_csv('dataset/spikes_bySOZ_T-R.csv', index_col=0)
    bilateral_spikes = pd.read_csv('dataset/bilateral_spikes_bySOZ_T-R.csv', index_col=0)
else:
    all_spikes = pd.read_csv('dataset/spikes_bySOZ.csv')
    bilateral_spikes = pd.read_csv('dataset/bilateral_MTLE_all_spikes.csv')
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

#look for only the channels that will move A -> P
mesial_temp_spikes = mesial_temp_spikes[mesial_temp_spikes['channel_label'].str.contains('A|B|C')]
non_mesial_temp_spikes = non_mesial_temp_spikes[non_mesial_temp_spikes['channel_label'].str.contains('A|B|C')]

#Take out the channels with the letter D in it, these are only 1 electrode
ch_to_remove = ['DA1','DA2','DA3','DA4','DA5','DHA1','DHA2','DHA3','DHA4','DHA5']
mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].isin(ch_to_remove)]
non_mesial_temp_spikes = non_mesial_temp_spikes[~(non_mesial_temp_spikes['channel_label'].isin(ch_to_remove))]


########################################
# 2. Filter Elecs, Group, and Analysis #
########################################

# #strip the letters from the channel_label column and keep only the numerical portion
# mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')
# non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '')

mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.extract(r'([ABC]\d*)')
non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.extract(r'([ABC]\d*)')

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
all_spikes_avg[['char', 'num']] = all_spikes_avg['channel_label'].str.extract(r'([ABCD])(\d+)')

# Convert 'num' column to numeric type
all_spikes_avg['num'] = pd.to_numeric(all_spikes_avg['num'], errors='coerce')

# Drop rows with NaN values in 'num' column
all_spikes_avg = all_spikes_avg.dropna(subset=['num'])

def map_to_nearest_multiple_of_3(num):
    return int(3 * ((num - 1) // 3 + 1))

# Apply the mapping function to the 'num' column
all_spikes_avg['num'] = all_spikes_avg['num'].apply(map_to_nearest_multiple_of_3)
all_spikes_avg['relabel'] = all_spikes_avg['char'] + all_spikes_avg['num'].astype(str)


#%%
all_spikes_avg_test = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='relabel', values=Feat_of_interest)
# all_spikes_avg_test = all_spikes_avg_test[['A3','B3','C3','A6','B6','C6','A9','B9','C9','A12','B12','C12']]
all_spikes_avg_test = all_spikes_avg_test[['A3','A6','A9','A12','B3','B6','B9','B12','C3','C6','C9','C12']]

#%%

# #create a pivot table for plotting
# all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=Feat_of_interest)
# all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

# #reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg_test = all_spikes_avg_test.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])


# #create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
# #the values are the average spike rate for each patient and channel
# mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)
# non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)

# #reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
# mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
# non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

# #remove 'HUP215' from all_spikes_avg
# if ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
#     if 'HUP215' in all_spikes_avg.index:
#         all_spikes_avg = all_spikes_avg.drop('HUP215')
#     if 'HUP099' in all_spikes_avg.index:
#         all_spikes_avg = all_spikes_avg.drop('HUP099')

#%%
####################
# 3. Plot Heatmaps #
####################

all_spikes_avg = all_spikes_avg_test
#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
#color in all the mesial temporal channels
plt.figure(figsize=(20,20))

sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)#, vmin=0, vmax=125)

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

#seperate the patients with a line to show where it splits
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
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
temporal_patch = mpatches.Patch(color='#00A087FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')

plt.legend(handles=[mesial_patch, other_patch, temporal_patch, neocort_patch], loc='upper right')

# plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_allptsbySOZ.png', dpi = 300)
plt.show()

# %%
###################
###################
###################
# MORAN's I
###################
###################
###################
###################

#SOME MORAN'S I work
from libpysal.weights import lat2W
from esda.moran import Moran
import numpy as np

moran_matrices = all_spikes_avg.dropna()

who = []
moranI = []

for i, rows in moran_matrices.iterrows():
    # Use your matrix here, instead of this random one
    Z  = np.array(rows).reshape(3, 4)
    # Create the matrix of weigthts 
    w = lat2W(Z.shape[0], Z.shape[1])
    # Create the pysal Moran object 
    mi = Moran(Z, w)
    # Verify Moran's I results 
    who.append(i)
    moranI.append(mi.I)

#lets get boxplots for Moran's I
    
moran_df = pd.DataFrame(who, columns = ['pt_id','SOZ'])
moran_df['I'] = moranI

plt.figure(figsize=(10,10))
my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal':'#3C5488FF', 'temporal neocortical':'#00A087FF'}
#change font to arial
plt.rcParams['font.family'] = 'Arial'
sns.violinplot(x='SOZ', y='I', data=moran_df, palette=my_palette, inner = 'point')
# sns.stripplot(x = 'SOZ', y = 'I', data = moran_df, color = 'k', jitter = 0.2, size =5)
plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel("Moran's I", fontsize=12)
plt.title("Moran's I for a subset of patients with full electrode data", fontsize = 16)
#change the x-tick labels to be more readable
plt.xticks(np.arange(4), ['Mesial Temporal', 'Other Cortex', 'Temporal', 'Neocortical'], fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#HEATMAP for Moran's I

all_spikes_avg = moran_matrices
#plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
#color in all the mesial temporal channels
plt.figure(figsize=(20,20))

sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)#, vmin=0, vmax=125)

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

#seperate the patients with a line to show where it splits
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
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
temporal_patch = mpatches.Patch(color='#00A087FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')

plt.legend(handles=[mesial_patch, other_patch, temporal_patch, neocort_patch], loc='upper right')

# plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_allptsbySOZ.png', dpi = 300)
plt.show()


# %%
