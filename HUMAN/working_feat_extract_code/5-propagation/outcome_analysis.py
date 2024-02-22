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
#load in the outcome data
redcap = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/Erin_Carlos_RedCAP_data.xlsx')

#create our 2 search queries. We want to look down LOCATION and SURGERY NOTES to get Mesial Temporal targets
outcomes = redcap[~redcap['Location?'].isna()]
outcomes_2 = redcap[~redcap['Surgery NOTES'].isna()]

#grab only mesial temporal structure targetted interventions
outcomes = outcomes[outcomes['Location?'].str.contains('Mesial|mesial|Hippo|hippo|amygd|Amygd')]
outcomes = outcomes[~outcomes['Procedure?'].str.contains('Resection|resection')]
outcomes = outcomes[outcomes['Location?'].str.contains('Mesial Temporal')]

#do the same but across outcomes_2
# outcomes_2 = outcomes_2[outcomes_2['Surgery NOTES'].str.contains('Mesial|mesial|Hippo|hippo|amygd|Amygd')]

#now merge them to see what we have
# mesial_pts = pd.concat([outcomes, outcomes_2])
mesial_pts = outcomes
mesial_pts = mesial_pts.drop_duplicates().reset_index(drop = True)
mesial_pts = mesial_pts[~mesial_pts['Outcomes?'].isna()]
mesial_pts = mesial_pts[~mesial_pts['Outcomes?'].str.contains('NONE|OTHER|None')]

#seperate between good and bad
split_outcomes = mesial_pts['Outcomes?'].str.split()
ilae_indices = split_outcomes.apply(lambda x: x[-1] for x in split_outcomes)
mesial_pts['ilae'] = ilae_indices.iloc[:, 1]

def map_ilae_to_go(ilae_value, which):
    if ilae_value in which:
        return 1
    else:
        return 0
    
which = ['1','1a']
mesial_pts['G/O v1'] = mesial_pts['ilae'].apply(lambda x: map_ilae_to_go(x, which))
which = ['1','2','1a']
mesial_pts['G/O v2'] = mesial_pts['ilae'].apply(lambda x: map_ilae_to_go(x, which))

pts_oi = mesial_pts[['HUP_id','G/O v1','G/O v2']]
# %%  ADD BILATERAL PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

Feat_of_interest = 'linelen' #recruiment_latency
GO_version = 'G/O v2'
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

#only keep the patients with outcome data for their mesial temporal intervention.
all_spikes_avg2 = all_spikes_avg.merge(pts_oi, left_on = 'pt_id', right_on = 'HUP_id', how = 'right')
#drop the NaN's
all_spikes_avg = all_spikes_avg2[~all_spikes_avg2['pt_id'].isna()]

#create a pivot table for plotting
all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ',GO_version], columns='channel_label', values=Feat_of_interest)
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
    if 'HUP215' in all_spikes_avg.index:
        all_spikes_avg = all_spikes_avg.drop('HUP215')
    if 'HUP099' in all_spikes_avg.index:
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
plt.clf()
#color in all the mesial temporal channels
plt.figure(figsize=(20,20))

# if Feat_of_interest == 'spike_rate':
#     sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=15)
# if Feat_of_interest == 'sharpness':
#     sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=200)
# if Feat_of_interest == 'linelen':
#     sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=4300)
# if Feat_of_interest == 'slow_max':
#     sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1, vmin=0, vmax=900)
# else:
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

# Add black dots for 'G/O v1'
for i, pt_id in enumerate(all_spikes_avg.index.get_level_values('pt_id')):
    if all_spikes_avg.index[i][2] == 1:  # Check if 'G/O v1' is True
        plt.text(0, i + 0.5, '●', ha='center', va='center', color='black')

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
temporal_patch = mpatches.Patch(color='#00A087FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')

plt.legend(handles=[mesial_patch, other_patch, temporal_patch, neocort_patch], loc='upper right')

# plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_allptsbySOZ.png', dpi = 300)
plt.show()

#%%
#find the spearman correlation of each row in all_spikes_avg
#initialize a list to store the spearman correlation
channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
channel_labels = [int(x) for x in channel_labels]
spearman_corr = []
label = []
for row in range(len(all_spikes_avg)):
    #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
    spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
    label.append(all_spikes_avg.index[row]) 

corr_df = pd.DataFrame(spearman_corr, columns=['correlation', 'p-value'])
corr_df['SOZ'] = [x[1] for x in label]
corr_df['pt_id'] = [x[0] for x in label]
corr_df['G/O'] = [x[2] for x in label]

# find the pearson correlation of each row in all_spikes_avg
# initialize a list to store the spearman correlation
pearson_corr = []
p_label = []
for row in range(len(all_spikes_avg)):
    # #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
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
pearson_df['G/O'] = [x[2] for x in label]

#%%
if remove_other_pts == True:
    # here we want to only keep the patients who have mTLE so that we analyze within group (lets try with sharpness)
    corr_df = corr_df[corr_df['SOZ'] == 1]
    pearson_df = pearson_df[pearson_df['SOZ'] == 1]

#%%
#Spearman Correlation STATS
#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
from scipy.stats import ranksums
good_outcome = corr_df[corr_df['G/O'] == 1]['correlation']
bad_outcome = corr_df[corr_df['G/O'] == 0]['correlation']

#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
print('Good Outcome vs. Bad Outcome')
print(ranksums(good_outcome, bad_outcome, nan_policy='omit'))


# Get Cohen's D
#correct if the population S.D. is expected to be equal for the two groups.
def cohend(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

cohend_spearman = cohend(good_outcome, bad_outcome)
print('cohen d for spearman', cohend_spearman )

########################
# MANUAL PLOTS (STATS) #
########################

#SPEARMAN CORRELATION PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(10,10))
# my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal':'#3C5488FF', 'temporal neocortical':'#00A087FF'}
my_palette = {0: '#666362', 1: '#202020'}
#change font to arial
plt.rcParams['font.family'] = 'Arial'
sns.boxplot(x='G/O', y='correlation', data=corr_df, palette=my_palette)
plt.xlabel('Outcome Type', fontsize=12)
plt.ylabel('Spearman Correlation', fontsize=12)
#change the x-tick labels to be more readable
if GO_version == 'G/O v1':
    plt.xticks(np.arange(2), ['ILAE 2+', 'ILAE 1/1a'], fontsize = 12)
if GO_version == 'G/O v2':
    plt.xticks(np.arange(2), ['ILAE 3+', 'ILAE 1/1a/2'], fontsize = 12)
plt.yticks(fontsize = 12)

#############################################################################################
#part to change
plt.title(f'Distribution of Spearman Correlation by Outcome (Feature = {Feat_of_interest})', fontsize=16)

# add a significance bar between -
# Mesial and Other C
# plt.plot([0, 0, 1, 1], [1.5, 1.6, 1.6, 1.5], lw=1.5, c='k')
# plt.text((0+1)*.5, 1.65, "***", ha='center', va='bottom', color='k')

# plt.plot([0, 0, 2, 2], [2,2.1,2.1,2], lw=1.5, c='k')
# plt.text((0+2)*.5, 2.15, "***", ha='center', va='bottom', color='k')

#############################################################################################

# plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/contact_control/spearman/{Feat_of_interest}-ranksum.png', dpi = 300, bbox_inches='tight')

#%%
#Spearman Correlation STATS
#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
from scipy.stats import ranksums
good_outcome = pearson_df[pearson_df['G/O'] == 1]['correlation']
bad_outcome = pearson_df[pearson_df['G/O'] == 0]['correlation']

#run a wilcoxon rank sum test to see if the distribution of correlation is different between mesial temporal and other cortex
print('Good Outcome vs. Bad Outcome')
print(ranksums(good_outcome, bad_outcome, nan_policy='omit'))

cohend_pearson = cohend(good_outcome, bad_outcome)
print('cohen d for spearman', cohend_pearson )

########################
# MANUAL PLOTS (STATS) #
########################

#Pearson CORRELATION PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(10,10))
# my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal':'#3C5488FF', 'temporal neocortical':'#00A087FF'}
my_palette = {0: '#666362', 1: '#202020'}
#change font to arial
plt.rcParams['font.family'] = 'Arial'
sns.boxplot(x='G/O', y='correlation', data=pearson_df, palette=my_palette)
plt.xlabel('Outcome Type', fontsize=12)
plt.ylabel('Pearson Correlation', fontsize=12)
#change the x-tick labels to be more readable
if GO_version == 'G/O v1':
    plt.xticks(np.arange(2), ['ILAE 2+', 'ILAE 1/1a'], fontsize = 12)
if GO_version == 'G/O v2':
    plt.xticks(np.arange(2), ['ILAE 3+', 'ILAE 1/1a/2'], fontsize = 12)
plt.yticks(fontsize = 12)

#############################################################################################
#part to change
plt.title(f'Distribution of Pearson Correlation by Outcome (Feature = {Feat_of_interest})', fontsize=16)

# add a significance bar between -
# Mesial and Other C
# plt.plot([0, 0, 1, 1], [1.5, 1.6, 1.6, 1.5], lw=1.5, c='k')
# plt.text((0+1)*.5, 1.65, "***", ha='center', va='bottom', color='k')

# plt.plot([0, 0, 2, 2], [2,2.1,2.1,2], lw=1.5, c='k')
# plt.text((0+2)*.5, 2.15, "***", ha='center', va='bottom', color='k')

#############################################################################################

# plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/contact_control/spearman/{Feat_of_interest}-ranksum.png', dpi = 300, bbox_inches='tight')


#%%
value_columns = ['1','2','3','4','5','6','7','8','9','10','11','12']

# Iterate through the rows of the pivot table
for idx, row in all_spikes_avg.iterrows():
    # Extract the index levels
    pt_id, soz, go_v1 = idx
    # Plot the line with colors based on the value of 'G/O v1'
    row = row.dropna()
    x = np.arange(1,len(row.dropna())+1,1)
    if go_v1 == 1:
        color = 'r' 
        # plt.plot(range(len(value_columns)), row.values, color=color)

        # Calculate linear regression line
        # slope, intercept = np.polyfit(x, row.values, 2)
        # plt.plot(x, slope*x + intercept, linestyle='--', color=color)
        params = np.polyfit(x, row.values, 3)
        polynomial = np.poly1d(params)
        plt.plot(x, polynomial(x), linestyle='--', color=color, label = 'GOOD')
    else:
        color = 'b'
        # plt.plot(range(len(value_columns)), row.values, color=color)

        # Calculate linear regression line
        # slope, intercept = np.polyfit(x, row.values, 2)
        # plt.plot(x, slope*x + intercept, linestyle='--', color=color)
        params = np.polyfit(x, row.values, 3)
        polynomial = np.poly1d(params)
        plt.plot(x, polynomial(x), linestyle='--', color=color, label = "BAD")
plt.legend()