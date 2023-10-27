#%%
#set up environment
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *
#pd.set_option('display.max_rows', None)

# %% BOXPLOT CODE
#load in the cleaned spike leaders
clean_spikes = pd.read_csv('../working features/clean_spikeleads/intra_soz_leaders.csv')

#make soz an integer
clean_spikes['soz'] = clean_spikes['soz'].astype(int)

#drop columns: 'pt_id','peak_index','channel_index','spike_sequence','peak','left_point','right_point','slow_end',
#'slow_max','interval number','peak_index_samples','peak_time_usec','new_spike_seq','is_spike_leader', 'final_label', 'channel_label'
clean_spikes = clean_spikes.drop(columns=['peak_index','channel_index','channel_label','spike_sequence','peak','left_point',
                                    'right_point','slow_end','slow_max','interval number','peak_index_samples',
                                    'peak_time_usec','new_spike_seq','is_spike_leader','final_label', 'average_amp'])

#split clean_spikes into SOZ and nonSOZ
soz_feats = clean_spikes[clean_spikes['soz'] == 1]
nonsoz_feats = clean_spikes[clean_spikes['soz'] == 0]

#remove soz column from soz_feats and nonsoz_feats
soz_feats = soz_feats.drop(columns = ['soz'])
nonsoz_feats = nonsoz_feats.drop(columns = ['soz'])

#add SOZ to the beginning of each column name in soz_feats except for the 'pt_id' column
soz_feats.columns = ['SOZ ' + str(col) if col != 'pt_id' else col for col in soz_feats.columns]
#add nonSOZ to the beginning of each column name in nonsoz_feats except for the 'pt_id' column
nonsoz_feats.columns = ['nonSOZ ' + str(col) if col != 'pt_id' else col for col in nonsoz_feats.columns]

#groupby pt_id and take the median of each column
soz_feats = soz_feats.groupby('pt_id').median()
nonsoz_feats = nonsoz_feats.groupby('pt_id').median()

#concatenate soz_feats and nonsoz_feats horizontally
median_feats = pd.concat([soz_feats, nonsoz_feats], axis = 1)

# what are the names of the patients with nans?
pts_to_look = median_feats[median_feats.isna().any(axis=1)].index.tolist()

# drop nans (some patients have no nonSOZ spikes? pretty crazy)
median_feats = median_feats.dropna()

#CREATE HUGE BOXPLOT WITH ALL FEATURES

feats_OI_pre = median_feats.columns.tolist()
#for each feature in feats_OI_pre that starts with SOZ, find its matching non SOZ feature
feats_OI = []
for feat in feats_OI_pre:
    if feat.startswith('SOZ'):
        #remove SOZ from the beginning of the feature name
        feat_name = feat[4:]
        #add nonSOZ to the beginning of the feature name
        nonSOZfeat = 'nonSOZ ' + feat_name
        #add the new feature name to feats_OI
        feats_OI.append([feat, nonSOZfeat])
#flatten feats_OI
feats_OI = [item for sublist in feats_OI for item in sublist]

#make sure there are no negatives in log scale boxplot
median_feats['SOZ rise_slope'] = median_feats['SOZ rise_slope'].abs()
median_feats['nonSOZ rise_slope'] = median_feats['nonSOZ rise_slope'].abs()
median_feats['SOZ decay_slope'] = median_feats['SOZ decay_slope'].abs()
median_feats['nonSOZ decay_slope'] = median_feats['nonSOZ decay_slope'].abs()
median_feats['SOZ sharpness'] = median_feats['SOZ sharpness'].abs()
median_feats['nonSOZ sharpness'] = median_feats['nonSOZ sharpness'].abs()

# median_feats['SOZ rise slope'] = median_feats['SOZ rise slope'].abs()
# median_feats['nonSOZ rise slope'] = median_feats['nonSOZ rise slope'].abs()
# median_feats['SOZ decay slope'] = median_feats['SOZ decay slope'].abs()
# median_feats['nonSOZ decay slope'] = median_feats['nonSOZ decay slope'].abs()
#create a boxplot of all the features in median_feats, but seperate the SOZ and nonSOZ by color
boxprops = dict(linestyle='-', linewidth=1.5, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='k')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
axes = median_feats.boxplot(column = feats_OI, figsize = (15,10), patch_artist = True, showfliers=False, notch = True, grid = False, boxprops = boxprops, medianprops = medianprops, whiskerprops=dict(linestyle='-', linewidth=1.5, color = 'k'))

plt.yscale('log')

import matplotlib

colors = ['r','b','r','b','r','b','r','b','r','b','r','b','r','b', 'r','b','r','b','r','b','r','b','r','b', 'r','b','r','b']
for i, color in enumerate(colors):
    axes.findobj(matplotlib.patches.Patch)[i].set_facecolor(color)

plt.xticks(ticks = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5], 
            labels = ['Rise Amplitude', 'Falling Amplitude','Slow Width','Slow Amplitude',
                     'Rise Slope','Falling Slope','Line Length', 'Sequence Dur.', 'Seqence 1-2',
                     'Average Latency', 'Spike Width', 'Sharpness', 'Rise Duration','Falling Duration'],
                      rotation = 45, ha = 'right')
#add xticks to the top of the plot as well to mirror the x axis
plt.tick_params(top = True)

#add ticks around SOZ rise_amp and nonSOZ rise_amp, SOZ decay_amp and nonSOZ decay_amp, SOZ linelen and nonSOZlinelen
#add significance stars
#rising apitude
plt.plot([1, 1, 2, 2], [(10**4+10**3)/2, 10**4, 10**4, (10**4+10**3)/2], lw=1.5, c='k')
plt.text((1+2)*.5, 10**4, "***", ha='center', va='bottom', color='k')
#falling amplitude
plt.plot([3, 3, 4, 4], [(10**4+10**3)/2, 10**4, 10**4, (10**4+10**3)/2], lw=1.5, c='k')
plt.text((4+3)*.5, 10**4, "***", ha='center', va='bottom', color='k')
#line length
plt.plot([13, 13, 14, 14], [(10**4+10**3)/2, 10**4, 10**4, (10**4+10**3)/2], lw=1.5, c='k')
plt.text((13+14)*.5, 10**4, "***", ha='center', va='bottom', color='k')

plt.title("Univariate Analysis of SOZ vs. non-SOZ Features Within Brain Regions")

SOZ = mpatches.Patch(color='r', label='SOZ')
nonSOZ = mpatches.Patch(color='b', label='non-SOZ')
plt.legend(handles=[SOZ,nonSOZ])
plt.ylabel('log(arbitrary units)')
plt.show()

#%% PAIRED PLOTS CODE

#load in the cleaned spike leaders
clean_spikes = pd.read_csv('../working features/clean_spikeleads/intra_soz_leaders.csv')

#drop columns: 'pt_id','peak_index','channel_index','spike_sequence','peak','left_point','right_point','slow_end',
#'slow_max','interval number','peak_index_samples','peak_time_usec','new_spike_seq','is_spike_leader', 'final_label', 'channel_label'
clean_spikes = clean_spikes.drop(columns=['peak_index','channel_index','channel_label','spike_sequence','peak','left_point',
                                    'right_point','slow_end','slow_max','interval number','peak_index_samples',
                                    'peak_time_usec','new_spike_seq','is_spike_leader','final_label', 'average_amp'])

#make soz an integer
clean_spikes['soz'] = clean_spikes['soz'].astype(int)

#split clean_spikes into SOZ and nonSOZ
soz_feats = clean_spikes[clean_spikes['soz'] == 1]
nonsoz_feats = clean_spikes[clean_spikes['soz'] == 0]

#remove soz column from soz_feats and nonsoz_feats
soz_feats = soz_feats.drop(columns = ['soz'])
nonsoz_feats = nonsoz_feats.drop(columns = ['soz'])

#add SOZ to the beginning of each column name in soz_feats except for the 'pt_id' column
soz_feats.columns = ['SOZ ' + str(col) if col != 'pt_id' else col for col in soz_feats.columns]
#add nonSOZ to the beginning of each column name in nonsoz_feats except for the 'pt_id' column
nonsoz_feats.columns = ['nonSOZ ' + str(col) if col != 'pt_id' else col for col in nonsoz_feats.columns]

#groupby pt_id and take the median of each column
soz_feats = soz_feats.groupby('pt_id').median()
nonsoz_feats = nonsoz_feats.groupby('pt_id').median()

#concatenate soz_feats and nonsoz_feats horizontally
median_feats = pd.concat([soz_feats, nonsoz_feats], axis = 1)

# what are the names of the patients with nans?
pts_to_look = median_feats[median_feats.isna().any(axis=1)].index.tolist()

# drop nans (some patients have no nonSOZ spikes? pretty crazy)
median_feats = median_feats.dropna()

feats_OI_pre = median_feats.columns.tolist()
#for each feature in feats_OI_pre that starts with SOZ, find its matching non SOZ feature
feats_OI = []
for feat in feats_OI_pre:
    if feat.startswith('SOZ'):
        #remove SOZ from the beginning of the feature name
        feat_name = feat[4:]
        #add nonSOZ to the beginning of the feature name
        nonSOZfeat = 'nonSOZ ' + feat_name
        #add the new feature name to feats_OI
        feats_OI.append([feat, nonSOZfeat])

#create a color column for each feature in feats_OI
newcolumns = ['color_riseamp', 'color_decayamp', 'color_slowwidth', 'color_slowamp', 'color_riseslope', 
              'color_decayslope', 'color_LL', 'color_seq_total_dur', 'color_seq_1_2', 'color_avg_latency','color_spikewidth', 'color_sharpness',
              'color_riseduration', 'color_decayduration']

for i, feat in enumerate(feats_OI):
    median_feats[newcolumns[i]] = median_feats[feat[0]] - median_feats[feat[1]]
    median_feats[newcolumns[i]] = median_feats[newcolumns[i]].apply(lambda x: True if x > 0 else False).astype(int)

#create paired plots
title = ['Rise Amplitude', 'Falling Amplitude','Slow Width','Slow Amplitude',
        'Rise Slope','Falling Slope','Line Length', 'Sequence Dur.', 'Seqence 1-2',
        'Average Latency', 'Spike Width', 'Sharpness', 'Rise Duration','Falling Duration']

for i, feat in enumerate(feats_OI):
    fig, ax = plt.subplots(1,1, figsize = (10,10))
    ax.scatter(median_feats[median_feats[newcolumns[i]] == 1][feat[0]], median_feats[median_feats[newcolumns[i]] == 1][feat[1]], color = 'r', label = "Patients w/ SOZ > ({})".format(len(median_feats[median_feats[newcolumns[i]] == 1])))
    ax.scatter(median_feats[median_feats[newcolumns[i]] == 0][feat[0]], median_feats[median_feats[newcolumns[i]] == 0][feat[1]], color = 'b', label = "Patients w/ non-SOZ > ({})".format(len(median_feats[median_feats[newcolumns[i]] == 0])))
    ax.set_xlabel(feat[0])
    ax.set_ylabel(feat[1])
    ax.set_title('SOZ vs. non-SOZ {}'.format(title[i]))
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    plt.show()

# %%
import scipy.stats as stats

# wilcoxon test
for feats in feats_OI:
    SOZfeat = feats[0]
    nonSOZfeat = feats[1]
    wilcoxon = stats.wilcoxon(median_feats[SOZfeat].to_numpy(), median_feats[nonSOZfeat].to_numpy(), nan_policy = 'omit')
    print(SOZfeat, nonSOZfeat)
    print(wilcoxon)


# %%
