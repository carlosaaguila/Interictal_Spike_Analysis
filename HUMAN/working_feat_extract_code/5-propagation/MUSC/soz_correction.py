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
from statannotations.Annotator import Annotator

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

nonnan_mask = MUSC_sozs.dropna()
pts_to_remove = nonnan_mask[nonnan_mask['Correction Notes'].str.contains('null')]['ParticipantID'].array

MUSC_spikes = MUSC_spikes[~MUSC_spikes['pt_id'].isin(pts_to_remove)]

# ADD MUSC PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

vs_other = True #CHANGE if you want to compare 2 groups, or 3. [False: you compare mtle, tle, other] [True: you compare mtle vs. other]

list_of_feats = ['spike_rate', 'rise_amp','decay_amp','sharpness','linelen','recruiment_latency','spike_width','slow_width','slow_amp']
for Feat_of_interest in list_of_feats:

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
        if vs_other == False:
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
        if vs_other == True:
            if row['MTL'] == 1:
                return 1
            elif row['Neo'] == 1:
                return 2
            elif row['Temporal'] == 1:
                return 2
            elif row['Other'] == 1:
                return 2
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

    # #remove 'HUP215' from all_spikes_avg
    # if ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    #     all_spikes_avg = all_spikes_avg.drop('HUP215')
    #     all_spikes_avg = all_spikes_avg.drop('HUP099')

    ####################
    # 3. Plot Heatmaps #
    ####################

    #plot mesial_temp_spikes_avg, non_mesial_temp_spikes_avg in a heatmap 
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font = 'Verdana')
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

    if vs_other == False:
        #add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
        import matplotlib.patches as mpatches
        mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
        neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')
        other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')

        plt.legend(handles=[mesial_patch, neocort_patch, other_patch], loc='upper right')

        plt.savefig(f'../figures/MUSC/soz_corrections/soz_analysis/{Feat_of_interest}_allptsbySOZ.pdf')
        plt.show()

    if vs_other == True:
        #add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
        import matplotlib.patches as mpatches
        mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
        neocort_patch = mpatches.Patch(color='#3C5488FF', label='Other Patients') #change this to other (despite misleading variable name)
        plt.legend(handles=[mesial_patch, neocort_patch], loc='upper right')

        plt.savefig(f'../figures/MUSC/soz_corrections/soz_analysis/vs_other/{Feat_of_interest}_allptsbySOZ.pdf')
        plt.show()

    #########################
    # Generate Correlations #
    #########################

    #find the spearman correlation of each row in all_spikes_avg
    #initialize a list to store the spearman correlation
    channel_labels = ['1','2','3','4','5','6','7','8','9','10']
    channel_labels = [int(x) for x in channel_labels]
    spearman_corr = []
    label = []
    for row in range(len(all_spikes_avg)):
        #if the row has less than 8 channels, omit from analysis
        if len(all_spikes_avg.iloc[row].dropna()) < 8:
            continue
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
        #if the row has less than 8 channels, omit from analysis
        if len(all_spikes_avg.iloc[row].dropna()) < 8:
            continue
        gradient = all_spikes_avg.iloc[row].to_list()
        channel_labels = ['1','2','3','4','5','6','7','8','9','10']
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

    ########################
    # MANUAL PLOTS (STATS) #
    ########################

    if vs_other == False:
        #SPEARMAN CORRELATION PLOTS
        #create a boxplot comparing the distribution of correlation across SOZ types
        plt.figure(figsize=(10,10))
        #where 1, is MTL, 2 is NEO, and 3 is Other
        my_palette = {1:'#E64B35FF', 2:'#3C5488FF', 3:'#7E6148FF'}
        
        #change font to arial
        plt.rcParams['font.family'] = 'Verdana'

        pairs=[(1, 2), (2,3), (1,3)]
        order = [1,2,3]
        ax = sns.boxplot(x='SOZ', y='correlation', data=corr_df, palette=my_palette, order = order, showfliers = False)
        sns.stripplot(x="SOZ", y="correlation", data=corr_df, color="black", alpha=0.5)
        annotator = Annotator(ax, pairs, data=corr_df, x="SOZ", y="correlation", order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose = True)
        annotator.apply_and_annotate()

        
        plt.xlabel('SOZ Type', fontsize=12)
        plt.ylabel('Spearman Correlation', fontsize=12)
        #change the x-tick labels to be more readable
        plt.xticks(np.arange(3), ['Mesial Temporal', 'Neocortical', 'Other Cortex'], fontsize = 12)
        plt.yticks(fontsize = 12)

        #part to change
        plt.title(f'Distribution of Spearman Correlation by SOZ Type (Feature = {Feat_of_interest})', fontsize=16)

        plt.savefig(f'../figures/MUSC/soz_corrections/stat_test/spearman/{Feat_of_interest}-ranksum.pdf')

        #Pearson Correlation PLOTS
        #create a boxplot comparing the distribution of correlation across SOZ types
        plt.figure(figsize=(10,10))
        my_palette = {1:'#E64B35FF', 2:'#3C5488FF', 3:'#7E6148FF'}
        #change font to arial
        plt.rcParams['font.family'] = 'Verdana'
        pairs=[(1, 2), (2,3), (1,3)]
        order = [1,2,3]
        ax = sns.boxplot(x='SOZ', y='correlation', data=pearson_df, palette=my_palette, order=order, showfliers = False)
        sns.stripplot(x="SOZ", y="correlation", data=pearson_df, color="black", alpha=0.5)
        annotator = Annotator(ax, pairs, data=pearson_df, x="SOZ", y="correlation", order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose = True)
        annotator.apply_and_annotate()

        plt.xlabel('SOZ Type', fontsize=12)
        plt.ylabel('Pearson Correlation', fontsize=12)
        #change the x-tick labels to be more readable
        plt.xticks(np.arange(3), ['Mesial Temporal', 'Neocortical', 'Other Cortex'], fontsize = 12)
        plt.yticks(fontsize = 12)

        #part to change
        plt.title(f'Distribution of Pearson Correlation by SOZ Type (Feature = {Feat_of_interest})', fontsize=16)

        plt.savefig(f'../figures/MUSC/soz_corrections/stat_test/pearson/{Feat_of_interest}-ranksum.pdf')

    if vs_other == True:
        #SPEARMAN CORRELATION PLOTS
        #create a boxplot comparing the distribution of correlation across SOZ types
        plt.figure(figsize=(10,10))
        #where 1, is MTL, 2 is NEO, and 3 is Other
        my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
        
        #change font to arial
        plt.rcParams['font.family'] = 'Verdana'

        pairs=[(1, 2)]
        order = [1,2]
        ax = sns.boxplot(x='SOZ', y='correlation', data=corr_df, palette=my_palette, order = order, showfliers = False)
        sns.stripplot(x="SOZ", y="correlation", data=corr_df, color="black", alpha=0.5)
        annotator = Annotator(ax, pairs, data=corr_df, x="SOZ", y="correlation", order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose = True)
        annotator.apply_and_annotate()

        
        plt.xlabel('SOZ Type', fontsize=12)
        plt.ylabel('Spearman Correlation', fontsize=12)
        #change the x-tick labels to be more readable
        plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
        plt.yticks(fontsize = 12)

        #part to change
        plt.title(f'Distribution of Spearman Correlation by SOZ Type (Feature = {Feat_of_interest})', fontsize=16)

        plt.savefig(f'../figures/MUSC/soz_corrections/stat_test/vs_other/spearman/{Feat_of_interest}-ranksum.pdf')

        #Pearson Correlation PLOTS
        #create a boxplot comparing the distribution of correlation across SOZ types
        plt.figure(figsize=(10,10))
        my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
        #change font to arial
        plt.rcParams['font.family'] = 'Verdana'
        pairs=[(1, 2)]
        order = [1,2]
        ax = sns.boxplot(x='SOZ', y='correlation', data=pearson_df, palette=my_palette, order=order, showfliers = False)
        sns.stripplot(x="SOZ", y="correlation", data=pearson_df, color="black", alpha=0.5)
        annotator = Annotator(ax, pairs, data=pearson_df, x="SOZ", y="correlation", order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose = True)
        annotator.apply_and_annotate()

        plt.xlabel('SOZ Type', fontsize=12)
        plt.ylabel('Pearson Correlation', fontsize=12)
        #change the x-tick labels to be more readable
        plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
        plt.yticks(fontsize = 12)

        #part to change
        plt.title(f'Distribution of Pearson Correlation by SOZ Type (Feature = {Feat_of_interest})', fontsize=16)

        plt.savefig(f'../figures/MUSC/soz_corrections/stat_test/vs_other/pearson/{Feat_of_interest}-ranksum.pdf')
