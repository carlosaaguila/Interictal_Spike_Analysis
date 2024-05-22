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
drop_pts = ['HUP093','HUP108','HUP113','HUP114','HUP116','HUP123','HUP087','HUP099','HUP111','HUP121','HUP105','HUP106','HUP107','HUP159'] #These are the patients with less than 8 contacts.
#load in both spike dataframes for HUP
spikes_full = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col = 0)
spikes_thresh = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
spikes_full = spikes_full[~spikes_full['pt_id'].isin(drop_pts)]
spikes_thresh = spikes_thresh[~spikes_thresh['pt_id'].isin(drop_pts)]
all_spike_list = [spikes_full, spikes_thresh]

# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES
take_spike_leads = False
#WHAT DO YOU WANT TO REMOVE FROM THE CORE PLOT (CHOICES: 'frontal','mesial temporal','other cortex', 'temporal neocortical','temporal')
soz_to_remove = ['temporal']

# list_of_feats = ['spike_rate', 'decay_amp', 'rise_amp','sharpness','linelen','spike_width','slow_width','slow_amp','recruitment_latency_thresh']
# list_of_feats = ['decay_amp','sharpness','linelen','slow_amp']
list_of_feats = ['spike_rate','recruitment_latency_thresh']

df_to_use = []
for Feat_of_interest in list_of_feats:
    if Feat_of_interest == 'recruitment_latency_thresh':
        df_to_use.append(1)
    else:
        df_to_use.append(0)

interp = False

print('Starting')
for i, Feat_of_interest in enumerate(list_of_feats):
    print('looking at:', Feat_of_interest)
    all_spikes = all_spike_list[df_to_use[i]]
    ####################
    # 1. Load in data  #
    ####################

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
    mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)
    non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)

    ########################################
    # 2. Filter Elecs, Group, and Analysis #
    ########################################

    #strip the letters from the channel_label column and keep only the numerical portion
    mesial_temp_spikes['channel_label'] = mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '', regex = True)
    non_mesial_temp_spikes['channel_label'] = non_mesial_temp_spikes['channel_label'].str.replace('L|R|A|H|B|C|D', '', regex = True)

    #replace "sharpness" with the absolute value of it
    mesial_temp_spikes[Feat_of_interest] = abs(mesial_temp_spikes[Feat_of_interest])
    non_mesial_temp_spikes[Feat_of_interest] = abs(non_mesial_temp_spikes[Feat_of_interest])

    #group by patient and channel_label and get the average spike rate for each patient and channel
    mesial_temp_spikes_avg = mesial_temp_spikes.groupby(['pt_id', 'channel_label'])[Feat_of_interest].mean().reset_index()
    #for non_mesial_temp_spikes_avg['SOZ'], only keep everything after '_'
    # non_mesial_temp_spikes['SOZ'] = non_mesial_temp_spikes['SOZ'].str.split('_').str[1]
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes.groupby(['pt_id', 'channel_label', 'SOZ'])[Feat_of_interest].mean().reset_index()

    # for mesial_temp_spikes_avg, add a column called 'mesial' and set it to 1
    mesial_temp_spikes_avg['SOZ'] = 'mesial temporal'

    #concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
    all_spikes_avg = pd.concat([mesial_temp_spikes_avg, non_mesial_temp_spikes_avg], axis=0).reset_index(drop=True)
    all_spikes_avg = all_spikes_avg.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=Feat_of_interest)
    all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

    #create a heat map where each row is a patient from pt_id and each column is a channel from channel_label
    #the values are the average spike rate for each patient and channel
    mesial_temp_spikes_avg = mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.pivot_table(index='pt_id', columns='channel_label', values=Feat_of_interest)

    #reorder columns so goes in [1,2,3,4,5,6,7,8,9,10,11,12]
    mesial_temp_spikes_avg = mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
    non_mesial_temp_spikes_avg = non_mesial_temp_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])


    def remove_rows_by_index(pivot_table, index_values_to_remove):
        """
        Remove rows from a pivot table based on index values.

        Args:
        - pivot_table (DataFrame): The pivot table to filter.
        - index_values_to_remove (str or list): Index value(s) to remove.

        Returns:
        - DataFrame: The filtered pivot table.
        """
        if isinstance(index_values_to_remove, str):
            index_values_to_remove = [index_values_to_remove]

        mask = ~pivot_table.index.get_level_values('SOZ').isin(index_values_to_remove)
        return pivot_table[mask]

    #look to move some SOZ's around:
    all_spikes_avg = remove_rows_by_index(all_spikes_avg, soz_to_remove)

    #probably change the Frontal SOZ to other? 
    if 'frontal' in all_spikes_avg.index.get_level_values(1):
        all_spikes_avg.rename(index={'frontal': 'other cortex'}, inplace=True, level = 'SOZ')

    #reorder all_spikes_avg, so that is_mesial is decesending
    all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])

    if interp == True:
        # Smooth out the data by interpolating along the rows while retaining the original number of rows
        all_spikes_avg_v2 = pd.DataFrame(index=all_spikes_avg.index, columns=np.linspace(0, len(all_spikes_avg.columns) - 1, 100), dtype=float)
        for i, row in all_spikes_avg.iterrows():
            all_spikes_avg_v2.loc[i] = interp1d(np.arange(len(row)), row, kind='linear')(np.linspace(0, len(row)-1, 100))

        all_spikes_avg = all_spikes_avg_v2

    ####################
    # 3. Plot Heatmaps #
    ####################

    sns.set_style('ticks')
    if interp == True:
        plt.figure(figsize=(20,20))
    else:
        plt.figure(figsize=(20,20))

    # sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)
    sns.heatmap(all_spikes_avg, cmap = 'rocket', alpha = 1)
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
    mesial_temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'mesial temporal'])

    plt.axhline(mesial_temp_pts, color='white', linewidth=2.5)
    plt.axhline(mesial_temp_pts+other_cortex_pts, color='white', linewidth=1.5, linestyle = '--')
    plt.axhline(mesial_temp_pts+other_cortex_pts+temp_pts, color='white', linewidth=1.5, linestyle = '--')
    #create a list of 48 colors
    colors = ['#E64B35FF']*mesial_temp_pts + ['#7E6148FF']*other_cortex_pts + ['#00A087FF']*temp_pts + ['#3C5488FF']*temp_neocort_pts
    for ytick, color in zip(plt.gca().get_yticklabels(), colors):
        ytick.set_color(color)

    #add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
    import matplotlib.patches as mpatches
    mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
    other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
    # temporal_patch = mpatches.Patch(color='#00A087FF', label='Temporal Patients')
    neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')

    plt.legend(handles=[mesial_patch, other_patch, neocort_patch], loc='upper right')
    sns.despine()


    if interp == True:
        plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_CONCEPT_MERGE.pdf')
        continue
    else: 
        plt.savefig(f'figures/sameside_perSOZ/bilateral/{Feat_of_interest}_allptsbySOZ_CLEAN.pdf')
    plt.show()

    # continue
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
        # #if the row has less than 8 channels, omit from analysis
        # if len(all_spikes_avg.iloc[row].dropna()) < 8:
        #     continue
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

    ### New METRIC
    coeff_5 = []
    coeff_10 = []
    firstonly = []
    m_label = []
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
        m_label.append(all_spikes_avg.index[row])

        # coeff_5.append((gradient[4]-gradient[0])/len(gradient[0:5]))
        coeff_10.append((gradient[-1]-gradient[0])/len(gradient))
        firstonly.append(gradient[0])

    slope_df = pd.DataFrame(data = coeff_5, columns = ['coef5'])
    slope_df['coef10'] = coeff_10
    slope_df['first_value'] = firstonly
    slope_df['SOZ'] = [x[1] for x in m_label]
    slope_df['pt_id'] = [x[0] for x in m_label]

    #remove the temporal patients for this plot (corr_df and pearson_df)
    # corr_df = corr_df[corr_df['SOZ'] != 'temporal']
    # pearson_df = pearson_df[pearson_df['SOZ'] != 'temporal']

    def soz_assigner(row):
        if row['SOZ'] == 'temporal neocortical':
            return int(2)
        elif row['SOZ'] == 'other cortex':
            return int(2)
        elif row['SOZ'] == 'mesial temporal':
            return int(1)
        else:
            return None

    corr_df['SOZ'] = corr_df.apply(soz_assigner, axis = 1)
    pearson_df['SOZ'] = pearson_df.apply(soz_assigner, axis = 1)
    slope_df['SOZ'] = slope_df.apply(soz_assigner, axis = 1)


    #SPEARMAN CORRELATION PLOTS
    #create a boxplot comparing the distribution of correlation across SOZ types
    plt.figure(figsize=(8,6))
    #where 1, is MTL, 2 is NEO, and 3 is Other
    #change font to arial
    plt.rcParams['font.family'] = 'Arial'

    my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal neocortical':'#00A087FF'} #'temporal':'#3C5488FF'
    pairs=[(1, 'temporal neocortical'), ('temporal neocortical','other cortex'), (1,'other cortex')]
    order = [1,'temporal neocortical','other cortex']

    my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
    pairs=[(1, 2)]
    order = [1,2]

    ax = sns.boxplot(x='SOZ', y='correlation', data=corr_df, palette=my_palette, order = order, showfliers = False)
    sns.stripplot(x="SOZ", y="correlation", data=corr_df, color="black", alpha=0.5)
    annotator = Annotator(ax, pairs, data=corr_df, x="SOZ", y="correlation", order=order)
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', verbose = True)
    annotator.apply_and_annotate()


    plt.xlabel('SOZ Type', fontsize=12)
    plt.ylabel('Spearman Correlation', fontsize=12)
    #change the x-tick labels to be more readable
    # plt.xticks(np.arange(3), ['Mesial Temporal', 'Neocortical', 'Other Cortex'], fontsize = 12)
    plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
    plt.yticks(fontsize = 12)

    #part to change
    plt.title(f'(Feature = {Feat_of_interest}) Directionality', fontsize=16)
    sns.despine()
    plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/spearman/{Feat_of_interest}-ranksum_CLEAN.pdf')
    plt.show()

    #Pearson Correlation PLOTS
    #create a boxplot comparing the distribution of correlation across SOZ types
    plt.figure(figsize=(8,6))
    #change font to arial
    plt.rcParams['font.family'] = 'Arial'
    my_palette = {1:'#E64B35FF', 'other cortex':'#7E6148FF', 'temporal neocortical':'#00A087FF'} #'temporal':'#3C5488FF'
    pairs=[(1, 'temporal neocortical'), ('temporal neocortical','other cortex'), (1,'other cortex')]
    order = [1,'temporal neocortical','other cortex']

    my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
    pairs=[(1, 2)]
    order = [1,2]
    ax = sns.boxplot(x='SOZ', y='correlation', data=pearson_df, palette=my_palette, order=order, showfliers = False)
    sns.stripplot(x="SOZ", y="correlation", data=pearson_df, color="black", alpha=0.5)
    annotator = Annotator(ax, pairs, data=pearson_df, x="SOZ", y="correlation", order=order)
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', verbose = True)
    annotator.apply_and_annotate()

    plt.xlabel('SOZ Type', fontsize=12)
    plt.ylabel('Pearson Correlation', fontsize=12)
    #change the x-tick labels to be more readable
    # plt.xticks(np.arange(3), ['Mesial Temporal', 'Neocortical', 'Other Cortex'], fontsize = 12)
    plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
    plt.yticks(fontsize = 12)

    #part to change
    plt.title(f'Feature = {Feat_of_interest} Directionality', fontsize=16)
    sns.despine()
    plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/pearson/{Feat_of_interest}-ranksum_CLEAN.pdf')
    plt.show()

    #SLOPE COEFFICIENT PLOTS for all contacts
    #create a boxplot comparing the distribution of correlation across SOZ types
    plt.figure(figsize=(10,10))
    my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
    #change font to arial
    plt.rcParams['font.family'] = 'Arial'
    pairs=[(1, 2)]
    order = [1,2]
    ax = sns.boxplot(x='SOZ', y='coef10', data=slope_df, palette=my_palette, order=order, showfliers = False)
    sns.stripplot(x="SOZ", y="coef10", data=slope_df, color="black", alpha=0.5)
    annotator = Annotator(ax, pairs, data=slope_df, x="SOZ", y="coef10", order=order)
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', verbose = True)
    annotator.apply_and_annotate()

    plt.xlabel('SOZ Type', fontsize=12)
    plt.ylabel('Slope Coefficient', fontsize=12)
    #change the x-tick labels to be more readable
    plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(f'Feature = {Feat_of_interest} Directionality', fontsize=16)
    sns.despine()
    plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/new_metrics/coef10_{Feat_of_interest}-ranksum_CLEAN.pdf')

    #SLOPE COEFFICIENT PLOTS for FIRST VALUE
    #create a boxplot comparing the distribut   ion of correlation across SOZ types
    plt.figure(figsize=(8,6))
    my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
    #change font to arial
    plt.rcParams['font.family'] = 'Arial'
    pairs=[(1, 2)]
    order = [1,2]
    ax = sns.boxplot(x='SOZ', y='first_value', data=slope_df, palette=my_palette, order=order, showfliers = False)
    sns.stripplot(x="SOZ", y="first_value", data=slope_df, color="black", alpha=0.5)
    annotator = Annotator(ax, pairs, data=slope_df, x="SOZ", y="first_value", order=order)
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', verbose = True)
    annotator.apply_and_annotate()

    plt.xlabel('SOZ Type', fontsize=12)
    plt.ylabel('Slope Coefficient', fontsize=12)
    #change the x-tick labels to be more readable
    plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(f'Feature = {Feat_of_interest} Directionality', fontsize=16)
    sns.despine()

    plt.savefig(f'figures/sameside_perSOZ/bilateral/statistical_test/new_metrics/first-value_{Feat_of_interest}-ranksum_CLEAN.pdf')

