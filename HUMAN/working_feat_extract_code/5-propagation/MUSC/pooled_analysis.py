#%%
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

# ADD MUSC PATIENTS
# KEEP THE SAME SIDE, PLUS FOR BILATERAL TAKE BOTH SIDES

list_of_feats = ['sharpness', 'spike_rate']#,'decay_amp','rise_amp','linelen','recruiment_latency','spike_width','slow_width','slow_amp']
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
    all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

    #reorder all_spikes_avg, so that is_mesial is decesending
    all_spikes_avg = all_spikes_avg.sort_values(by=['region', 'pt_id'], ascending=[True, True])

#%%
MUSC_spikes_avg = all_spikes_avg

#%%
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
Feat_of_interest = 'spike_rate'
take_spike_leads = False

####################
# 1. Load in data  #
####################

#load spikes from dataset
if ('rate' in Feat_of_interest) | ('latency' in Feat_of_interest) | (Feat_of_interest == 'seq_spike_time_diff'):
    all_spikes = pd.read_csv('../dataset/spikes_bySOZ_T-R.csv', index_col=0)
    bilateral_spikes = pd.read_csv('../dataset/bilateral_spikes_bySOZ_T-R.csv', index_col=0)
else:
    all_spikes = pd.read_csv('../dataset/spikes_bySOZ.csv')
    bilateral_spikes = pd.read_csv('../dataset/bilateral_MTLE_all_spikes.csv')
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
HUP_spikes_avg = all_spikes_avg

#%%
pearson_corr_1 = []
p_label = []
for row in range(len(MUSC_spikes_avg)):
    #if the row has less than 8 channels, omit from analysis
    if len(MUSC_spikes_avg.iloc[row].dropna()) < 8:
        continue
    gradient = MUSC_spikes_avg.iloc[row].to_list()
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

    pearson_corr_1.append(stats.pearsonr(channel_labels,gradient))
    p_label.append(MUSC_spikes_avg.index[row])

pearson_df_1 = pd.DataFrame(pearson_corr_1, columns=['correlation', 'p-value'])
pearson_df_1['SOZ'] = [x[1] for x in p_label]
pearson_df_1['pt_id'] = [x[0] for x in p_label]

#%%
pearson_corr_2 = []
p_label = []
for row in range(len(HUP_spikes_avg)):
    #if the row has less than 8 channels, omit from analysis
    if len(HUP_spikes_avg.iloc[row].dropna()) < 8:
        continue
    gradient = HUP_spikes_avg.iloc[row].to_list()
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

    pearson_corr_2.append(stats.pearsonr(channel_labels,gradient))
    p_label.append(HUP_spikes_avg.index[row])

pearson_df_2 = pd.DataFrame(pearson_corr_2, columns=['correlation', 'p-value'])
pearson_df_2['SOZ'] = [x[1] for x in p_label]
pearson_df_2['pt_id'] = [x[0] for x in p_label]

#%%
pearson_df_concat = pd.concat([pearson_df_1, pearson_df_2])
pearson_df_concat['SOZ'].unique()

pearson_df_concat_copy = pearson_df_concat.copy() 

def soz_assigner(row):
    if row['SOZ'] == 1:
        return 1
    elif row['SOZ'] == 2:
        return 2
    elif row['SOZ'] == 3:
        return 2
    elif row['SOZ'] == 'temporal neocortical':
        return 2
    elif row['SOZ'] == 'other cortex':
        return 2
    elif row['SOZ'] == 'temporal':
        return None
    else:
        return None

pearson_df_concat_copy['SOZ'] = pearson_df_concat_copy.apply(soz_assigner, axis = 1)

pearson_df_concat_copy = pearson_df_concat_copy.reset_index(drop = True)

pearson_df_concat_copy = pearson_df_concat_copy.dropna()

import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


#Pearson Correlation PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(10,10))
my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
#change font to arial
plt.rcParams['font.family'] = 'Arial'
pairs=[(1, 2)]
order = [1,2]
ax = sns.boxplot(x='SOZ', y='correlation', data=pearson_df_concat_copy, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="SOZ", y="correlation", data=pearson_df_concat_copy, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=pearson_df_concat_copy, x="SOZ", y="correlation", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose = True)
annotator.apply_and_annotate()

plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Pearson Correlation', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Mesial Temporal', 'Other'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title('Distribution of Pearson Correlation by SOZ Type (Feature = Spike Rate)', fontsize=16)

plt.savefig('../figures/MUSC+HUP/stat_test/pearson/spikerate-ranksum.pdf')