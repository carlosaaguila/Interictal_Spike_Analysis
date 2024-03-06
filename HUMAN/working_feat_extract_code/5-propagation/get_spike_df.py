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

#load all the filenames (long form IEEG filenames)
filenames_w_ids = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
#load the list of patients to exclude
blacklist = ['HUP101' ,'HUP112','HUP115','HUP119','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176',
             'HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071',
             'HUP072','HUP073','HUP085','HUP094', 'HUP173', 'HUP211','HUP224'] #check why HUP173/HUP211/HUP224 didn't make it through the pipeline

"""
# remove the patients in the blacklist from filenames_w_ids
filenames_w_ids = filenames_w_ids[~filenames_w_ids['hup_id'].isin(blacklist)]
#only keep rows where the column "to use" is a 1
filenames_w_ids = filenames_w_ids[filenames_w_ids['to use'] == 1].reset_index(drop=True)


#find the patients we want (seperate by SOZ)
# load all the SOZ's
SOZ_list = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/soz_locations.csv', index_col = 0)

# remove rows that have 'diffuse' or 'bilateral' in the 'lateralization' column
# SOZ_list = SOZ_list[~SOZ_list['lateralization'].isin(['diffuse', 'bilateral'])].reset_index(drop=True)
#only keep rows where lateralization is bilateral
SOZ_list = SOZ_list[SOZ_list['lateralization'] == 'bilateral'].reset_index(drop=True)

# remove any nan's in lateralization
SOZ_list = SOZ_list[~SOZ_list['lateralization'].isna()].reset_index(drop=True) 

# remove any 'frontal','multifocal','diffuse','temporal multifocal' in 'region' column
SOZ_list = SOZ_list[~SOZ_list['region'].isin(['frontal','multifocal','diffuse','temporal multifocal'])].reset_index(drop=True)
# remove any nans in 'region' column
SOZ_list = SOZ_list[~SOZ_list['region'].isna()].reset_index(drop=True)

#only get mesial_temporal
# SOZ_list = SOZ_list[SOZ_list['region'] == 'mesial temporal'].reset_index(drop=True)

#patients in SOZ_list that are in filenames_w_ids
pt_in_soz = filenames_w_ids[filenames_w_ids['hup_id'].isin(SOZ_list['name'])].reset_index(drop=True)
"""

#filenames for MUSC patients
MUSC_pts = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC_Emory_LEN_SOZ_type.xlsx')
MUSC_pts_cleaned = MUSC_pts[MUSC_pts['Site_1MUSC_2Emory'] == 1]
# MUSC_pts_cleaned2 = MUSC_pts_cleaned[((MUSC_pts_cleaned['MTL'] == 1) & (MUSC_pts_cleaned['Neo'] == 0))| ((MUSC_pts_cleaned['MTL'] == 0) & (MUSC_pts_cleaned['Neo'] == 1))]
pt_in_soz = MUSC_pts_cleaned
def soz_assigner(row):
    if row['MTL'] == 1 and row['Neo'] == 0:
        return 1
    elif row['MTL'] == 0 and row['Neo'] == 1:
        return 2
    elif row['MTL'] == 1 and row['Neo'] == 1:
        return 3
    else:
        return None
    
pt_in_soz['SOZ'] = pt_in_soz.apply(soz_assigner, axis = 1)

 # %%
all_spikes = pd.DataFrame()

for index, row in pt_in_soz.iterrows():
    print(f'Processing {row["filename"]}')
    print(f'{index} out of {len(pt_in_soz)}')

    filename = row['filename']
    hup_id = row['ParticipantID']

    # load the data
    try:
        spike_output_DF = pd.read_csv(f'{data_directory[0]}/spike_leaders/MUSC/{filename}_spike_output.csv').dropna()
        spike_output_DF.columns = ['peak_index', 'channel_index', 'channel_label', 'spike_sequence', 'peak',
                                'left_point', 'right_point','slow_end','slow_max','rise_amp','decay_amp',
                                'slow_width','slow_amp','rise_slope','decay_slope','average_amp','linelen',
                                'interval number', 'peak_index_samples', 'peak_time_usec']
    except:
        print(f"{filename} does not exist")
        continue

    # check if the first row column 'peak_index' is equal to 'peak_index', if so drop the row
    if spike_output_DF['peak_index'].iloc[0] == 'peak_index':
        spike_output_DF = spike_output_DF.drop(spike_output_DF.index[0]).reset_index(drop=True)

    #set new types for each column
    spike_output_DF = spike_output_DF.astype({'peak_index': 'int', 'channel_index': 'int', 'channel_label': 'str', 'spike_sequence': 'int',
                                              'peak': 'int', 'left_point': 'int', 'right_point': 'int', 'slow_end': 'int',
                                                'slow_max': 'int', 'rise_amp': 'float64', 'decay_amp': 'float64', 'slow_width': 'float64',
                                                'slow_amp': 'float64', 'rise_slope': 'float64', 'decay_slope': 'float64', 'average_amp': 'float64',
                                                'linelen': 'float64', 'interval number': 'int', 'peak_index_samples': 'int64', 'peak_time_usec': 'float64'})

    #for each dataframe, add a column with a unique tag for each spike_sequence and interval number
    #this will be used to identify each spike sequence and interval when we combine all the dataframes
    spike_output_DF['new_spike_seq'] = spike_output_DF.groupby(['interval number','spike_sequence']).ngroup()

    #sort by new_spike_seq, drop all rows with NaN values
    spike_output_DF = spike_output_DF.sort_values(by=['new_spike_seq']).dropna()

    #create a new column called "is_spike_leader" that is 1 if the spike is a spike leader and 0 if it is not
    spike_output_DF['is_spike_leader'] = 0
    #set the first spike in each spike sequence to be a spike leader, based on smallest peak_index in each group
    spike_output_DF.loc[spike_output_DF.groupby(['new_spike_seq'])['peak_index'].idxmin(),'is_spike_leader'] = 1

    """
    # load the patient data
    if os.path.exists(data_directory[0] + '/pickle_spike/{}_obj.pkl'.format(hup_id)):
        spike, brain_df, onsetzone, ids = load_ptall(hup_id, data_directory)
    else:
        print(f'no pickle file for {hup_id}')
        print('skipping patient')
        continue


    # check if brain_df is a dataframe, if not you can skip this patient
    if isinstance(brain_df, pd.DataFrame) == False:
        print('brain_df is not a dataframe')
        continue
    """

    #clean labels
    spike_output_DF['channel_label'] = spike_output_DF['channel_label'].apply(lambda x: decompose_labels(x, hup_id))
    
    """
    #clean brain_df labels
    brain_df['key_0'] = brain_df['key_0'].apply(lambda x: decompose_labels(x, hup_id))

    #merge brain_df using "key_0" to spike_output_DF using "channel_label", to get 'final_label' in spike_output_DF
    spike_output_DF = spike_output_DF.merge(brain_df[['key_0','final_label']], left_on='channel_label', right_on='key_0', how='left')
    #drop the extra column 'key_0'
    spike_output_DF = spike_output_DF.drop(columns=['key_0'])
    """

    #For each group of new_spike_seq, find the difference between the peak_index_samples between the smallest and largest peak_index_samples of the group
    spike_output_DF['seq_total_dur'] = spike_output_DF.groupby(['new_spike_seq'])['peak_index_samples'].transform(lambda x: x.max() - x.min())

    #For each group of new_spike_seq, find the difference between the peak_index_samples between the smallest and second_smallest peak_index_samples of the group
    spike_output_DF['seq_onetwo_time'] = spike_output_DF.groupby(['new_spike_seq'])['peak_index_samples'].transform(lambda x: x.nsmallest(2).max() - x.nsmallest(2).min())

    #For each group of new_spike_seq, sort the values by peak_index_sample and find the difference between each peak_index_samples
    spike_output_DF['seq_spike_time_diff'] = spike_output_DF.groupby(['new_spike_seq'])['peak_index_samples'].transform(lambda x: x.sort_values().diff())

    #for each group of new_spike_seq, find the mean of the seq_spike_time_diff
    spike_output_DF['avg_latency'] = spike_output_DF.groupby(['new_spike_seq'])['seq_spike_time_diff'].transform(lambda x: x.mean())

    # drop seq_spike_time_diff
    spike_output_DF = spike_output_DF.drop(columns=['seq_spike_time_diff'])  

    #calculate spike_width
    spike_output_DF['spike_width'] = spike_output_DF['right_point']-spike_output_DF['left_point']

    #calculate the sharpness of a spike, by subtracting decay_slope from rise_slope
    spike_output_DF['sharpness'] = spike_output_DF['rise_slope']-spike_output_DF['decay_slope']

    #calculate rise_duration of a spike, by subtracting left_point from peak
    spike_output_DF['rise_duration'] = spike_output_DF['peak']-spike_output_DF['left_point']

    #calculate decay_duration of a spike, by subtracting peak from right_point
    spike_output_DF['decay_duration'] = spike_output_DF['right_point']-spike_output_DF['peak']

    #add patient id
    spike_output_DF['pt_id'] = hup_id

    #add SOZ
    spike_output_DF['region'] = row['SOZ']

    #add laterality
    spike_output_DF['lateralization_left'] = row['Left']
    spike_output_DF['lateralization_right'] = row['Right']

    """
    #add spikerate
    master_elecs = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/master_elecs.csv')
    sozlist = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/all_ptids.csv', index_col=0)
    #merge hup_id on rid from  onto master_elecs
    master_elecs = master_elecs.merge(sozlist, left_on='rid', right_on = 'r_id', how='left')
    master_elecs['name'] = master_elecs['name'].apply(lambda x: decompose_labels(x, hup_id))

    #merge master_elecs on channel_label from spike_output_DF
    spike_output_DF = spike_output_DF.merge(master_elecs[['engel','hup_id','name','soz', 'spike_rate']], left_on=['channel_label', 'pt_id'], right_on = ['name','hup_id'], how='left')

    #add patient regionality
    spike_output_DF['region'] = SOZ_list[SOZ_list['name'] == hup_id]['region'].values[0]

    #add patient lateralization
    spike_output_DF['lateralization'] = SOZ_list[SOZ_list['name'] == hup_id]['lateralization'].values[0]

    #add patient SOZ
    spike_output_DF['clinic_SOZ'] = spike_output_DF['lateralization'] + '_' + spike_output_DF['region']

    #drop region and lateralization
    spike_output_DF = spike_output_DF.drop(columns=['region', 'lateralization'])
    """
    #concat spike_output_DF to all_spikes
    all_spikes = pd.concat([all_spikes, spike_output_DF], ignore_index=True)

#save the new dataframe as a csv
all_spikes.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/MUSC_allspikes.csv', index=False)
# %%
