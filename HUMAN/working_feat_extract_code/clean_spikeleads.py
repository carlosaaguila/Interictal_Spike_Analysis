#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re

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

# remove the patients in the blacklist from filenames_w_ids
filenames_w_ids = filenames_w_ids[~filenames_w_ids['hup_id'].isin(blacklist)]
#only keep rows where the column "to use" is a 1
filenames_w_ids = filenames_w_ids[filenames_w_ids['to use'] == 1].reset_index(drop=True)
#%%
"""
#%%
# pick a patient
patient = "HUP143"
#find the row with "hup_id" == patient
patient_row = filenames_w_ids[filenames_w_ids['hup_id'] == patient].reset_index(drop=True)
#find the filename
filename = patient_row['filename'].to_list()[0]
#load the data
spike_output_DF = pd.read_csv(f'{data_directory[0]}/spike_leaders/{filename}_spike_output.csv', header = None).dropna()
spike_output_DF.columns = ['peak_index', 'channel_index', 'channel_label', 'spike_sequence', 'peak',
                           'left_point', 'right_point','slow_end','slow_max','rise_amp','decay_amp',
                           'slow_width','slow_amp','rise_slope','decay_slope','average_amp','linelen',
                           'interval number', 'peak_index_samples', 'peak_time_usec']


#%%
#for each dataframe, add a column with a unique tag for each spike_sequence and interval number
#this will be used to identify each spike sequence and interval when we combine all the dataframes
spike_output_DF['new_spike_seq'] = spike_output_DF.groupby(['interval number','spike_sequence']).ngroup()

#sort by new_spike_seq, drop all rows with NaN values
spike_output_DF = spike_output_DF.sort_values(by=['new_spike_seq']).dropna()

# %%
#create a new column called "is_spike_leader" that is 1 if the spike is a spike leader and 0 if it is not
spike_output_DF['is_spike_leader'] = 0
#set the first spike in each spike sequence to be a spike leader, based on smallest peak_index in each group
spike_output_DF.loc[spike_output_DF.groupby(['new_spike_seq'])['peak_index'].idxmin(),'is_spike_leader'] = 1

spike_output_DF
# %% FIND THE DISTRIBUTION OF SPIKE SEQUENCE SIZES
#what are the different sizes for each group of new_spike_seq
spike_output_DF.groupby('new_spike_seq').size()
#plot the distribution of spike sequence sizes
spike_output_DF.groupby('new_spike_seq').size().hist(bins=50)

#%%
# load the patient data for patient = 'HUP143 using load_ptall
spike, brain_df, onsetzone, ids = load_ptall(patient, data_directory)

#%%
#merge brain_df using "key_0" to spike_output_DF using "channel_label", to get 'final_label' in spike_output_DF
spike_output_DF = spike_output_DF.merge(brain_df[['key_0','final_label']], left_on='channel_label', right_on='key_0', how='left')
#drop the extra column 'key_0'
spike_output_DF = spike_output_DF.drop(columns=['key_0'])

#%%
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

# %%
#only keep the spike leaders
spike_output_DF_leads = spike_output_DF[spike_output_DF['is_spike_leader'] == 1].reset_index(drop=True)

#add patient id
spike_output_DF_leads['pt_id'] = patient

#calculate spike_width
spike_output_DF_leads['spike_width'] = spike_output_DF_leads['right_point']-spike_output_DF_leads['left_point']

#%%
# tell me how many unique 'channel_label' there are
spike_output_DF_leads['channel_label'].nunique()
# what is the distribution of channel labels?
spike_output_DF_leads['channel_label'].value_counts().to_dict()
# what is the distribution of 'final_label'?
spike_output_DF_leads['final_label'].value_counts().to_dict()

# %%
#calculate the sharpness of a spike, by subtracting decay_slope from rise_slope
spike_output_DF_leads['sharpness'] = spike_output_DF_leads['rise_slope']-spike_output_DF_leads['decay_slope']

#calculate rise_duration of a spike, by subtracting left_point from peak
spike_output_DF_leads['rise_duration'] = spike_output_DF_leads['peak']-spike_output_DF_leads['left_point']

#calculate decay_duration of a spike, by subtracting peak from right_point
spike_output_DF_leads['decay_duration'] = spike_output_DF_leads['right_point']-spike_output_DF_leads['peak']
"""
# %%
#create a loop to go through each patient and concatenate all the spike_output_DF_leads
#initialize a dataframe
spike_output_DF_leads_all = pd.DataFrame()
#loop through each row in filenames_w_ids
for index, row in filenames_w_ids.iterrows():
    hup_id = row['hup_id']
    filename = row['filename']
    print('hup_id: ', hup_id)
    print('filename: ', filename)

    #load the data
    spike_output_DF = pd.read_csv(f'{data_directory[0]}/spike_leaders/{filename}_spike_output.csv').dropna()
    spike_output_DF.columns = ['peak_index', 'channel_index', 'channel_label', 'spike_sequence', 'peak',
                            'left_point', 'right_point','slow_end','slow_max','rise_amp','decay_amp',
                            'slow_width','slow_amp','rise_slope','decay_slope','average_amp','linelen',
                            'interval number', 'peak_index_samples', 'peak_time_usec']
    
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

    #clean labels
    spike_output_DF['channel_label'] = spike_output_DF['channel_label'].apply(lambda x: decompose_labels(x, hup_id))
    #merge brain_df using "key_0" to spike_output_DF using "channel_label", to get 'final_label' in spike_output_DF
    spike_output_DF = spike_output_DF.merge(brain_df[['key_0','final_label']], left_on='channel_label', right_on='key_0', how='left')
    #drop the extra column 'key_0'
    spike_output_DF = spike_output_DF.drop(columns=['key_0'])

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

    # initialize spike_output_DF_leads
    spike_output_DF_leads = pd.DataFrame()  
    
    #only keep the spike leaders
    spike_output_DF_leads = spike_output_DF[spike_output_DF['is_spike_leader'] == 1].reset_index(drop=True)

    #add patient id
    spike_output_DF_leads['pt_id'] = hup_id

    #calculate spike_width
    spike_output_DF_leads['spike_width'] = spike_output_DF_leads['right_point']-spike_output_DF_leads['left_point']

    #calculate the sharpness of a spike, by subtracting decay_slope from rise_slope
    spike_output_DF_leads['sharpness'] = spike_output_DF_leads['rise_slope']-spike_output_DF_leads['decay_slope']

    #calculate rise_duration of a spike, by subtracting left_point from peak
    spike_output_DF_leads['rise_duration'] = spike_output_DF_leads['peak']-spike_output_DF_leads['left_point']

    #calculate decay_duration of a spike, by subtracting peak from right_point
    spike_output_DF_leads['decay_duration'] = spike_output_DF_leads['right_point']-spike_output_DF_leads['peak']

    #add spike_output_DF_leads to spike_output_DF_leads_all
    spike_output_DF_leads_all = spike_output_DF_leads_all.append(spike_output_DF_leads)

#save spike_output_DF_leads_all 
spike_output_DF_leads_all.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/spike_output_DF_leads_all', index=False)



# %%
