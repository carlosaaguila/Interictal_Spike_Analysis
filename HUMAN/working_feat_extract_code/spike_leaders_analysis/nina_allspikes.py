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

#load in nina's list of patients to use
nina_pts = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/filenames_w_ids_nina.csv')
#drop na
nina_pts = nina_pts.dropna()
#remove patients in blacklist
nina_pts = nina_pts[~nina_pts['hup_id'].isin(blacklist)]
#make to_use_nina into int
nina_pts['to_use_nina'] = nina_pts['to_use_nina'].astype(int)
#keep rows where to_use_nina is 1
nina_pts = nina_pts[nina_pts['to_use_nina'] == 1]
 # %%
all_spikes = pd.DataFrame()

for index, row in nina_pts.iterrows():
    print(f'Processing {row["filename"]}')
    print(f'{index} out of {len(nina_pts)}')

    filename = row['filename']
    hup_id = row['hup_id']

    # load the data
    try:
        spike_output_DF = pd.read_csv(f'{data_directory[0]}/spike_leaders/{filename}_spike_output.csv').dropna()
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

    #For each group of new_spike_seq, find the difference between the peak_index_samples between the smallest and largest peak_index_samples of the group
    spike_output_DF['seq_total_dur'] = spike_output_DF.groupby(['new_spike_seq'])['peak_index_samples'].transform(lambda x: x.max() - x.min())

    #For each group of new_spike_seq, sort the values by peak_index_sample and find the difference between each peak_index_samples
    spike_output_DF['intra_spike_timediff'] = spike_output_DF.groupby(['new_spike_seq'])['peak_index_samples'].transform(lambda x: x.sort_values().diff())

    #sort by new_spike_seq, drop all rows with NaN values
    spike_output_DF = spike_output_DF.sort_values(by=['new_spike_seq']).dropna()

    #create a new column called "is_spike_leader" that is 1 if the spike is a spike leader and 0 if it is not
    spike_output_DF['is_spike_leader'] = 0
    #set the first spike in each spike sequence to be a spike leader, based on smallest peak_index in each group
    spike_output_DF.loc[spike_output_DF.groupby(['new_spike_seq'])['peak_index'].idxmin(),'is_spike_leader'] = 1

    #clean labels
    spike_output_DF['channel_label'] = spike_output_DF['channel_label'].apply(lambda x: decompose_labels(x, hup_id))

    #For each group of new_spike_seq, find the difference between the peak_index_samples between the smallest and largest peak_index_samples of the group
    spike_output_DF['seq_total_dur'] = spike_output_DF.groupby(['new_spike_seq'])['peak_index_samples'].transform(lambda x: x.max() - x.min())

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

    #add filename
    spike_output_DF['filename'] = filename

    spike_output_DF = spike_output_DF[['filename','pt_id','channel_label','peak_index',
                                       'peak','peak_time_usec', 'peak_index_samples',
                                       'new_spike_seq','seq_total_dur','intraspike_interval',
                                       'interval number', 'is_spike_leader']]
    
    #sequence count (number of spikes)
    spike_output_DF['num_spikes_per_seq'] = spike_output_DF

    # Get the last value of each group
    last_values = spike_output_DF.groupby('new_spike_seq')['peak_index_samples'].last()

    # Get the first value of each group, except the first group
    first_values = spike_output_DF.groupby('new_spike_seq')['peak_index_samples'].first().shift(-1)

    # Calculate the interspike interval as the difference between the first value of each group (except the first group) and the last value of the previous group
    spike_output_DF['interspike_interval'] = first_values - last_values

    interspike_intervals = {}
    for idx, last_value in last_values.iteritems():
        next_group_first_value = first_values.get(idx)
        if next_group_first_value is not None:
            interspike_intervals[idx+1] = next_group_first_value - last_value

    spike_output_DF['interspike_interval'] = spike_output_DF['new_spike_seq'].map(interspike_intervals)

    #concat spike_output_DF to all_spikes
    all_spikes = pd.concat([all_spikes, spike_output_DF], ignore_index=True)

#save the new dataframe as a csv
all_spikes.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nina_allspikes/nina_allspikes.csv', index=False)
# %%
