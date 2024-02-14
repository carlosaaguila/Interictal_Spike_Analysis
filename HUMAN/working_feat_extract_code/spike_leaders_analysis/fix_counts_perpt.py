#%%
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
             'HUP072','HUP073','HUP085','HUP094']

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
# nina_pts = nina_pts[nina_pts['hup_id'] == 'HUP214']

#%% function for fill_missing_values

def fill_missing_values(dict, total_length):
    filled_dict = dict.copy()
    for key in range(0, total_length):
        if key not in filled_dict:
            filled_dict[key] = 0
    return filled_dict

# %%
counts_per_time_all = pd.DataFrame()

#loop through each row in filenames_w_ids
# for index, row in filenames_w_ids.iterrows():
for index, row in nina_pts.iterrows():  
    hup_id = row['hup_id']
    filename = row['filename']
    print('hup_id: ', hup_id)
    print('filename: ', filename)

    #load the data
    if os.path.exists(f'{data_directory[0]}/spike_leaders/{filename}_spike_output.csv') == True:
        spike_output_DF = pd.read_csv(f'{data_directory[0]}/spike_leaders/{filename}_spike_output.csv').dropna()
        spike_output_DF.columns = ['peak_index', 'channel_index', 'channel_label', 'spike_sequence', 'peak',
                                'left_point', 'right_point','slow_end','slow_max','rise_amp','decay_amp',
                                'slow_width','slow_amp','rise_slope','decay_slope','average_amp','linelen',
                                'interval number', 'peak_index_samples', 'peak_time_usec']
    else: 
        print('no spike_output.csv file')
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

    counts_per_time = spike_output_DF.groupby(['interval number','channel_label']).count().reset_index()
    counts_per_time = counts_per_time[['interval number','channel_label','peak']]
    counts_per_time = counts_per_time.rename(columns={'peak':'count'})
    counts_per_time['hup_id'] = hup_id
    counts_per_time['filename'] = filename

    #fix count_per_time

    password_bin_filepath = "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin"
    with open(password_bin_filepath, "r") as f:
        session = Session("aguilac", f.read())
    dataset = session.open_dataset(filename)

    all_channel_labels = np.array(dataset.get_channel_labels())
    channel_labels_to_download = all_channel_labels[
        electrode_selection(all_channel_labels)
    ]

    duration_usec = dataset.get_time_series_details(
        channel_labels_to_download[0]
    ).duration
    duration_secs = duration_usec / 1e6

    #create a range spanning from 0 to duration_secs in 600 second intervals   
    intervals = np.arange(0, duration_secs, 600)
    #create a list of tuples where each tuple is a start and stop time for each interval
    intervals = list(zip(intervals[:-1], intervals[1:]))
    # for each tuple range, pick a random 60 second interval
    random_intervals = [np.random.randint(i[0], i[1] - 60) for i in intervals]
    #create a list of tuples where each tuple is a start and stop time +- 30 seconds from the random interval
    random_intervals = [(i - 30, i + 30) for i in random_intervals]
    #make sure the first interval is >0
    random_intervals[0] = (150, 210)

    #find how long the filename structure should be:
    total = len(random_intervals)
    #find total count
    count1 = counts_per_time.groupby('interval number').sum().reset_index()

    intnum1 = np.array(count1['interval number'])
    count1_array = np.array(count1['count'])
    
    dict1 = dict(zip(intnum1, count1_array))
    dict1_fixed = fill_missing_values(dict1, total_length = total)
    dict1_sorted = sorted(dict1_fixed.items())
    x = [x[0] for x in dict1_sorted]
    y = [y[1] for y in dict1_sorted]

    new_counts = pd.DataFrame(columns = ['HUP_id','filename', 'interval_number', 'total_count'])
    new_counts['total_count'] = y
    new_counts['interval_number'] = x
    new_counts['filename'] = filename
    new_counts['HUP_id'] = hup_id
    new_counts['interval_length'] = total
     
    counts_per_time_all = counts_per_time_all.append(new_counts)

# counts_per_time_all.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/counts_per_time_all_v2.csv', index=False)
    
clean_counts = pd.DataFrame()
for id_group in counts_per_time_all.groupby('HUP_id'):
    id_group_df = id_group[1]
    id_group_df['interval_number'] = range(0, len(id_group_df))
    clean_counts = pd.concat([clean_counts, id_group_df])

clean_counts.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/counts_per_time_all_v2.csv', index=False)

# %%
