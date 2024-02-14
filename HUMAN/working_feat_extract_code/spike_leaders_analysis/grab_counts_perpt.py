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

    counts_per_time_all = counts_per_time_all.append(counts_per_time)

counts_per_time_all.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/counts_per_time_all.csv', index=False)

# %%
#which filenames from counts_per_time_all are not in nina_pts filenames list
counts_per_time_all_not_in_nina = nina_pts[~nina_pts['filename'].isin(counts_per_time_all['filename'])]
# %%


counts = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/counts_per_time_all.csv')
#filter by hup_id and separate files
counts = counts[counts['filename'].str.contains('HUP201')]
phase1 = counts[counts['filename'].str.contains('D01')]
phase2 = counts[counts['filename'].str.contains('D02')]

#get counts by interval number
phase1.groupby('interval number').sum()
phase2.groupby('interval number').sum()

count1 = phase1.groupby('interval number').sum().reset_index()
count2 = phase2.groupby('interval number').sum().reset_index()

#turn to lists
intnum1 = np.array(count1['interval number'])
count1_array = np.array(count1['count'])

intnum2 = np.array(count2['interval number'])
count2_array = np.array(count2['count'])

#combine them into pairs (a python thing)
dict1 = zip(intnum1, count1_array)
dict2 = zip(intnum2, count2_array)

#fill in the missing values
dict1_fixed = fill_missing_values(dict1, total_length = total1, default_value = 0)
dict2_fixed = fill_missing_values(dict2, total_length = total2, default_value = 0)

#shift the second file by the total1, which is the number of intervals of 1st file
dict2_shifted = shift_keys(dict2_fixed, total1)
all_dict = concatenate_dicts(dict1_fixed,dict2_shifted)
all_sorted = sorted(all_dict.items())
x = [x[0] for x in all_sorted]
y = [y[1] for y in all_sorted]
plt.figure(figsize = (10,5))
plt.plot(x,y)
plt.xlabel('interval')
plt.ylabel('counts')
plt.title('HUP201')