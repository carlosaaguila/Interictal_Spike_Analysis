#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample

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

# remove the patients in the blacklist from filenames_w_ids
filenames_w_ids = filenames_w_ids[~filenames_w_ids['hup_id'].isin(blacklist)]
#only keep rows where the column "to use" is a 1
filenames_w_ids = filenames_w_ids[filenames_w_ids['to use'] == 1].reset_index(drop=True)

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

#sort by new_spike_seq
spike_output_DF = spike_output_DF.sort_values(by=['new_spike_seq']).dropna()

# %%
#create a new column called "is_spike_leader" that is 1 if the spike is a spike leader and 0 if it is not
spike_output_DF['is_spike_leader'] = 0
#set the first spike in each spike sequence to be a spike leader, based on smallest peak_index in each group
spike_output_DF.loc[spike_output_DF.groupby(['interval number','spike_sequence'])['peak_index'].idxmin(),'is_spike_leader'] = 1

spike_output_DF
# %% FIND THE DISTRIBUTION OF SPIKE SEQUENCE SIZES
#what are the different sizes for each group of new_spike_seq
spike_output_DF.groupby('new_spike_seq').size()
#plot the distribution of spike sequence sizes
spike_output_DF.groupby('new_spike_seq').size().hist(bins=50)

# %%
