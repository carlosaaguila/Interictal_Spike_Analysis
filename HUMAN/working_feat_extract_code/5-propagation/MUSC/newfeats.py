# Code to generate a new spike-rate column, and a new spike timing column called "retention latency"
# required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re

import warnings
warnings.filterwarnings('ignore')

#display 999 lines
pd.set_option('display.max_rows', 999)
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

# Load data
MUSC_spikes = pd.read_csv('../dataset/MUSC_allspikes.csv')
all_spikes = MUSC_spikes
# all_spikes = all_spikes.drop(['engel','hup_id','name','spike_rate'], axis=1)

#Add new spike_rate to all_spikes
spike_count= all_spikes.groupby(['pt_id','channel_label']).count()

#from spike_count, create a dataframe that has pt_id, channel_label, and spike_count
spike_count = spike_count.reset_index()
spike_count = spike_count[['pt_id','channel_label','peak']]

#rename peak to spike_count
spike_count = spike_count.rename(columns={'peak':'spike_count'})

#get the unique count of 'interval number' for each pt_id
interval_count = all_spikes.groupby(['pt_id'])['interval number'].max()
interval_count = interval_count.reset_index()

#merge spike_count and interval_count on pt_id
spike_count = spike_count.merge(interval_count, on='pt_id')

# calulate spike rate by dividing spike_count by interval number
spike_count['spike_rate'] = spike_count['spike_count']/spike_count['interval number']

#merge spike_count with all_spikes on pt_id and channel_label
all_spikes = all_spikes.merge(spike_count[['pt_id','channel_label','spike_rate']], on=['pt_id','channel_label'])

#subtract 'peak' column values from 1000
all_spikes['peak'] = all_spikes['peak'] -1000

#now subtract 'peak' from 'peak_index_samples'
all_spikes['peak_index_samples'] = all_spikes['peak_index_samples'] + all_spikes['peak']

#calculate retention_latency
#for each new_spike_seq, order them by peak_index in ascending order, and then calculate the difference between each peak_index
all_spikes['seq_spike_time_diff'] = all_spikes.groupby(['new_spike_seq','pt_id'])['peak_index_samples'].transform(lambda x: x.sort_values().diff())

#for each new_spike_seq, order them by peak_index in ascending order, and calculate the difference between each peak_index and the first peak_index denoted by is_spike_leader
all_spikes['recruiment_latency'] = all_spikes.groupby(['new_spike_seq','pt_id'])['peak_index_samples'].transform(lambda x: x.sort_values() - x.sort_values()[x.sort_values().index[0]])

#create a new column called "is_spike_leader" that is 1 if the spike is a spike leader and 0 if it is not
all_spikes['is_spike_leader'] = 0
#set the first spike in each spike sequence to be a spike leader, based on smallest peak_index in each group
all_spikes.loc[all_spikes.groupby(['new_spike_seq','pt_id'])['peak_index_samples'].idxmin(),'is_spike_leader'] = 1


#if is_spike_leader == 1, change seq_spike_time_diff to 0
all_spikes.loc[all_spikes['is_spike_leader'] == 1, 'seq_spike_time_diff'] = 0

all_spikes.to_csv('../dataset/MUSC_allspikes_v2.csv')


#TEST - all_spikes[(all_spikes['pt_id'] == "HUP105") & (all_spikes['new_spike_seq'] == 400)].sort_values(by = 'peak_index')[['peak_index','channel_label','new_spike_seq','seq_spike_time_diff','is_spike_leader','pt_id', 'recruiment_latency']]
