#%%
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
from sklearn.linear_model import LinearRegression

# Import custom functions
import sys, os
code_v2_path = os.path.dirname('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/')
sys.path.append(code_v2_path)
from get_iEEG_data2 import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *
import pywt
from tqdm import tqdm
from datetime import datetime

code_path = os.path.dirname('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#######################
#Grab the MUSC DATASET#
#######################

## load the spike data
MUSC_spikes = pd.read_csv('../working_feat_extract_code/5-propagation/dataset/complete_dfs/MUSC_thresholded.csv', index_col=0)

#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]
MUSC_sozs = MUSC_sozs.drop(columns=['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'])

#fix SOZ and laterality
MUSC_spikes = MUSC_spikes.merge(MUSC_sozs, left_on = 'pt_id', right_on = 'ParticipantID', how = 'inner')
MUSC_spikes = MUSC_spikes.drop(columns=['ParticipantID','Site_1MUSC_2Emory','IfNeocortical_Location','Correction Notes','lateralization_left','lateralization_right','region'])

#find the patients that should be null, and remove them for the full dataset
nonnan_mask = MUSC_sozs.dropna()
pts_to_remove = nonnan_mask[nonnan_mask['Correction Notes'].str.contains('null')]['ParticipantID'].array
MUSC_spikes = MUSC_spikes[~MUSC_spikes['pt_id'].isin(pts_to_remove)]
MUSC_full = MUSC_spikes

all_spikes = MUSC_full
#########################
# 1. Organize the data  #
#########################

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

MUSC_SPIKES = all_spikes

#%%
password_bin_filepath = "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin"
with open(password_bin_filepath, "r") as f:
    session = Session("aguilac", f.read())


#%%
filenames = MUSC_SPIKES['filename'].unique()

log_file = os.path.join("/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/logs_gamma", f"musc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    sys.stdout = open(log_file, 'w')
except IOError as e:
    print(f"Error: Unable to create or write to log file. {e}")
    sys.exit(1)

for i, filename in tqdm(enumerate(filenames)):
    try:
        sub_df = MUSC_SPIKES[MUSC_SPIKES.filename == filename].sample(n=1000, random_state=42)
    except: 
        sub_df = MUSC_SPIKES[MUSC_SPIKES.filename == filename]
    
    #grab dataset name
    dataset_name = filename
    load_attempt = 1

    while True:
        try: 
            dataset = session.open_dataset(dataset_name)
            break
        except Exception as e:
            if any(str(code) in str(e) for code in ['503', '504', '502', '500']):
                load_attempt += 1
                print(f'Failed to retrieve ieeg.org data, trying loading dataset again (attempt {load_attempt})')

            else:
                error_message = f"Non ieeg.org error opening dataset: {str(e)}"
                print(f"Error processing {filename}: {error_message}")
                print(f"didn't work for {filename}")
                continue    

    for index, row in sub_df.iterrows():

        #load data
        all_channel_labels = np.array(dataset.get_channel_labels())
        channel_labels_to_download = all_channel_labels[
            electrode_selection(all_channel_labels)
        ]

        peak_time_usec = row.peak_time_usec
        ieeg_data, fs = get_iEEG_data2(
            "aguilac",
            password_bin_filepath,
            dataset_name,
            peak_time_usec - 2e6,
            peak_time_usec + 2e6,
            channel_labels_to_download,
        )

        hup_id = row.pt_id
        fs=int(fs)
        #redo the labels (clean)
        channel_labels_to_download = [decompose_labels(x, hup_id) for x in channel_labels_to_download]
        ieeg_data.columns = channel_labels_to_download

        ch_label = row.channel_label
        signal = ieeg_data[ch_label]

        if fs>500:
            signal = resample(np.array(signal), fs, 500)
        fs = 500

        signal = notch_filter(signal,60, fs)
        signal = bandpass_filter(signal, 30, 100, fs, order = 3)

        # get points
        left_point = row['left_point']
        right_point = row['right_point']
        slow_end = row['slow_end']
        peak = row['peak']
        middle_point = len(signal)/2 + peak 

        max_gamma_power, gamma_freq, dur_gamma = compute_gamma(signal, fs, left_point, right_point, slow_end)
        # Update main dataframe with gamma metrics
        sub_df.at[row.name, 'max_gamma_power'] = max_gamma_power
        sub_df.at[row.name, 'gamma_freq'] = gamma_freq
        sub_df.at[row.name, 'dur_gamma'] = dur_gamma

    if i == 0:
        sub_df.to_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/musc_spikes_with_gamma.csv')
    else: 
        sub_df.to_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/musc_spikes_with_gamma.csv', mode = 'a', header = False)