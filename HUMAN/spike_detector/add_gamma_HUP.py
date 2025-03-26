#%%
######################
#Grab the HUP DATASET#
######################
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

data_directory = ['/mnt/sauce/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/sauce/littlab/data/Human_Data']
drop_pts = ['HUP093','HUP108','HUP113','HUP114','HUP116','HUP123','HUP087','HUP099','HUP111','HUP121','HUP105','HUP106','HUP107','HUP159'] #These are the patients with less than 8 contacts.
#load in both spike dataframes for HUP
spikes_thresh = pd.read_csv('/mnt/sauce/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
spikes_thresh = spikes_thresh[~spikes_thresh['pt_id'].isin(drop_pts)]

all_spikes = spikes_thresh
#WHAT DO YOU WANT TO REMOVE FROM THE CORE PLOT (CHOICES: 'frontal','mesial temporal','other cortex', 'temporal neocortical','temporal')
soz_to_remove = ['temporal']

#channels to keep 
# chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']
chs_tokeep = ['RA','LA','RDA','LDA','LDH','RDH','LHD', 'RHD','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']


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
# mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)
# non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)

mesial_temp_spikes = mesial_temp_spikes[~mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RCB|Z')].reset_index(drop=True)
non_mesial_temp_spikes = non_mesial_temp_spikes[~non_mesial_temp_spikes['channel_label'].str.contains('T|F|P|RCC|RCA|RCB|Z')].reset_index(drop=True)

all_spikes = pd.concat([mesial_temp_spikes, non_mesial_temp_spikes], axis=0).reset_index(drop=True)

HUP_SPIKES = all_spikes

HUP_SPIKES['max_gamma_power'] = np.nan
HUP_SPIKES['gamma_freq'] = np.nan 
HUP_SPIKES['dur_gamma'] = np.nan

#%%
password_bin_filepath = "/mnt/sauce/littlab/users/aguilac/tools/agu_ieeglogin.bin"
with open(password_bin_filepath, "r") as f:
    session = Session("aguilac", f.read())

#%%
filenames = HUP_SPIKES['filename'].unique()

log_file = os.path.join("/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/logs_gamma", f"FULL_hup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    sys.stdout = open(log_file, 'w')
except IOError as e:
    print(f"Error: Unable to create or write to log file. {e}")
    sys.exit(1)

try:
    df_prog = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/full_hup_spikes_with_gamma.csv', index_col=0)
except: 
    print('making new one or doesnt exist')
    
for i, filename in tqdm(enumerate(filenames)):
    print(f"starting: {filename}")
    try:
        sub_df = HUP_SPIKES[HUP_SPIKES.filename == filename].sample(n=10000, random_state=42)
    except: 
        sub_df = HUP_SPIKES[HUP_SPIKES.filename == filename]

    try:
        if filename in df_prog.filename.unique():
            print(f"Skipping {filename} as it is already in the dataframe")
            continue
    except: 
        print('df prog not saved yet doesnt exist')
    
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
        try:
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
        except:
            # max_gamma_power,gamma_freq,dur_gamma = 9999.5555
            # Update main dataframe with gamma metrics
            sub_df.at[row.name, 'max_gamma_power'] = 9999.5555
            sub_df.at[row.name, 'gamma_freq'] = 9999.5555
            sub_df.at[row.name, 'dur_gamma'] = 9999.5555

    if i == 0:
        sub_df.to_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/full_hup_spikes_with_gamma.csv')
    else: 
        sub_df.to_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/full_hup_spikes_with_gamma.csv', mode = 'a', header = False)
