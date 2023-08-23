#%% Load environment
import pickle
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings
import random
import re
from ieeg.auth import Session

#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *
from get_iEEG_data import *
from morphology_pipeline import *
from iEEG_helper_functions import *

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
# %% import dataframe for leading spikes
lead_spikes_pt1 = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/feasbility tests/HUP_210_spikes_for_Carlos.csv')

#%% load the spike atlas
pt_name = 'HUP210'
spike, atlas, _,_ = load_ptall(pt_name, data_directory)

#%% find spikes that are related to the atlas
#clean labels
def decompose_labels(chLabel, name):
    clean_label = []
    elec = []
    number = []
    label = chLabel

    if isinstance(label, str):
        label_str = label
    else:
        label_str = label[0]

    # Remove leading zero
    label_num_idx = re.search(r'\d', label_str)
    if label_num_idx:
        label_non_num = label_str[:label_num_idx.start()]
        label_num = label_str[label_num_idx.start():]

        if label_num.startswith('0'):
            label_num = label_num[1:]

        label_str = label_non_num + label_num

    # Remove 'EEG '
    eeg_text = 'EEG '
    if eeg_text in label_str:
        label_str = label_str.replace(eeg_text, '')

    # Remove '-Ref'
    ref_text = '-Ref'
    if ref_text in label_str:
        label_str = label_str.replace(ref_text, '')

    # Remove spaces
    label_str = label_str.replace(' ', '')

    # Remove '-'
    label_str = label_str.replace('-', '')

    # Remove CAR
    label_str = label_str.replace('CAR', '')

    # Switch HIPP to DH, AMY to DA
    label_str = label_str.replace('HIPP', 'DH')
    label_str = label_str.replace('AMY', 'DA')

    # Dumb fixes specific to individual patients
    if name == 'HUP099':
        if label_str.startswith('R'):
            label_str = label_str[1:]

    if name == 'HUP189':
        label_str = label_str.replace('Gr', 'G')

    clean_label = label_str

    # Get the non-numerical portion
    label_num_idx = re.search(r'\d', label_str)
    label_non_num = label_str[:label_num_idx.start()]
    elec = label_non_num

    # Get numerical portion
    label_num = re.search(r'\d+', label_str[label_num_idx.start():])
    label_num = int(label_num.group()) if label_num else float('nan')
    number = label_num

    if 'Fp1' in label_str.lower():
        clean_label = 'Fp1'

    if 'Fp2' in label_str.lower():
        clean_label = 'Fp2'

    return clean_label#, elec, number

#clean the labels in the spike dataframe
lead_spikes_pt1['channel_label'] = lead_spikes_pt1['channel_label'].apply(lambda x: decompose_labels(x, pt_name))

#%% check to see how many clean spikes that are true in is_leader, are in the atlas
#find the spikes that are true in is_leader
lead_spikes_leaders = lead_spikes_pt1[lead_spikes_pt1['is_leader'] == True]
#find the spikes that are in the atlas
in_atlas = lead_spikes_leaders[lead_spikes_leaders['channel_label'].isin(atlas['key_0'])]
#find the spikes that are not in the atlas
no_atlas = lead_spikes_leaders[~lead_spikes_leaders['channel_label'].isin(atlas['key_0'])]

#%% merge the atlas and the spike dataframe
#merge the atlas and the spike dataframe
atlas_spikes = pd.merge(in_atlas, atlas[['key_0','final_label']], left_on = 'channel_label', right_on = 'key_0', how = 'left')

#%% pull data from IEEG

with open("/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())

dataset = session.open_dataset('HUP210_phaseII')
all_channel_labels = np.array(dataset.get_channel_labels())

ieeg_data, fs = get_iEEG_data(
            "aguilac",
            "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin",
            "HUP210_phaseII",
            (70886208/1024 * 1e6) - (4 * 1e6),
            (70886208/1024 * 1e6) + (4 * 1e6),
            all_channel_labels,
        )
fs = int(fs)
#%% reference
channel_labels_to_download = all_channel_labels[electrode_selection(all_channel_labels)]

duration_usec = dataset.get_time_series_details(channel_labels_to_download[0]).duration
duration_hours = int(duration_usec / 1000000 / 60 / 60)
enlarged_duration_hours = duration_hours + 24

print(f"Opening {dataset_name} with duration {duration_hours} hours")

# Calculate the total number of 2-minute intervals in the enlarged duration
total_intervals = enlarged_duration_hours * 30  # 60min/hour / 2min = 30

synchrony_broadband_vector_to_save = np.full(total_intervals, np.nan)
synchrony_60_100_vector_to_save = np.full(total_intervals, np.nan)
synchrony_100_125_vector_to_save = np.full(total_intervals, np.nan)

# Loop through each 2-minute interval
for interval in range(total_intervals):
    print(f"Getting iEEG data for interval {interval} out of {total_intervals}")
    duration_usec = 1.2e8  # 2 minutes
    start_time_usec = interval * 2 * 60 * 1e6  # 2 minutes in microseconds
    stop_time_usec = start_time_usec + duration_usec

    try:
        ieeg_data, fs = get_iEEG_data(
            "dma",
            "dma_ieeglogin.bin",
            dataset_name,
            start_time_usec,
            stop_time_usec,
            channel_labels_to_download,
        )
        fs = int(fs)
    except Exception as e:
        # handle the exception
        print(f"Error: {e}")
        break

    # Drop rows that has any nan
    ieeg_data = ieeg_data.dropna(axis=0, how="any")
    if ieeg_data.empty:
        print("Empty dataframe after dropping nan, skip...")
        continue

    good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
    good_channel_indicies = good_channels_res[0]
    good_channel_labels = channel_labels_to_download[good_channel_indicies]
    ieeg_data = ieeg_data[good_channel_labels].to_numpy()

    # Check if ieeg_data is empty after dropping bad channels
    if ieeg_data.size == 0:
        print("Empty dataframe after dropping bad channels, skip...")
        continue

    ieeg_data = common_average_montage(ieeg_data)

    # Apply the filters directly on the DataFrame
    ieeg_data = notch_filter(ieeg_data, 59, 61, fs)

    ##############################
    # Calculate synchrony (broadband)
    ##############################
    _, R = calculate_synchrony(ieeg_data.T)
    synchrony_broadband_vector_to_save[interval] = R

    ##############################
    # Calculate synchrony (60-100Hz)
    ##############################
    ieeg_data_60_100 = bandpass_filter(ieeg_data, 60, 100, fs)
    _, R = calculate_synchrony(ieeg_data_60_100.T)
    synchrony_60_100_vector_to_save[interval] = R

    ##############################
    # Calculate synchrony (100-125Hz)
    ##############################
    try:
        ieeg_data_100_125 = bandpass_filter(ieeg_data, 100, 125, fs)
        _, R = calculate_synchrony(ieeg_data_100_125.T)
        synchrony_100_125_vector_to_save[interval] = R
    except Exception as e:
        print(f"Error: {e}")

    print(f"Finished calculating synchrony for interval {interval}")

    ##############################
    # Detect spikes
    ##############################
    ieeg_data_for_spikes = bandpass_filter(ieeg_data, 1, 70, fs)

    spike_output = spike_detector(
        data=ieeg_data_for_spikes,
        fs=fs,
        labels=good_channel_labels,
    )
    if len(spike_output) == 0:
        print("No spikes detected, skip saving...")
        continue
    else:
        print(f"Detected {len(spike_output)} spikes")


#%% find the variance across
#looks like for HUP201, there is 101k spikes that are "leaders"
#a lot of leaders are in the SOZ. The split is 73k/27k for SOZ/NSOZ. 
#


#%% link the spikes to the atlas

