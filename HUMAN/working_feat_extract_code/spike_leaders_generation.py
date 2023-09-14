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

#simple function to plot the a spike sequence
def simple_eeg_plot(values, chLabels, redChannels):
    offset = 0

    plt.figure(figsize=(25, 25))
    for ich in range(values.shape[1]):
        eeg = values.iloc[:, ich].to_numpy()
        if np.any(~np.isnan(eeg)):
            if chLabels[ich] in redChannels:
                color = 'r'
            else:
                color = 'k'
            plt.plot(eeg - offset, color)
            plt.text(len(eeg) + 0.05, -offset + np.nanmedian(eeg), chLabels[ich])
            last_min = np.nanmin(eeg)

        if ich < values.shape[1] - 1:
            next_eeg = values.iloc[:, ich + 1].to_numpy()
            if np.any(~np.isnan(next_eeg)) and not np.isnan(last_min):
                offset = offset - (last_min - np.nanmax(next_eeg))
    #pass an plt.xlim that focuses around the center of the spike
    plt.show()

#create function called simple_eeg_plot_red that only plots the channels in redChannels, and skips the others
def simple_eeg_plot_red(values, chLabels, redChannels):
    offset = 0

    plt.figure(figsize=(10,10))
    for ich in range(values.shape[1]):
        eeg = values.iloc[:, ich].to_numpy()
        if np.any(~np.isnan(eeg)):
            if chLabels[ich] in redChannels:
                color = 'r'
            else:
                continue
            plt.plot(eeg - offset, color)
            plt.text(len(eeg) + 0.05, -offset + np.nanmedian(eeg), chLabels[ich])
            last_min = np.nanmin(eeg)

        if ich < values.shape[1] - 1:
            next_eeg = values.iloc[:, ich + 1].to_numpy()
            if np.any(~np.isnan(next_eeg)) and not np.isnan(last_min):
                offset = offset - (last_min - np.nanmax(next_eeg))
    plt.show()

#function to plot just the lead spike
def simple_eeg_plot_lead(values, chLabels, redChannels):
    offset = 0
    count = 0
    plt.figure(figsize=(10,10))
    for ich in range(values.shape[1]):
        eeg = values.iloc[:, ich].to_numpy()
        if np.any(~np.isnan(eeg)):
            if chLabels[ich] in redChannels:
                color = 'r'
                count = count+1
            else:
                continue
            plt.plot(eeg - offset, color)
            plt.text(len(eeg) + 0.05, -offset + np.nanmedian(eeg), chLabels[ich])
            #plt.text((len(eeg)/2) + 1000.05, -offset + np.nanmedian(eeg), chLabels[ich])
            last_min = np.nanmin(eeg)

        if ich < values.shape[1] - 1:
            next_eeg = values.iloc[:, ich + 1].to_numpy()
            if np.any(~np.isnan(next_eeg)) and not np.isnan(last_min):
                offset = offset - (last_min - np.nanmax(next_eeg))

    if count > 0:
        print('no valid lead channel')

    #plt.xlim(((len(eeg)/2) - 1000, (len(eeg)/2) + 1000))
    plt.show()

#clean labels
def decompose_labels(chLabel, name):
    """
    clean the channel labels, one at a time.
    """
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

    if 'Fp1' in label_str.lower():
        clean_label = 'Fp1'

    if 'Fp2' in label_str.lower():
        clean_label = 'Fp2'

    return clean_label

#a function that will drop any sequence_index that doesn't have over 6 unique channels
def drop_seq(df):
    """
    drop any sequence_index that doesn't have over 5 unique channels
    """
    #find the unique sequence_index
    unique_seq = df['sequence_index'].unique()
    #create a list to store the sequence_index that have over 6 unique channels
    keep_seq = []
    #loop through the unique sequence_index
    for seq in unique_seq:
        #find the number of unique channels in the sequence_index
        num_unique = len(df[df['sequence_index'] == seq]['channel_label'].unique())
        #if the number of unique channels is greater than 6, append the sequence_index to keep_seq
        if num_unique > 6:
            keep_seq.append(seq)
    #return the dataframe with only the sequence_index that have over 6 unique channels
    return df[df['sequence_index'].isin(keep_seq)]

#%%
#clean the labels in the spike dataframe
lead_spikes_pt1['channel_label'] = lead_spikes_pt1['channel_label'].apply(lambda x: decompose_labels(x, pt_name))

#drop any sequence_index that doesn't have over 6 unique channels
lead_spikes_drop = drop_seq(lead_spikes_pt1)

#%% check to see how many clean spikes that are true in is_leader, are in the atlas
#find the spikes that are true in is_leader
lead_spikes_leaders = lead_spikes_drop[lead_spikes_drop['is_leader'] == True]
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

#get the channel_index from lead_spikes_leaders for a random sequence_index
#change the sequence_index == X for a different peak_index
X = 14;
seq_start = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X]['peak_index'].to_list()[0]
ch_labels = all_channel_labels[electrode_selection(all_channel_labels)]

ieeg_data, fs = get_iEEG_data(
            "aguilac",
            "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin",
            "HUP210_phaseII",
            (seq_start/1024 * 1e6) - (5 * 1e6),
            (seq_start/1024 * 1e6) + (5 * 1e6),
            ch_labels,
        )

fs = int(fs)

#%% 
#reject bad channels
channel_mask, details = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)

# Apply CAR Montage
CAR_data = common_average_montage(ieeg_data.to_numpy())
clean_labels = [decompose_labels(x, pt_name) for x in ieeg_data.columns]
#apply bandpass filter
CAR_data_filt = butter_bp_filter(CAR_data, 1, 70, fs, order=4)
CAR_data = pd.DataFrame(CAR_data_filt, columns = clean_labels)

#Apply Bipolar Montage
BIP_data, BIP_labels = automatic_bipolar_montage(ieeg_data.to_numpy(), ieeg_data.columns)
BIP_clean_labels = [decompose_labels(x, pt_name) for x in BIP_labels]
#apply bandpass filter
BIP_data_filt = butter_bp_filter(BIP_data, 1, 70, fs, order=4)
BIP_data = pd.DataFrame(BIP_data_filt, columns = BIP_clean_labels)


#%% plot the spike sequence
#RAW

red_chs = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X].sort_values(['peak_index'])['channel_label'].to_list()
clean_labels = [decompose_labels(x, pt_name) for x in ch_labels]
simple_eeg_plot(ieeg_data, clean_labels, red_chs)
simple_eeg_plot_red(ieeg_data, clean_labels, red_chs)

#plot just the leader spike
lead = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X][lead_spikes_pt1['is_leader'] == True].sort_values(['peak_index'])['channel_label'].to_list()
simple_eeg_plot_lead(ieeg_data, clean_labels, lead)

#%% plot the spike sequence
#CAR

red_chs = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X].sort_values(['peak_index'])['channel_label'].to_list()
clean_labels = [decompose_labels(x, pt_name) for x in ch_labels]
simple_eeg_plot(CAR_data, clean_labels, red_chs)
simple_eeg_plot_red(CAR_data, clean_labels, red_chs)

#plot just the leader spike
lead = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X][lead_spikes_pt1['is_leader'] == True].sort_values(['peak_index'])['channel_label'].to_list()
simple_eeg_plot_lead(CAR_data, clean_labels, lead)


#%% plot the spike sequence
#BIPOLAR

red_chs = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X].sort_values(['peak_index'])['channel_label'].to_list()
clean_labels = [decompose_labels(x, pt_name) for x in ch_labels]
simple_eeg_plot(BIP_data, clean_labels, red_chs)
simple_eeg_plot_red(BIP_data, clean_labels, red_chs)

#plot just the leader spike
lead = lead_spikes_pt1[lead_spikes_pt1['sequence_index'] == X][lead_spikes_pt1['is_leader'] == True].sort_values(['peak_index'])['channel_label'].to_list()
simple_eeg_plot_lead(BIP_data, clean_labels, lead)
# %%
