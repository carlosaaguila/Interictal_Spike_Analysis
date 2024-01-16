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

spikes = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/spikes_bySOZ_T-R_v2.csv', index_col = 0)

#%%
seq = 100
spiketrain = spikes[spikes['new_spike_seq'] == seq]

#pick a random 'pt_id' in spiketrain 
pt_id = spiketrain['pt_id'].sample(1).values[0]

#grab all spikes for that pt_id
pt_spikes = spiketrain[spiketrain['pt_id'] == pt_id]

# %%
with open("/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())

#load filenames_w_ids.csv 
filenamescsv = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
#look for the pt_id in filenamescsv
pt_filenames = filenamescsv[filenamescsv['hup_id'] == pt_id]
#grab the filename for that pt_id where to_use == 1
filename = pt_filenames[pt_filenames['to use'] == 1]['filename'].values[0]
dataset = session.open_dataset(filename)

all_channel_labels = np.array(dataset.get_channel_labels())

#get the channel_index from lead_spikes_leaders for a random sequence_index
#change the sequence_index == X for a different peak_index
seq_start = pt_spikes['peak_index'].to_list()[0]
ch_labels = all_channel_labels[electrode_selection(all_channel_labels)]
#get the peak_time_usec for pt_spikes of the is_leaderspike ==1
peak_time_usec = pt_spikes[pt_spikes['is_spike_leader'] == 1]['peak_time_usec'].values[0]

ieeg_data, fs = get_iEEG_data(
            "aguilac",
            "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin",
            filename,
            (seq_start/1024 * 1e6) - (1 * 1e6),
            (seq_start/1024 * 1e6) + (1 * 1e6),
            ch_labels
        )

fs = int(fs)
# %%
# order the pt_spikes by 'peak_index_samples'
pt_spikes = pt_spikes.sort_values(by=['peak_index_samples'])

#only keep the columns in ieeg_data that are in pt_spikes['channel_label']
# ieeg_data = ieeg_data[pt_spikes['channel_label'].to_list()]

channel_mask, details = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)

pt_name=pt_id
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

    if name == 'HUP106':
        label_str = label_str.replace('LDA', 'LA')
        label_str = label_str.replace('LDH', 'LH')
        label_str = label_str.replace('RDA', 'RA')
        label_str = label_str.replace('RDH', 'RH')

    clean_label = label_str

    if 'Fp1' in label_str.lower():
        clean_label = 'Fp1'

    if 'Fp2' in label_str.lower():
        clean_label = 'Fp2'

    return clean_label

simple_eeg_plot_red(CAR_data, all_channel_labels, pt_spikes['channel_label'].to_list())
display(pt_spikes[['peak_time_usec','channel_label','is_spike_leader','rise_amp']])
# %%
