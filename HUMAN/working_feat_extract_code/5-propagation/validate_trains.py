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

#load spikes
all_spikes = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/spikes_bySOZ.csv')
bilateral_spikes = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/bilateral_MTLE_all_spikes.csv')
bilateral_spikes = bilateral_spikes.drop(['engel','hup_id','name','spike_rate'], axis=1)

#rename 'clinic_SOZ' to 'SOZ'
bilateral_spikes = bilateral_spikes.rename(columns={'clinic_SOZ':'SOZ'})
spikes = pd.concat([all_spikes, bilateral_spikes], axis=0).reset_index(drop=True)

#load filenames_w_ids.csv 
filenamescsv = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
filenamescsv = filenamescsv[filenamescsv['to use'] == 1]
multi_file = filenamescsv[filenamescsv['hup_id'].duplicated()]['hup_id'].unique()
pt_ids = filenamescsv[~filenamescsv['hup_id'].isin(multi_file)]

#%%
def plot_train(train, eeg_data):
    """
    This function will take a train dataframe that contains all the information about a spike train & eeg_data, which is a cleaned common avergae refence dataframe.
    It will return a plot of the spike train, in order of peaks.
    """
    #make it 2 seconds before spike train starts
    lower_bound_idx = 30*fs - int(0.25*fs)
    #make it 2 seconds AFTER spike trains last onset
    upper_bound_idx = np.shape(CAR_data)[0] - (30*fs) + int(0.25*fs)
    
    #shape of the subplots we might want to plot on
    rows = train.shape[0]
    cols = 1
    fig, axs = plt.subplots(rows, cols, figsize=(8, 15), sharex=True)

    #line them up 
    train = train.sort_values(by = 'order_of_sorting', ascending = True).reset_index(drop=True)
    
    first_spike = train['order_of_sorting'].min()

    for i, SS in train.iterrows():
        ch_to_plot = SS['channel_label']
        if ch_to_plot in eeg_data.columns.to_list():
            diff_from_first = SS['order_of_sorting'] - first_spike
            PEAK = (30*fs) + diff_from_first
            RIGHT = PEAK + SS['new_right']
            LEFT = PEAK + SS['new_left']
            print("this is the x-value of the peak:",PEAK)
            axs[i].plot(range(lower_bound_idx, upper_bound_idx), eeg_data[ch_to_plot].iloc[lower_bound_idx:upper_bound_idx])
            axs[i].get_xaxis().set_visible(False)
            axs[i].plot(PEAK, eeg_data[ch_to_plot].iloc[PEAK], 'x', color = 'r')
            axs[i].plot(LEFT, eeg_data[ch_to_plot].iloc[LEFT], 'x', color = 'r')
            axs[i].plot(RIGHT, eeg_data[ch_to_plot].iloc[RIGHT], 'x', color = 'r')

        else:
            print(f'{ch_to_plot} is a BAD CHANNEL')
        plt.subplots_adjust(hspace=0.1)

#%%
#get a patient row.
row = pt_ids.iloc[2]
pt_id = row['hup_id']
filename = row['filename']

#look for spikes in a specific patient
spikes_oi = spikes[spikes['pt_id'] == pt_id]

#grab a random spike train index
spike_train_index = spikes_oi['new_spike_seq'].sample(1).values

#grab the random spike train
train = spikes_oi[spikes_oi['new_spike_seq'] == spike_train_index[0]][['peak_index', 'new_spike_seq','peak_time_usec','channel_label','channel_index','peak','left_point','right_point','slow_end']]

#%%
train['new_peak'] = (train['peak'] - 1000)
train['new_left'] = train['left_point'] - 1000
train['new_right'] = train['right_point'] - 1000
train['order_of_sorting'] = train['peak_index'] + train['new_peak']


# %%
#Load in the spike.
with open("/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())

dataset = session.open_dataset(filename)

all_channel_labels = np.array(dataset.get_channel_labels())

#change the sequence_index == X for a different peak_index
ch_labels = all_channel_labels[electrode_selection(all_channel_labels)]

fs = int(dataset.get_time_series_details(dataset.ch_labels[0]).sample_rate)  # get sample rate

#find a minute of data around the spike train we want.
step_in_usec = (1 * fs) * 1e6 #1second in u_sec
lower_bound = (train['peak_time_usec']+((train['new_peak']/fs) * 1e6)).min()
upper_bound = (train['peak_time_usec']+((train['new_peak']/fs) * 1e6)).max()

ieeg_data, fs = get_iEEG_data(
                            "aguilac",
                            "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin",
                            filename,
                            (lower_bound) - (30 * 1e6),
                            (upper_bound) + (30 * 1e6),
                            ch_labels
                        )

fs = int(fs)

#look for bad channels
good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
good_channel_indicies = good_channels_res[0]
good_channel_labels = ch_labels[good_channel_indicies]
ieeg_data = ieeg_data[good_channel_labels].to_numpy()

# Apply CAR Montage
CAR_data = common_average_montage(ieeg_data)

#apply bandpass filter
CAR_data_filt = new_bandpass_filt(CAR_data, 1, 70, fs, order=4)
CAR_data = pd.DataFrame(CAR_data_filt, columns = good_channel_labels)

# %%
plot_train(train, CAR_data)
# # %%
# #make it 2 seconds before spike train starts
# lower_bound_idx = 30*fs - int(0.25*fs)
# #make it 2 seconds AFTER spike trains last onset
# upper_bound_idx = np.shape(CAR_data)[0] - (30*fs) + int(0.25*fs)

# #line them up 
# train['order_of_sorting'] = train['peak_index'] + train['new_peak']
# train = train.sort_values(by = 'order_of_sorting', ascending = True).reset_index(drop=True)

# first_spike = train['order_of_sorting'].min()

# SS = train.iloc[0]
# ch_to_plot = SS['channel_label']
# plt.plot(range(lower_bound_idx,upper_bound_idx), CAR_data[ch_to_plot].iloc[lower_bound_idx:upper_bound_idx])

