#%% import
#import libraries
import os
import numpy as np
import pandas as pd
from ieeg.auth import Session
import matplotlib.pyplot as plt
import re


# Import custom functions
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *

#%% function to plot spike train
def viz_spiketrain(ieeg_data, sequence, chlabels):
    #function to plot the spike train given from yo
    #check if ieeg_data is a dataframe
    if not isinstance(ieeg_data, pd.DataFrame):    
        ieeg_data = pd.DataFrame(ieeg_data)
    
    offset = 0
    plt.figure(figsize=(10,10))
    #find the first peak_index only in the sequence
    first_peak_index = sequence['peak_index'].iloc[0]
    #plot the first 500 samples before and 1000 samples after, for the channel_label in sequence
    for i, ch in enumerate(sequence['channel_label']):
        where = np.where(chlabels == ch)[0][0]
        to_plot = ieeg_data.iloc[first_peak_index-250:first_peak_index+750, where]
        plt.plot(to_plot - offset, 'k')
        plt.text(first_peak_index+750+10, -offset + np.nanmedian(to_plot), chlabels[where])
        plt.plot((sequence['peak_index'].iloc[i]), ieeg_data.iloc[(sequence['peak_index'].iloc[i]), where] - offset, marker = "x", color = 'r')
        last_min = np.nanmin(to_plot)
        if i < len(sequence['channel_label'])-1:
            next_where = np.where(chlabels == sequence['channel_label'].iloc[i+1])[0][0]
            next_plot = ieeg_data.iloc[first_peak_index-250:first_peak_index+750, next_where]
            offset = offset - (last_min - np.nanmax(next_plot))
        #offset = offset - (np.nanmin(to_plot) - np.nanmax(to_plot))

#%% get data
# Get the data
password_bin_filepath = "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin"
with open(password_bin_filepath, "r") as f:
    session = Session("aguilac", f.read())

blocktime = [279100, 279160]

dataset_name = "HUP210_phaseII"

dataset = session.open_dataset(dataset_name)
all_channel_labels = np.array(dataset.get_channel_labels())
channel_labels_to_download = all_channel_labels[
    electrode_selection(all_channel_labels)
]


ieeg_data, fs = get_iEEG_data(
    "aguilac",
    password_bin_filepath,
    dataset_name,
    blocktime[0]*1e6,
    blocktime[1]*1e6,
    channel_labels_to_download,
)

fs = int(fs)

#%%spike detector
# clean the data
good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
good_channel_indicies = good_channels_res[0]
good_channel_labels = channel_labels_to_download[good_channel_indicies]
ieeg_data = ieeg_data[good_channel_labels].to_numpy()

ieeg_data = common_average_montage(ieeg_data)

# Apply the filters directly on the DataFrame
ieeg_data = notch_filter(ieeg_data, 60, fs) #less than 1 microvolt difference
ieeg_data = new_bandpass_filt(ieeg_data, 1, 70, fs, order = 4) #few differences here by a couple of microvolts

# Detect spikes
spike_output = spike_detector(
    data=ieeg_data,
    fs=fs,
    electrode_labels=good_channel_labels,
)

#clean output
spike_output = spike_output.astype(int)
actual_number_of_spikes = len(spike_output)

channel_labels_mapped = good_channel_labels[spike_output[:, 1]]

spike_output_df = pd.DataFrame(
    spike_output, columns=["peak_index", "channel_label", "sequence_index"]
)
spike_output_df["channel_label"] = channel_labels_mapped

#%% number of spikes detected
print(actual_number_of_spikes)

#%% the spike detector output cleaned
display(spike_output_df)

#%% display how many spikes are in each sequence
display(spike_output_df.groupby('sequence_index').count())

#%% display the sequence
yo = spike_output_df[spike_output_df['sequence_index'] == 3]
display(yo)
print(yo.shape)

# %% Visualize the spike train, confirm that the LEAD spike is actually LEAD
viz_spiketrain(ieeg_data, yo, good_channel_labels)

# %%
