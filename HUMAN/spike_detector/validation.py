#%% import
#import libraries
import os
import numpy as np
import pandas as pd
from ieeg.auth import Session
import matplotlib.pyplot as plt

# Import custom functions
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *

#%% get data
# Get the data
password_bin_filepath = "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin"
with open(password_bin_filepath, "r") as f:
    session = Session("aguilac", f.read())

blocktime = [279792,279852] #in seconds #checks out 1 to 1
blocktime = [92919,92979] #1 to 1
blocktime = [181085,181145] # missing 4 spikes, leaders are right, duplicates in the same channels are dropped

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

#%%
spike_output_df.groupby('sequence_index').count()
spike_output_df[spike_output_df['sequence_index'] == 0]


#%%
