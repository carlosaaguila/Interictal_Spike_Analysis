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
spikes_thresh = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
spikes = spikes_thresh

#load filenames_w_ids.csv 
filenamescsv = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
filenamescsv = filenamescsv[filenamescsv['to use'] == 1]
multi_file = filenamescsv[filenamescsv['hup_id'].duplicated()]['hup_id'].unique()
pt_ids = filenamescsv[~filenamescsv['hup_id'].isin(multi_file)]

#%%
def plot_train(train, eeg_data, pt_id, i, fs):
    lower_bound_idx = 30*fs - int(0.25*fs)
    upper_bound_idx = np.shape(eeg_data)[0] - (30*fs) + int(0.25*fs)
    
    rows = train.shape[0]
    cols = 1
    fig, axs = plt.subplots(rows, cols, figsize=(5, 10), sharex=True)

    train = train.sort_values(by='order_of_sorting', ascending=True).reset_index(drop=True)
    first_spike = train['order_of_sorting'].min()

    for index, SS in train.iterrows():
        ch_to_plot = SS['channel_label']
        if ch_to_plot in eeg_data.columns.to_list():
            diff_from_first = SS['order_of_sorting'] - first_spike
            PEAK = (30*fs) + diff_from_first
            RIGHT = PEAK + SS['new_right']
            LEFT = PEAK + SS['new_left']
            print("this is the x-value of the peak:", PEAK)
            
            if rows > 1:
                ax = axs[index]
            else:
                ax = axs  # If there's only one subplot, axs is not a list
            
            ax.plot(range(lower_bound_idx, upper_bound_idx), eeg_data[ch_to_plot].iloc[lower_bound_idx:upper_bound_idx], color='k', lw=3)
            ax.get_xaxis().set_visible(False)
            ax.plot(PEAK, eeg_data[ch_to_plot].iloc[PEAK], 'o', color='#00a087', markersize=7)
        else:
            print(f'{ch_to_plot} is a BAD CHANNEL')

    plt.subplots_adjust(hspace=0.1)
    sns.despine()
    plt.tight_layout()

    plt.savefig(f'figures/MUSC+HUP/official/{pt_id}_{i}_example_train.pdf')
    plt.close(fig)  # Close the figure to free up memory


#%%
spikes = spikes[spikes['pt_id'].str.contains('170|179|185|192|187')]
pt_ids_of_interest = spikes['pt_id'].unique()
pt_ids = pt_ids[pt_ids['hup_id'].isin(pt_ids_of_interest)]

for index, row in pt_ids.iterrows():
    #get a patient row.
    pt_id = row['hup_id']
    filename = row['filename']

    #look for spikes in a specific patient
    spikes_oi = spikes[spikes['pt_id'] == pt_id]

    for i in range(5):
        
        #grab a random spike train index
        spike_train_index = spikes_oi['new_spike_seq'].sample(1).values
        # spike_train_index = np.array([5102])
        #grab the random spike train
        train = spikes_oi[spikes_oi['new_spike_seq'] == spike_train_index[0]][['peak_index', 'new_spike_seq','peak_time_usec','channel_label','channel_index','peak','left_point','right_point','slow_end']]
        
        if (len(train)<3) | (len(train)>10):
            while (len(train)<3) | (len(train)>10):
                #grab a random spike train index
                spike_train_index = spikes_oi['new_spike_seq'].sample(1).values
                # spike_train_index = np.array([5102])
                #grab the random spike train
                train = spikes_oi[spikes_oi['new_spike_seq'] == spike_train_index[0]][['peak_index', 'new_spike_seq','peak_time_usec','channel_label','channel_index','peak','left_point','right_point','slow_end']]

        train['new_peak'] = (train['peak'])
        train['new_left'] = train['left_point'] - 1000
        train['new_right'] = train['right_point'] - 1000
        train['order_of_sorting'] = train['peak_index'] + train['new_peak']

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
        good_channel_labels = [decompose_labels(x, pt_id) for x in good_channel_labels]
        CAR_data = pd.DataFrame(CAR_data_filt, columns = good_channel_labels)
        plot_train(train, CAR_data, pt_id, i, fs)