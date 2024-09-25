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

spikes_thresh = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']

    #if channel_label contains any of the strings in chs_tokeep, keep it
spikes_thresh = spikes_thresh[spikes_thresh['channel_label'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

# spikes_thresh = spikes_thresh[spikes_thresh['filename'].str.contains('170|179|185|192|187')]
spikes_thresh = spikes_thresh[spikes_thresh['filename'].str.contains('192|204|215|105|221')]
pt_ids = spikes_thresh['filename'].unique()
spikes = spikes_thresh

with open("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/error_log.txt", "a") as log_file:
    for index, filename in enumerate(pt_ids):
        pt_id = spikes[spikes['filename'] == filename]['pt_id'].iloc[0]
        rows = 3
        cols = 2
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(10,10)) 

    
        for i in range(num_plots):
            #find all spikes in HUP105
            spikes_oi = spikes[spikes['filename'] == filename]

            #pick a random spike in record
            single_spike = spikes_oi.sample(1)

            # find important characteristics of that random spike
            peak_time_usec = single_spike['peak_time_usec']
            ch_oi = single_spike['channel_label']

            #Load in the spike.
            with open("/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin", "r") as f:
                session = Session("aguilac", f.read())

            dataset = session.open_dataset(filename)

            all_channel_labels = np.array(dataset.get_channel_labels())

            #change the sequence_index == X for a different peak_index
            ch_labels = all_channel_labels[electrode_selection(all_channel_labels)]

            #load in 1 minute centered around the spike (for CAR)
            ieeg_data, fs = get_iEEG_data(
                        "aguilac",
                        "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin",
                        filename,
                        (peak_time_usec) - (30 * 1e6),
                        (peak_time_usec) + (30 * 1e6),
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

            
            #get times surrounding the middle
            start = int(len(CAR_data[ch_oi])/2 - 1.5*fs)
            stop = int(len(CAR_data[ch_oi])/2 + 1.5*fs)
            mid = int(len(CAR_data[ch_oi])/2)
            plt_samps = np.arange(start,stop,1)

            try:
                #plot morphology points
                peak = mid + single_spike['peak'].values[0]
                left = mid + (single_spike['left_point'].values[0] - 1000)
                right = mid + (single_spike['right_point'].values[0] - 1000)
                slow_end = mid + (single_spike['slow_end'].values[0]-1000)
            except:
                print('no morphology feats, skip...')
                continue

            #plot the spike we are looking for.
            row = i // cols
            col = i % cols
            ax = axes[row,col]
            ax.plot(plt_samps, CAR_data[ch_oi][start:stop])
            #plot morphology
            ax.plot(peak, CAR_data[ch_oi].iloc[peak], 'o', color = 'r', label = 'Peak')
            ax.plot(left, CAR_data[ch_oi].iloc[left], 'o', color = 'b')
            ax.plot(right, CAR_data[ch_oi].iloc[right], 'o', color = 'b')
            ax.plot(slow_end, CAR_data[ch_oi].iloc[slow_end], 'o', color = 'k')
            ax.set_title(f'Spike at {ch_oi.values[0]}')
            ax.set_xlabel('Sample', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)


        fig.suptitle(f'Random Spikes for {pt_id}', fontsize = 20)
        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.5)  # Increase  spacing
        plt.subplots_adjust(top = 0.93)
        # plt.show()

        plt.savefig(f'/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/supplemental_figs/random_detects_and_morph/{pt_id}_supplementspikes_2.pdf')




# %%
