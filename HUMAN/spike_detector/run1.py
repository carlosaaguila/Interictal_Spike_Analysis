#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session

# Import custom functions
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *

import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

#load all the filenames (long form IEEG filenames)
filenames_w_ids = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
#load the list of patients to exclude
blacklist = ['HUP101' ,'HUP112','HUP115','HUP119','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
# remove the patients in the blacklist from filenames_w_ids
filenames_w_ids = filenames_w_ids[~filenames_w_ids['hup_id'].isin(blacklist)]
#only keep rows where the column "to use" is a 1
filenames_w_ids = filenames_w_ids[filenames_w_ids['to use'] == 1]

#split filenames_w_ids dataframe into 10 dataframes
pt_files_split = np.array_split(filenames_w_ids, 7)

#%% load the session
#use Carlos's Session
password_bin_filepath = "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin"
with open(password_bin_filepath, "r") as f:
    session = Session("aguilac", f.read())

#%% loop through each patient

#pick a split dataframe from the 7 from pt_files_split
pt_files = pt_files_split[0]

#loop through each patient
for index, row in pt_files.iterrows():
    hup_id = row['hup_id']
    dataset_name = row['filename']

    print("\n")
    print(f"------Processing HUP {hup_id} with dataset {dataset_name}------")

    ########################################
    # Get the data from IEEG
    ########################################

    dataset = session.open_dataset(dataset_name)

    all_channel_labels = np.array(dataset.get_channel_labels())
    channel_labels_to_download = all_channel_labels[
        electrode_selection(all_channel_labels)
    ]

    duration_usec = dataset.get_time_series_details(
        channel_labels_to_download[0]
    ).duration
    duration_hours = int(duration_usec / 1000000 / 60 / 60)
    enlarged_duration_hours = duration_hours + 24

    print(f"Opening {dataset_name} with duration {duration_hours} hours")

    # Calculate the total number of 1-minute intervals in the enlarged duration
    total_intervals = enlarged_duration_hours * 60  # 60min/hour / 1min = 60

    # Choose 5 unique random intervals before the loop
    chosen_intervals = random.sample(range(total_intervals), 5)
    print(f"Chosen intervals: {chosen_intervals}")

    # Loop through each 2-minute interval
    for interval in range(total_intervals):
        print(
            f"Getting iEEG data for interval {interval} out of {total_intervals} for HUP {hup_id}"
        )
        duration_usec = 6e7  # 1 minute
        start_time_usec = interval * 6e7  # 1 minutes in microseconds
        stop_time_usec = start_time_usec + duration_usec

        try:
            ieeg_data, fs = get_iEEG_data(
                "aguilac",
                "agu_ieeglogin.bin",
                dataset_name,
                start_time_usec,
                stop_time_usec,
                channel_labels_to_download,
            )
            fs = int(fs)
            if interval in chosen_intervals:
                save_path = os.path.join(
                    RANDOMLY_SAVE_CLIPS_DIR,
                    f"ieeg_data_{dataset_name}_{interval}.pkl",
                )
                with open(save_path, "wb") as file:
                    pickle.dump(ieeg_data, file)
                print(f"Saved ieeg_data segment to {save_path}")
        except:
            continue

        # Check if ieeg_data dataframe is all NaNs
        if ieeg_data.isnull().values.all():
            print(f"Empty dataframe after download, skip...")
            continue

        good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
        good_channel_indicies = good_channels_res[0]
        good_channel_labels = channel_labels_to_download[good_channel_indicies]
        ieeg_data = ieeg_data[good_channel_labels].to_numpy()

        # Check if ieeg_data is empty after dropping bad channels
        if ieeg_data.size == 0:
            print(f"Empty dataframe after artifact rejection, skip...")
            continue

        ieeg_data = common_average_montage(ieeg_data)

        # Apply the filters directly on the DataFrame
        ieeg_data = notch_filter(ieeg_data, 59, 61, fs)
        ieeg_data = bandpass_filter(ieeg_data, 1, 70, fs)

        ##############################
        # Detect spikes
        ##############################

        spike_output = spike_detector(
            data=ieeg_data,
            fs=fs,
            electrode_labels=good_channel_labels,
        )
        spike_output = spike_output.astype(int)
        actual_number_of_spikes = len(spike_output)

        if actual_number_of_spikes == 0:
            print(f"No spikes detected, skip saving...")
            continue
        else:
            # Map the channel indices to the corresponding good_channel_labels
            channel_labels_mapped = good_channel_labels[spike_output[:, 1]]

            # Create the structured array
            spike_output_to_save = np.array(
                list(
                    zip(
                        channel_labels_mapped,
                        spike_output[:, 0],
                        spike_output[:, 2],
                    )
                ),
                dtype=dt,
            )
            np.save(
                os.path.join(SPIKES_OUTPUT_DIR, f"{dataset_name}_{interval}.npy"),
                spike_output_to_save,
            )
            print(
                f"Saved {actual_number_of_spikes} spikes to {dataset_name}_{interval}.npy"
            )
# Restore the standard output to its original value
sys.stdout = original_stdout
