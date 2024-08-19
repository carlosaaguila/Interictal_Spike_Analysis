##########################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ieeg.auth import Session
from scipy.signal import convolve2d
from scipy import stats
from scipy import signal
import mne
import json
import pandas as pd
import os
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from multiprocessing import Pool
from utils import get_iEEG_data, notch_filter


# iEEG Functions

#  Detect Artifacts : from Akash
def detect_artifacts(data: np.ndarray, fs: float, discon=1/12, noise=15000, win_size=1) -> np.ndarray:
    win_size_samples = int(win_size * fs)
    print(f'ceiling of datashape {np.ceil(data.shape[0])}')
    print(f'win size samples {win_size_samples}')
    n_wins = int(np.ceil(data.shape[0] / win_size_samples))
    print(f'n_wins: {n_wins}')
    max_inds = n_wins * win_size_samples
    all_inds = np.arange(max_inds, dtype=float)  # Ensure it's floating-point
    all_inds[data.shape[0]:] = np.nan  # Fill excess with NaN
    ind_overlap = np.reshape(all_inds, (-1, win_size_samples))
    artifacts = np.zeros_like(data, dtype=bool)

    for win_inds in ind_overlap:
        # print("NaNs present:", np.isnan(win_inds).any())
        win_inds = win_inds[~np.isnan(win_inds)].astype(int)
        if win_inds.size == 0:
            continue

        window_data = data[win_inds, :]
        is_disconnected = np.sum(np.abs(window_data), axis=0) < discon
        is_noise = np.sqrt(np.sum(np.power(np.diff(window_data, axis=0), 2), axis=0)) > noise
        artifacts[win_inds, :] = np.logical_or(is_disconnected, is_noise)

    return artifacts

# Remove artifacts if over 20% noisy, interpolate if under
def remove_or_interpolate(data, artifacts, channel_names, threshold=0.2):
    num_samples = data.shape[0]
    clean_data = np.copy(data)
    updated_channel_names = []
    removed_channels = []
    interpolated_channels = []

    # Create a mask for channels to keep
    keep_mask = np.ones(data.shape[1], dtype=bool)  

    for ch in range(data.shape[1]):
        artifact_count = np.sum(artifacts[:, ch])
        artifact_percentage = artifact_count / num_samples
        
        if artifact_percentage > threshold:
            keep_mask[ch] = False
            removed_channels.append(channel_names[ch])
        else:
            clean_data[:, ch] = interpolate_channel(data[:, ch], artifacts[:, ch])
            if np.any(artifacts[:, ch]):
                interpolated_channels.append(channel_names[ch])
            updated_channel_names.append(channel_names[ch])

    # Remove channels marked for removal
    clean_data = clean_data[:, keep_mask]
    updated_channel_names = [name for i, name in enumerate(channel_names) if keep_mask[i]]

    return clean_data, updated_channel_names, removed_channels, interpolated_channels


# Interpolate Channels
def interpolate_channel(channel_data: np.ndarray, artifacts: np.ndarray) -> np.ndarray:
    valid_data = np.copy(channel_data)
    valid_data[artifacts] = np.nan
    interpolated_data = pd.Series(valid_data).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').to_numpy()
    return interpolated_data

# Decompose Channel labels
def decompose_labels(chLabel):
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
    clean_label = label_str

    if 'Fp1' in label_str.lower():
        clean_label = 'Fp1'

    if 'Fp2' in label_str.lower():
        clean_label = 'Fp2'

    return clean_label

def check_channel_types(ch_list, threshold=15):
    ch_df = pd.DataFrame([{
        "name": ch, 
        "lead": re.match(r"(\D+)(\d+)?", ch, re.IGNORECASE).group(1).upper() if re.match(r"(\D+)(\d+)?", ch, re.IGNORECASE) else "MISC", 
        "contact": int(re.match(r"(\D+)(\d+)?", ch, re.IGNORECASE).group(2)) if re.match(r"(\D+)(\d+)?", ch, re.IGNORECASE) and re.match(r"(\D+)(\d+)?", ch, re.IGNORECASE).group(2) else 0
    } for ch in ch_list])
    
    # Initialize type column to 'misc'
    ch_df['type'] = 'misc'

    def debug_channel_assignment(lead, group):
        print(f"Processing lead: {lead}")
        print(f"Group indices: {group.index.tolist()}")
        print(f"Current type assignments: {ch_df.loc[group.index, 'type'].tolist()}")

    # Assign types based on lead names or count of contacts
    for lead, group in ch_df.groupby("lead"):
        debug_channel_assignment(lead, group)
        if lead in ["ECG", "EKG", "EKGL", "EKGR"]:
            ch_df.loc[group.index, "type"] = "ecg"
        elif lead in ["C", "CZ", "F", "FP", "FZ", "O", "P", "PZ", "T", "EEG", "A"]:
            ch_df.loc[group.index, "type"] = "eeg"
        elif len(group) > threshold:
            ch_df.loc[group.index, "type"] = "ecog"
        else:
            ch_df.loc[group.index, "type"] = "seeg"
    
    ch_df['type'] = ch_df['type'].fillna('misc')
    return ch_df.groupby('type')['name'].apply(list).to_dict()


def open_ieeg_session(pw_path):
    with open(pw_path, 'r') as f:
        session = Session('aguilac', f.read().strip())
    return session

def get_ieeg_dataset(session, dataset_name):
    return session.open_dataset(dataset_name)


def format_dataset_name(hupid, ieeg_filename_df):
    numeric_part = re.sub(r'\D', '', hupid)  # Remove non-digit characters
    return numeric_part


def process_patient_data(args):
    row, pw_path, ieeg_filename_df = args
    session = open_ieeg_session(pw_path)
    return process_single_patient(row, session, pw_path, ieeg_filename_df)

def format_hupid(numeric_part):
    return f"HUP{int(numeric_part):03}"

def find_matching_row(numeric_part, ictal_start, df, time_tolerance=1):
    formatted_hupid = format_hupid(numeric_part)
    df['time_lower'] = df['start'] - time_tolerance
    df['time_upper'] = df['start'] + time_tolerance

    matching_rows = df[(df['Patient'] == formatted_hupid) &
                       (df['time_lower'] <= ictal_start) &
                       (df['time_upper'] >= ictal_start)]

    if not matching_rows.empty:
        return matching_rows['IEEGname'].iloc[0]
    else:
        return None
    


# EI Functions: https://github.com/allucas/IEEG_EI
# 

# In[6]:


def compute_hfer(target_data, base_data, fs):
    '''
    :param target_data: (Channels x Time) data with pre-ictal to ictal transition
    :param base_data: (Channels x Time) pre-ictal baseline data
    :param fs: sampling frequency
    :return: normalized high frequency energy for baseline and target data
    '''
    target_sq = target_data ** 2
    base_sq = base_data ** 2
    window = int(fs / 2.0)
    target_energy=convolve2d(target_sq,np.ones((1,window)),'same')
    base_energy=convolve2d(base_sq,np.ones((1,window)),'same')
    base_energy_ref = np.sum(base_energy, axis=1) / base_energy.shape[1]
    target_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, target_energy.shape[1]))
    base_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, base_energy.shape[1]))
    norm_target_energy = target_energy / target_de_matrix.astype(np.float32)
    norm_base_energy = base_energy / base_de_matrix.astype(np.float32)
    return norm_target_energy, norm_base_energy

#% Extra functions
def determine_threshold_onset(target, base):
    base_data = base.copy()
    target_data = target.copy()
    sigma = np.std(base_data, axis=1, ddof=1)
    channel_max_base = np.max(base_data, axis=1)
    thresh_value = channel_max_base + 10 * sigma
    print(f"Threshold value: {thresh_value}")
    onset_location = np.zeros(shape=(target_data.shape[0],))
    for channel_idx in range(target_data.shape[0]):
        logic_vec = target_data[channel_idx, :] > thresh_value[channel_idx]
        if np.sum(logic_vec) == 0:
            onset_location[channel_idx] = len(logic_vec)
        else:
            onset_location[channel_idx] = np.where(logic_vec != 0)[0][0]
    return onset_location

def compute_ei_index(target, base, fs):
    target, base = compute_hfer(target, base, fs)
    ei = np.zeros([1, target.shape[0]])
    hfer = np.zeros([1, target.shape[0]])
    onset_rank = np.zeros([1, target.shape[0]])
    channel_onset = determine_threshold_onset(target, base)
    print(f'channel onset: {channel_onset}')
    #added this
    if channel_onset.size == 0:
        print("No seizure onset detected.")
        return np.array([])
    
    seizure_location = np.min(channel_onset)
    onset_channel = np.argmin(channel_onset)
    hfer = np.sum(target[:, int(seizure_location):int(seizure_location + 0.25 * fs)], axis=1) / (fs * 0.25)
    print(f'seizure location: {seizure_location}')
    print(f'onset channel: {onset_channel}')
    print(f"HFER: {hfer}")
    onset_asend = np.sort(channel_onset)
    time_rank_tmp = np.argsort(channel_onset)
    onset_rank = np.argsort(time_rank_tmp) + 1
    onset_rank = np.ones((onset_rank.shape[0],)) / np.float32(onset_rank)
    print(f'onset rank: {onset_rank}')
    ei = np.sqrt(hfer * onset_rank)
    for i in range(len(ei)):
        if np.isnan(ei[i]) or np.isinf(ei[i]):
            ei[i] = 0
    if np.max(ei) > 0:
        ei = ei / np.max(ei)
    return hfer #ei

def get_threshold(norm_base_data, sd_val=10):
    '''
    :param norm_base_data: (channels x time) normalized baseline energy data
    :param sd_val: (int) how many standard deviations above the mean baseline energy to define the threshold per channel
    :return: thresh: threshold per channel
    '''
    thresh = np.max(norm_base_data,axis=1) + (sd_val*np.std(norm_base_data,axis=1, ddof=1))
    return thresh

def get_onset(norm_target_energy, norm_base_energy):

    # get the threshold for each channel
    thresh = get_threshold(norm_base_energy, 10)
    print(f'threhold: {thresh}')
    # define the onset vector
    onset = []

    # compute the onset time for each channel
    for i in range(len(thresh)):
        if np.sum(norm_target_energy[i,:] > thresh[i])>0:
            onset.append(np.argwhere(norm_target_energy[i,:] > thresh[i])[0][0])
        else:
            # if no onset is found, assign the onset time to the length of the signal
            onset.append(len(norm_target_energy[i,:]))

    rank = stats.rankdata(onset)+1
    
    #rank = np.argsort(onset)+1
    tc = 1/rank
    print(f'rank: {rank}')
    print(f'tc: {tc}')
    print(f"Min Onset = {np.min(onset)}")
    return onset, tc

def calculate_ei(norm_target_energy, fs, onset, tc):
    ec = np.mean(norm_target_energy[onset:onset + int(fs * 0.250)]) # energy coefficient, mean of energy 250ms after detection
    return np.sqrt(ec*tc)

def get_ei_all(target, base, fs):
    norm_target_energy, norm_base_energy = compute_hfer(target, base, fs)
    onset, tc = get_onset(norm_target_energy, norm_base_energy)
    ei_vec = []
    for i in range(len(onset)):
        ei_vec.append(calculate_ei(norm_target_energy[i,:],fs,np.min(onset), tc[i]))
    ei_vec = np.array(ei_vec)

    if np.isnan(np.nanmax(ei_vec)):
        ei_vec = np.zeros(len(ei_vec))
    else:
        ei_vec[np.isnan(ei_vec)] = 0
        ei_vec = ei_vec / np.nanmax(ei_vec)
    return ei_vec

def get_ei_from_data(data, fs,  bl_range, target_range):
    data = data.T
    # filter the signal
    if int(fs/2)<140:
        b, a = signal.butter(4, [70,int(fs/2)-1], 'bandpass', fs=fs)
    else:
        b, a = signal.butter(4, [70, 140], 'bandpass', fs=fs)
    data_filt = signal.filtfilt(b, a, data)
    try:
        # ei = get_ei_all(data_filt[:,20000:60000],data_filt[:,:20000],fs=fs)
        ei = compute_ei_index(data_filt[:, target_range[0]:target_range[1]], data_filt[:, bl_range[0]:bl_range[1]], fs=fs)
    except Exception as e:
            print(f"An error occurred in EI calc: {e}, {bl_range}, {target_range}")
    return ei

def save_ei(directory, fname, ei, ch_names):
    ei_table = []
    for i in range(len(ei)):
        ei_table.append([ch_names[i], ei[i]])
    ei_table = np.array(ei_table, dtype=object)
    np.savetxt(os.path.join(directory, fname), ei_table, delimiter=',', fmt='%s')



# HUP Patient Workflow (uncomment for HUP)

# In[7]:


# Function to initialize iEEG session
def open_ieeg_session(pw_path):
    with open(pw_path, 'r') as f:
        session = Session('aguilac', f.read().strip())
    return session

# def find_dataset(session, hupid):
#     numeric_part = re.findall(r'\d+', hupid)[0]  # This will find all numeric characters in the string
#     base_name = f"HUP{int(numeric_part):03d}"
#     extensions = ["_phaseII", "_D01", "_D02", "_D03", "_D04", "_D01_phaseII", "_D02_phaseII", "b_phaseII", "c_phaseII"]
#     for ext in extensions:
#         dataset_name = base_name + ext
#         try:
#             dataset = session.open_dataset(dataset_name)
#             return dataset
#         except Exception as e:
#             print(f"Attempted {dataset_name}, but failed to open: {e}")
#     print(f"Failed to find dataset for {hupid}, skipping.")
#     return None

def get_ieeg_dataset(session, dataset_name):
    return session.open_dataset(dataset_name)

def format_dataset_name(hupid, ieeg_filename_df):
    numeric_part = re.sub(r'\D', '', hupid)  # Remove non-digit characters

def check_channel_types(ch_list, threshold=15):
    ch_df = pd.DataFrame([{"name": ch, "lead": re.match(r"(\D+)(\d+)", ch).groups()[0] if re.match(r"(\D+)(\d+)", ch) else "misc", "contact": int(re.match(r"(\D+)(\d+)", ch).groups()[1]) if re.match(r"(\D+)(\d+)", ch) else 0} for ch in ch_list])
    
    # Assign types based on lead names or count of contacts
    for lead, group in ch_df.groupby("lead"):
        if lead in ["ECG", "EKG"]:
            ch_df.loc[group.index, "type"] = "ecg"
        elif lead in ["C", "Cz", "CZ", "F", "Fp", "FP", "Fz", "FZ", "O", "P", "Pz", "PZ", "T", "EEG"]:
            ch_df.loc[group.index, "type"] = "eeg"
        elif len(group) > threshold:
            ch_df.loc[group.index, "type"] = "ecog"
        else:
            ch_df.loc[group.index, "type"] = "seeg"
    
    ch_df['type'] = ch_df.get('type', 'misc')
    return ch_df.groupby('type')['name'].apply(list).to_dict()

def process_patient_data(args):
    row, pw_path, ieeg_filename_df = args
    session = open_ieeg_session(pw_path)
    return process_single_patient(row, session, pw_path, ieeg_filename_df)

def format_hupid(numeric_part):
    return f"HUP{int(numeric_part):03}"

def find_matching_row(numeric_part, ictal_start, df, time_tolerance=1):
    formatted_hupid = format_hupid(numeric_part)
    # Convert times into a comparable format, assuming times are in seconds or convert as needed
    df['time_lower'] = df['start'] - time_tolerance
    df['time_upper'] = df['start'] + time_tolerance

    # Find matching row
    matching_rows = df[(df['Patient'] == formatted_hupid) &
                       (df['time_lower'] <= ictal_start) &
                       (df['time_upper'] >= ictal_start)]

    if not matching_rows.empty:
        return matching_rows['IEEGname'].iloc[0]
    else:
        return None
    
def process_single_patient(row, session, pw_path, ieeg_filename_df):
    start_time = row['ictal_start'] 
    hupid = row['hupid']
    selected_channels = row['channel_label']
    numeric_part = re.sub(r'\D', '', hupid)  # Remove non-digit characters
    numeric_part = int(numeric_part)  # Convert to integer to remove leading zeros and back to string
    dataset_name = find_matching_row(numeric_part, start_time, ieeg_filename_df)
    print(dataset_name)
    if dataset_name == '':
        print(f"No matching dataset found for HUP ID {hupid}, {numeric_part}")
        return
    # time points in seconds
    bl_start = (start_time - 200)
    bl_end = (start_time - 140)
    target_end = (start_time + 80)

    # time points in u seconds
    start_usec = bl_start*1e6
    stop_usec = target_end*1e6
    try:
        # get iEEG_data
        df, fs = get_iEEG_data(
            username='aguilac',
            password_bin_file=pw_path,
            iEEG_filename=dataset_name,
            start_time_usec=start_usec,
            stop_time_usec=stop_usec
        )

        # Apply notch filters at 60 Hz and harmonics
        data = notch_filter(df.to_numpy(), fs) # Change to Bandstop or remove the Phase and Amplitude of 60z & Harms

        if fs > 500:
            print(f'df before downsample = {df}')
            # Downsample
            # Target sampling frequency
            target_fs = 500
            fs = int(fs)
            # Calculate the resampling factors
            gcd = np.gcd(fs, target_fs)
            up = target_fs // gcd
            down = fs // gcd
            # Resample each channel using polynomial interpolation
            downsampled_data = signal.resample_poly(data, up, down, axis=0)
            # Convert the downsampled numpy array back to a DataFrame
            df_downsampled = pd.DataFrame(downsampled_data, columns=df.columns)
            print(f'df after downsample = {df_downsampled}')
            # New sampling frequency
            fs = target_fs
            data = df_downsampled

        try:
            # Include only channel labels from list
            # -------------------------------------

            channel_names = df.columns.tolist()  # Ensure it's a list if using .index
            data = df.to_numpy()  # Explicit conversion to numpy array for operations
            print(f'original data shape: {data.shape}')
            channel_types = check_channel_types(channel_names)
            exclude_types = {'eeg', 'ecg', 'misc'}
            valid_channels = [ch for ctype, channels in channel_types.items() if ctype not in exclude_types for ch in channels]

            # selected_channels_clean = [decompose_labels(name) for name in selected_channels]
            cleaned_channel_names = [decompose_labels(name) for name in channel_names]
            cleaned_valid_channel_names = [decompose_labels(name) for name in valid_channels]

            # Print the length of each list
            print("Length of cleaned_channel_names:", len(cleaned_channel_names))
            print("Length of valid_channel_names_clean:", len(cleaned_valid_channel_names))
            print("Length of selected_channels:", len(selected_channels))

            # Check the type of each element in each list
            print("\nTypes in cleaned_channel_names:")
            for item in cleaned_channel_names:
                print(type(item), item)

            print("\nTypes in valid_channel_names_clean:")
            for item in cleaned_valid_channel_names:
                print(type(item), item)

            # Ensure all elements are strings
            cleaned_channel_names = [str(item) for item in cleaned_channel_names]
            cleaned_valid_channel_names = [str(item) for item in cleaned_valid_channel_names]

            if all(len(item) == 1 for item in selected_channels):
                # It seems like selected_channels was split into characters, let's join them back
                joined_string = ''.join(selected_channels)
                
                selected_channels = re.findall(r'[A-Z]+[0-9]+', joined_string)

            print("\nTypes in selected_channels:")
            for item in selected_channels:
                print(type(item), item)
                
            set_channel_names_clean = set(cleaned_channel_names)
            set_valid_channel_names_clean = set(cleaned_valid_channel_names)
            set_selected_channels_clean = set(selected_channels)

            # Find mutual names
            mutual_names = set_selected_channels_clean & set_channel_names_clean 

            print(f'Mutual names: {mutual_names}')

            # Check if there are mismatches
            for ch in selected_channels:
                if ch not in cleaned_channel_names:
                    print(f'{ch} from selected channels is not in cleaned channel names')

            # Find indices of selected channels that are valid and exist in the cleaned list
            selected_valid_indices = [
                cleaned_channel_names.index(ch)
                for ch in selected_channels
                if ch in cleaned_valid_channel_names and ch in cleaned_channel_names
            ]

            data = data[:, selected_valid_indices]

            print(f" Data after removing types: {data.shape}")
            artifacts = detect_artifacts(data, fs)
            clean_data, updated_channel_names, removed_channels, interpolated_channels = remove_or_interpolate(data, artifacts, [channel_names[i] for i in selected_valid_indices])
        except Exception as e:
            print(f"An error occurred in artifact removal: {e}")
        print(f"Data and sampling rate for {row['hupid']} acquired, clean_data shape is {clean_data.shape}")
        print(f'clean data: {clean_data}')
        try: 
            bl_start_samples = int(bl_start*fs)
            bl_end_samples = int(bl_end*fs)
            target_end_samples = int(target_end*fs)
            print(f'bl start samples = {bl_start_samples}, bl end samples = {bl_end_samples}, target end samples = {target_end_samples}')
            print((bl_end_samples - bl_start_samples), (bl_end_samples - bl_start_samples), (target_end_samples- bl_start_samples))

            ei = get_ei_from_data(clean_data, fs, [0, bl_end_samples - bl_start_samples], [bl_end_samples - bl_start_samples, target_end_samples- bl_start_samples])
            print(f"EI calulated for {row['hupid']}")
            return {
                'hupID': row['hupid'],
                'rid': row['rid'],
                'ictal_start_time': start_time,
                'EI': ei,
                'name': updated_channel_names,
                'removed_channels': removed_channels
            }
        except Exception as e:
            print(f"An error occurred in EI calc: {e}")
    except Exception as e:
        print(f"Error processing patient {row['hupid']}: {e}")


# In[9]:


import time

def process_all_patients(df, pw_path, ieeg_filename_df):
    with Pool(8) as pool:
        args = [(row, pw_path, ieeg_filename_df) for index, row in df.iterrows()]
        start_time = time.time()
        results = []

        for i, result in enumerate(pool.imap(process_patient_data, args)):
            results.append(result)
            elapsed_time = time.time() - start_time
            average_time_per_item = elapsed_time / (i + 1)
            remaining_items = len(args) - (i + 1)
            estimated_remaining_time = average_time_per_item * remaining_items
            print(f'Processed {i + 1}/{len(args)}. Estimated time remaining: {estimated_remaining_time:.2f} seconds.')

    return results


pw_path = '/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin'

df = pd.read_csv('/mnt/leif/littlab/users/slavelle/iEEG_Atlas/Tables/sz_table.csv')
channel_names_df = pd.read_csv('../data/HUP_files.csv')
ieeg_filename_df = pd.read_csv('/mnt/leif/littlab/users/slavelle/iEEG_Atlas/Tables/Master_Table_EIs/Manual_validation_seizures.csv')
merged_df = pd.merge(df, channel_names_df, left_on='hupid', right_on='pt_id', how='right')
merged_df.drop(columns=["ictal_exists", "ictal_path", "interictal_exists", "interictal_path", "both_exists"])

all_results = process_all_patients(merged_df, pw_path, ieeg_filename_df)
results_df = pd.DataFrame([result for result in all_results if result is not None])
results_df.to_csv('../data/self_run/hfer_HUP_60s.csv', index=False)
print("All results saved.")


# MUSC Patients

# In[ ]:


def process_single_patient(row, session, pw_path, ieeg_filename_df):
    print(f"Row File: {row['File']}")
    # if (row['Onset time'] == 146857.0016):
    #     print(f"Skipping row MP0002_D02 146857.0016s due to stalling values")
    #     return None
    start_time = row['Onset time'] 
    print(f"start time: {row['Onset time']}")
    pt_id = row['File'] 
    dataset_name = pt_id
    print(dataset_name)
    if dataset_name == '':
        print(f"No matching dataset found for HUP ID {pt_id}, {start_time}")
        return
    
    bl_start = (start_time - 100)
    bl_end = (start_time - 40)
    target_end = (start_time + 60)

    start_usec = bl_start*1e6
    stop_usec = target_end*1e6
    try:
        df, fs = get_iEEG_data(
            username='aguilac',
            password_bin_file=pw_path,
            iEEG_filename=dataset_name,
            start_time_usec=start_usec,
            stop_time_usec=stop_usec
        )

        # Apply notch filters at 60 Hz and harmonics
        data = notch_filter(df.to_numpy(), fs) # Change to Bandstop or remove the Phase and Amplitude of 60z & Harms

        if fs > 500:
            print(f'df before downsample = {df}')
            # Downsample
            # Target sampling frequency
            target_fs = 500
            fs = int(fs)
            # Calculate the resampling factors
            gcd = np.gcd(fs, target_fs)
            up = target_fs // gcd
            down = fs // gcd
            # Resample each channel using polynomial interpolation
            downsampled_data = signal.resample_poly(data, up, down, axis=0)
            # Convert the downsampled numpy array back to a DataFrame
            df_downsampled = pd.DataFrame(downsampled_data, columns=df.columns)
            print(f'df after downsample = {df_downsampled}')
            # New sampling frequency
            fs = target_fs
            data = df_downsampled

        try:
            channel_names = df.columns.tolist()
            print(f'CHANNEL NAMES ORIGINAL: {channel_names}')
            data = df.to_numpy()
            print(f'Original data shape: {data.shape}')
            channel_types = check_channel_types(channel_names)
            exclude_types = {'eeg', 'ecg', 'misc'}
            valid_channels = [ch for ctype, channels in channel_types.items() if ctype not in exclude_types for ch in channels]

            cleaned_channel_names = [decompose_labels(name) for name in channel_names]
            cleaned_valid_channel_names = [decompose_labels(name) for name in valid_channels]

            print("Length of cleaned_channel_names:", len(cleaned_channel_names))
            print("Length of valid_channel_names_clean:", len(cleaned_valid_channel_names))

            print("\nTypes in cleaned_channel_names:")
            for item in cleaned_channel_names:
                print(type(item), item)

            print("\nTypes in valid_channel_names_clean:")
            for item in cleaned_valid_channel_names:
                print(type(item), item)

            cleaned_channel_names = [str(item).strip().upper() for item in cleaned_channel_names]
            cleaned_valid_channel_names = [str(item).strip().upper() for item in cleaned_valid_channel_names]

            clean_channel_indices = [i for i, ch in enumerate(cleaned_channel_names) if ch in cleaned_valid_channel_names]

            print(f"Cleaned channel indices = {clean_channel_indices}")
            data = data[:, clean_channel_indices]

            print(f"Data after removing types: {data.shape}")
            artifacts = detect_artifacts(data, fs)
            clean_data, updated_channel_names, removed_channels, interpolated_channels = remove_or_interpolate(data, artifacts, [channel_names[i] for i in clean_channel_indices])
        except Exception as e:
            print(f"An error occurred in artifact removal: {e}")
            
        print(f"Data and sampling rate for {row['File']} acquired, clean_data shape is {clean_data.shape}")
        print(f'clean data: {clean_data}')
        try:
            bl_start_samples = int(bl_start*fs)
            bl_end_samples = int(bl_end*fs)
            target_end_samples = int(target_end*fs)
            print(f'bl start samples = {bl_start_samples}, bl end samples = {bl_end_samples}, target end samples = {target_end_samples}')
            print((bl_end_samples - bl_start_samples), (bl_end_samples - bl_start_samples), (target_end_samples- bl_start_samples))

            ei = get_ei_from_data(clean_data, fs, [0, bl_end_samples - bl_start_samples], [bl_end_samples - bl_start_samples, target_end_samples- bl_start_samples])
            print(f"EI calculated for {row['File']}")
            return {
                'MUSC_ID': row['File'],
                'ictal_start_time': start_time,
                'EI': ei,
                'name': updated_channel_names,
                'removed_channels': removed_channels
            }
        except Exception as e:
            print(f"An error occurred in EI calculation: {e}")
    except Exception as e:
        print(f"Error processing patient {row['File']}: {e}")


# In[10]:


pw_path = '/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin'

df = pd.read_csv('../data/MUSC_seizure_times.csv')
channel_names_df = pd.read_csv('../data/MUSC_files.csv')
ieeg_filename_df = pd.read_csv('../data/MUSC_seizure_times.csv')

df = df.dropna()
ieeg_filename_df = ieeg_filename_df.dropna()

all_results = process_all_patients(df, pw_path, ieeg_filename_df)
results_df = pd.DataFrame([result for result in all_results if result is not None])
results_df.to_csv('../data/self_run/hfer_MUSC_60s.csv', index=False)
print("All results saved.")
