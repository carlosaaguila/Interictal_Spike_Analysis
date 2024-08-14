#%%
import pandas as pd
import numpy as np
# from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
drop_pts = ['HUP093','HUP108','HUP113','HUP114','HUP116','HUP123','HUP087','HUP099','HUP111','HUP121','HUP105','HUP106','HUP107','HUP159'] #These are the patients with less than 8 contacts.
#load in both spike dataframes for HUP
spikes_thresh = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
HUP_thresh = spikes_thresh[~spikes_thresh['pt_id'].isin(drop_pts)]


#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]
MUSC_sozs = MUSC_sozs.drop(columns=['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'])

#find the patients that should be null, and remove them for the full dataset
nonnan_mask = MUSC_sozs.dropna()
pts_to_remove = nonnan_mask[nonnan_mask['Correction Notes'].str.contains('null')]['ParticipantID'].array

## load the spike data
MUSC_spikes = pd.read_csv('../dataset/complete_dfs/MUSC_thresholded.csv', index_col=0)

#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]
MUSC_sozs = MUSC_sozs.drop(columns=['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'])

#fix SOZ and laterality
MUSC_spikes = MUSC_spikes.merge(MUSC_sozs, left_on = 'pt_id', right_on = 'ParticipantID', how = 'inner')
MUSC_spikes = MUSC_spikes.drop(columns=['ParticipantID','Site_1MUSC_2Emory','IfNeocortical_Location','Correction Notes','lateralization_left','lateralization_right','region'])

#remove the patients that should be NULL for the thresholded dataset
MUSC_spikes = MUSC_spikes[~MUSC_spikes['pt_id'].isin(pts_to_remove)]
MUSC_thresh = MUSC_spikes

def calculate_dp(df):
    total_spikes = len(df) - 1
    dp_values = 100 * (total_spikes - 2 * df.index) / total_spikes
    return dp_values

def designate(dp):
    if dp >= 20:
        return 'up'
    elif dp < -20:
        return 'down'
    else:
        return 'inter'

def process_patient_HUP(pt_id):
    print(pt_id)
    yo = HUP_thresh[HUP_thresh['filename'] == pt_id]
    yo = yo[['new_spike_seq', 'filename', 'pt_id', 'recruitment_latency_thresh', 'channel_label','peak_time_usec']]
    yo['peak_time_usec'] = yo['peak_time_usec']/6e7 # convert it into minutes
    yo = yo.rename(columns = {'peak_time_usec':'peak_time_min'})
    patient_df = pd.DataFrame()

    for spike_seq in yo['new_spike_seq'].unique():
        df = yo[yo['new_spike_seq'] == spike_seq]
        df_sorted = df.sort_values(by='recruitment_latency_thresh').reset_index(drop=True)
        df_sorted['DP'] = calculate_dp(df_sorted)
        df_sorted['designation'] = df_sorted['DP'].apply(designate)
        patient_df = pd.concat([patient_df, df_sorted], ignore_index=True)

    return patient_df

def process_patient_MUSC(pt_id):
    print(pt_id)
    yo = MUSC_thresh[MUSC_thresh['filename'] == pt_id]
    yo = yo[['new_spike_seq', 'filename', 'pt_id', 'recruitment_latency_thresh', 'channel_label','peak_time_usec']]
    yo['peak_time_usec'] = yo['peak_time_usec']/6e7 # convert it into minutes
    yo = yo.rename(columns = {'peak_time_usec':'peak_time_min'})
    patient_df = pd.DataFrame()

    for spike_seq in yo['new_spike_seq'].unique():
        df = yo[yo['new_spike_seq'] == spike_seq]
        df_sorted = df.sort_values(by='recruitment_latency_thresh').reset_index(drop=True)
        df_sorted['DP'] = calculate_dp(df_sorted)
        df_sorted['designation'] = df_sorted['DP'].apply(designate)
        patient_df = pd.concat([patient_df, df_sorted], ignore_index=True)

    return patient_df

# Parallel processing for HUP data
with ProcessPoolExecutor() as executor:
    HUP_clean = pd.concat(executor.map(process_patient_HUP, HUP_thresh['filename'].unique()), ignore_index=True)

HUP_clean.reset_index(drop=True, inplace=True)
HUP_clean.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/HUP_clean.csv')

# Parallel processing for MUSC data
with ProcessPoolExecutor() as executor:
    MUSC_clean = pd.concat(executor.map(process_patient_MUSC, MUSC_thresh['filename'].unique()), ignore_index=True)

MUSC_clean.reset_index(drop=True, inplace=True)
MUSC_clean.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/MUSC_clean.csv')

# %%
