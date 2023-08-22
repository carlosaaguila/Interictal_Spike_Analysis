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

#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

# %% import dataframe for leading spikes
lead_spikes_pt1 = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/feasbility tests/HUP_210_spikes_for_Carlos.csv')

#%% load the spike atlas
pt_name = 'HUP210'
spike, atlas, _,_ = load_ptall(pt_name, data_directory)

#%% find spikes that are related to the atlas
#clean labels
def decompose_labels(chLabel, name):
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

    # Get the non-numerical portion
    label_num_idx = re.search(r'\d', label_str)
    label_non_num = label_str[:label_num_idx.start()]
    elec = label_non_num

    # Get numerical portion
    label_num = re.search(r'\d+', label_str[label_num_idx.start():])
    label_num = int(label_num.group()) if label_num else float('nan')
    number = label_num

    if 'Fp1' in label_str.lower():
        clean_label = 'Fp1'

    if 'Fp2' in label_str.lower():
        clean_label = 'Fp2'

    return clean_label#, elec, number

#clean the labels in the spike dataframe
lead_spikes_pt1['channel_label'] = lead_spikes_pt1['channel_label'].apply(lambda x: decompose_labels(x, pt_name))

#%% check to see how many clean spikes that are true in is_leader, are in the atlas
#find the spikes that are true in is_leader
lead_spikes_leaders = lead_spikes_pt1[lead_spikes_pt1['is_leader'] == True]
#find the spikes that are in the atlas
in_atlas = lead_spikes_leaders[lead_spikes_leaders['channel_label'].isin(atlas['key_0'])]
#find the spikes that are not in the atlas
no_atlas = lead_spikes_leaders[~lead_spikes_leaders['channel_label'].isin(atlas['key_0'])]

#%% merge the atlas and the spike dataframe
#merge the atlas and the spike dataframe
atlas_spikes = pd.merge(in_atlas, atlas[['key_0','final_label']], left_on = 'channel_label', right_on = 'key_0', how = 'left')

#%% find the variance across
#looks like for HUP201, there is 101k spikes that are "leaders"
#a lot of leaders are in the SOZ. The split is 73k/27k for SOZ/NSOZ. 
#


#%% link the spikes to the atlas

