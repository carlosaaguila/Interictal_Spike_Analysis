#%% Environment
import pickle as pkl
import numpy as np
import pandas as pd
import sys, os
code_path = os.path.dirname('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_functions import *

# %%
#where data is located
data_directory = '/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Patient/pt_database'

#load pts we are doing
pts = pd.read_csv('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Patient/pt_database/pt_data/mat_to_pkl_list.csv')
pts_list = pts['pt'].to_list()

#loop to create spikes object and pickle save
for pt in pts_list:
    print(pt)
    spike = load_pt(pt, data_directory) # spike contains values, randi_list, fs, chlabels, and SOZ
    with open(data_directory + '/pickle_spike/{}_obj.pkl'.format(pt), 'wb') as outp:
        pkl.dump(spike, outp, pkl.HIGHEST_PROTOCOL)