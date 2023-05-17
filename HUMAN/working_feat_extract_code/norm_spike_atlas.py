#%%
#set up environment
import pickle
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings

#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#%%

#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

# %%
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
emptylabel = ['EmptyLabel','NaN']
L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']

#L_OC = [' left inferior parietal ', ' left postcentral ',' @left cerebellum exterior ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' @left putamen ', ' left posterior cingulate ', ' @left caudate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
#R_OC = [' right inferior parietal ', ' right postcentral ',' right cerebellum exterior ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right putamen ', ' right posterior cingulate ', ' right caudate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']
roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]

#%% chop up the ptnames into lists of lists
n=3
lists_ptnames = divide_chunks(ptnames, n)
lists_ptnames = list(lists_ptnames)[0:3]

#%%
Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz = feat_extract(lists_ptnames, roilist, data_directory)

# %%
test_df = pd.DataFrame(data = totalcount_perpt)
soz_df  = pd.DataFrame(data = clinic_soz)
test_df = test_df.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df = test_df.drop(columns = 'Empty Label')
soz_df = soz_df.rename(columns = {0:'region', 1:'lateralization'})
amp_df_combine =  pd.concat([test_df, soz_df], axis = 1)

for l in to_remove: #remove to_remove variables
    amp_df_combine = amp_df_combine[amp_df_combine['region'] != l]

amp_df_drop1 = amp_df_combine
amp_df_drop1['soz'] = amp_df_combine['lateralization'] + " - " + amp_df_combine['region'] 

#WANT TO FIND THE PERCENTAGE BREAKDOWN PER PT BEFORE GROUPING. 
count_df = amp_df_drop1.drop(columns = ['region','lateralization'])
count_df['soz'] = count_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

count_percent_perpt = count_df[['L_Mesial', 'L_Lateral', 'R_Mesial', 'R_Lateral', 'L_OtherCortex', 'R_OtherCortex']].div(count_df.sum(axis=1), axis =0).mul(100)
count_percent_perpt['soz'] = count_df['soz']
count_percent_perpt = count_percent_perpt.groupby(by='soz').median()

display(count_percent_perpt)

# %%
