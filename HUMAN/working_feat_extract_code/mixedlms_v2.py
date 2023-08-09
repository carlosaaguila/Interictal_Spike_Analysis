#%% 
#set up environment
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import warnings
#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

#establish ROI's
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
emptylabel = ['EmptyLabel','NaN']
L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']

roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]

# %% load all tables w/ new features

# load the tables V1
df_riseamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/riseamp.csv")
df_decayamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/decayamp.csv")
df_slowwidth = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/slowwidth.csv")
df_slowamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/slowamp.csv")
df_riseslope = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/riseslope.csv")
df_decayslope = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/decayslope.csv")
df_averageamp = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/averageamp.csv")
df_LL = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/LL.csv")

#load the saved tables V2

df_riseamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/riseamp_v2.csv")
df_decayamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/decayamp_v2.csv")
df_slowwidth_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/slowwidth_v2.csv")
df_slowamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/slowamp_v2.csv")
df_riseslope_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/riseslope_v2.csv")
df_decayslope_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/decayslope_v2.csv")
df_averageamp_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/averageamp_v2.csv")
df_LL_v2 = pd.read_csv("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/LL_v2.csv")

#%%

