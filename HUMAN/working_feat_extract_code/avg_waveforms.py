#set up environment
import pickle
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings
warnings.filterwarnings('ignore')
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#setup
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)
ptnames = ['HUP105','HUP107','HUP111']

#Left Mesial Temporal
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
print('Starting L Mesial')
fig, fig_abs, total_avg, abs_total_avg, all_chs = plot_avgROIwave_multipt(ptnames, data_directory, roiL_mesial, title='Left Mesial Temporal')
save = 'L_mesial'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/abs/{}_abs".format(save)) #save as jpg

#left Laterial Temporal
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform ']
print('Starting L Lateral')
fig, fig_abs, total_avg, abs_total_avg, all_chs = plot_avgROIwave_multipt(ptnames, data_directory, roiL_lateral, title='Left Lateral Temporal')

save = 'L_lateral'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/abs/{}_abs".format(save)) #save as jpg

#Right Mesial
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
print('Starting R Mesial')
fig, fig_abs, total_avg, abs_total_avg, all_chs = plot_avgROIwave_multipt(ptnames, data_directory, roiR_mesial, title='Right Mesial Temporal')

save = 'R_mesial'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/abs/{}_abs".format(save)) #save as jpg

#Right Lateral
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
print('Starting R Lateral')
fig, fig_abs, total_avg, abs_total_avg, all_chs = plot_avgROIwave_multipt(ptnames, data_directory, roiR_lateral, title='Right Lateral Temporal')

save = 'R_lateral'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/avg_waveforms_output/abs/{}_abs".format(save)) #save as jpg



