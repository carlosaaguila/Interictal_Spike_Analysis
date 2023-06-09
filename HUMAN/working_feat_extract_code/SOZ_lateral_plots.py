#%% set up environment
import pickle
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

# %% Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

# %% region of interest lists
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

#%% create lists with the patients of interest.
clinic_soz_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/clinic_soz_df.csv')
clinic_soz_df = clinic_soz_df.drop(columns = 'Unnamed: 0')
mesialtemp = clinic_soz_df[clinic_soz_df['region'] == 'mesial temporal']
neocortemp = clinic_soz_df[clinic_soz_df['region'] == 'temporal neocortical']
othercor = clinic_soz_df[clinic_soz_df['region'] == 'other cortex']
L_mesi_pts = mesialtemp['pt'][mesialtemp['lateralization'] == 'left'].to_list()
R_mesi_pts = mesialtemp['pt'][mesialtemp['lateralization'] == 'right'].to_list()
L_neo_pts = neocortemp['pt'][neocortemp['lateralization'] == 'left'].to_list()
R_neo_pts = neocortemp['pt'][neocortemp['lateralization'] == 'right'].to_list()
BI_mesi_pts = mesialtemp['pt'][mesialtemp['lateralization'] == 'bilateral'].to_list()

#%% Get the average waveforms - both ipsi and contra of the SOZ

# Left Mesial Temporal Patients
# IPSI
LEFTSIDE = roiL_mesial+roiL_lateral+L_OC
RIGHTSIDE = roiR_mesial+roiR_lateral+R_OC

#%% L_mesi_pts
ptnames = L_mesi_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, LEFTSIDE, title='Ipsi SOZ - Left Mesial Temporal', ymin=-500, ymax=800)
save = 'IPSI_L_mesialpt2'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))
# CONTRA
ptnames = L_mesi_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, RIGHTSIDE, title='Contra SOZ - Left Mesial Temporal', ymin=-500, ymax=800)
save = 'CONTRA_L_mesialpt2'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))

#%% Right Mesial Temporal Patients 
# IPSI
ptnames = R_mesi_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, RIGHTSIDE, title='Ipsi SOZ - Right Mesial Temporal', ymin=-500, ymax=800)
save = 'IPSI_R_mesialpt2'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))
# CONTRA
ptnames = R_mesi_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, LEFTSIDE, title='Contra SOZ - Right Mesial Temporal', ymin=-500, ymax=800)
save = 'CONTRA_R_mesialpt2'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))

#%% BIlateral mesial temporal patients
ptnames = BI_mesi_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, RIGHTSIDE, title='RIGHT - Bilateral Mesial Temporal', ymin=-500, ymax=800)
save = 'right_BI_mesialpt'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))
# CONTRA
ptnames = BI_mesi_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, LEFTSIDE, title='LEFT - Bilateral Mesial Temporal', ymin=-500, ymax=800)
save = 'left_BI_mesialpt'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))


#%% Left Neocortical Temporal Patients
# IPSI
ptnames = L_neo_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roiL_mesial, title='Ipsi SOZ - Left Neocortical Temporal', ymin=-500, ymax=800)
save = 'IPSI_L_neopt'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))
# CONTRA
ptnames = L_neo_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roiR_mesial, title='Contra SOZ - Left Neocortical Temporal', ymin=-500, ymax=800)
save = 'CONTRA_L_neopt'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))

# RIGHT Neocortical Temporal Patients
# IPSI
ptnames = R_neo_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roiR_mesial, title='Ipsi SOZ - Right Neocortical Temporal', ymin=-500, ymax=800)
save = 'IPSI_R_neopt'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))
# CONTRA
ptnames = R_neo_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roiL_mesial, title='Contra SOZ - Right Neocortical Temporal', ymin=-500, ymax=800)
save = 'CONTRA_R_neopt'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/norm/{}_norm".format(save)) #save as jpg
fig_abs.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/abs/{}_abs".format(save)) #save as jpg
fig_flip.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/ipsi_contra_soz_plts/flip/{}_flip".format(save))


#%% SOZ lateralization code

#right sided pts
right_pts = clinic_soz_df['pt'][clinic_soz_df['lateralization'] == 'right'].to_list()
ptnames = right_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, LEFTSIDE, title='Contra Avg. Spike for Right-sided Epilepsy Pts.', ymin=-500, ymax=800)
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, RIGHTSIDE, title='Ipsi Avg. Spike for Right-sided Epilepsy Pts.', ymin=-500, ymax=800)


#left sided pts 
left_pts = clinic_soz_df['pt'][clinic_soz_df['lateralization'] == 'left'].to_list()
ptnames = left_pts
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, LEFTSIDE, title='Ipsi Avg. Spike for Left-sided Epilepsy Pts.', ymin=-500, ymax=800)
fig, fig_abs, fig_flip, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, RIGHTSIDE, title='Contra Avg. Spike for Left-sided Epilepsy Pts.', ymin=-500, ymax=800)


# %% localization ROI code MESIAL 


ptnames = mesialtemp['pt'].to_list()

roi = roiL_mesial+roiR_mesial

fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roi, title = 'Avg. Mesial Temporal Spikes - MT patients', ymin=-500, ymax=800)

save = 'mt_mesial'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/norm/{}_norm".format(save)) #save as jpg
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/abs/{}abs".format(save)) #save as jpg
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/flip/{}_flip".format(save)) #save as jpg

roi2 = roiL_lateral + roiR_lateral

fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roi2, title = 'Avg. Lateral Spikes - MT patients', ymin=-500, ymax=800)

save = 'mt_lateral'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/norm/{}_norm".format(save)) #save as jpg
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/abs/{}abs".format(save)) #save as jpg
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/flip/{}_flip".format(save)) #save as jpg

roi3 = L_OC + R_OC

fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roi3, title = 'Avg. Other Spikes - MT patients', ymin=-500, ymax=800)

save = 'mt_other'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/norm/{}_norm".format(save)) #save as jpg
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/abs/{}abs".format(save)) #save as jpg
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/flip/{}_flip".format(save)) #save as jpg


#%% Localization ROI CODE LATERAL


ptnames = neocortemp['pt'].to_list()

roi = roiL_mesial+roiR_mesial

fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roi, title = 'Avg. Mesial Temporal Spikes - Neo patients', ymin=-500, ymax=800)

save = 'neo_mesial'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/norm/{}_norm".format(save)) #save as jpg
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/abs/{}abs".format(save)) #save as jpg
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/flip/{}_flip".format(save)) #save as jpg

roi2 = roiL_lateral + roiR_lateral

fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roi2, title = 'Avg. Lateral Spikes - Neo patients', ymin=-500, ymax=800)

save = 'neo_lateral'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/norm/{}_norm".format(save)) #save as jpg
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/abs/{}abs".format(save)) #save as jpg
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/flip/{}_flip".format(save)) #save as jpg

roi3 = L_OC + R_OC

fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs = plot_avgROIwave_multipt(ptnames, data_directory, roi3, title = 'Avg. Other Spikes - Neo patients', ymin=-500, ymax=800)

save = 'neo_other'
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/norm/{}_norm".format(save)) #save as jpg
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/abs/{}abs".format(save)) #save as jpg
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/localization_waves/Lateral/flip/{}_flip".format(save)) #save as jpg


#%% Localization ROI CODE OTHER?