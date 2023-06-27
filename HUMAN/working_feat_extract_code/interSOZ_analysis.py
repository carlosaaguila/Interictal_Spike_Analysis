#%%
#set up environment
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


#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

#all the ROIs of interest
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
emptylabel = ['EmptyLabel','NaN']
L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']
roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]


#%% 
def value_basis_interSOZ(ptname, data_directory):
    """
    input: ptname, data_directory
    output: all_vals - a list containing [SOZ, non-SOZ] values for a single patient
            all_idx_roich - a list containing [SOZ, non-SOZ] channel indices for a single patient
            all_chs - a list containing [SOZ, non-SOZ] randomly selected channels, matlab-indexed, for a single patient
            all_select_oi - a list containing [SOZ, non-SOZ] indices for a single patient
    explanation: In this function we are looking to compare SOZ electrodes in a specific region vs. non-SOZ electrodes in the same specific region.
                 example - "middle temporal gyrus" SOZ electrodes vs. "middle temporal gyrus" non-SOZ electrodes
    """
    #load patient
    spike, brain_df, onsetzone, ids = load_ptall(ptname, data_directory)

    #stops the function if there is "no image"
    if isinstance(brain_df, pd.DataFrame) == False:
        all_vals = None
        all_idx_roich = None
        all_chs = None
        all_select_oi = None
        print('no image -- skip')
        return all_vals, all_idx_roich, all_chs, all_select_oi
    
    #load master file with all elecs + SOZ's
    master_elecs = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/master_elecs.csv')
    #clean up
    pt_slice_df = master_elecs[master_elecs['rid'] == ids[1]].reset_index().drop(columns = 'index')
    #merge brain_df and the master file slice to get a clean DF.
    merge_finallabel = brain_df.merge(pt_slice_df[['vox_x','vox_y','vox_z','label','soz','resected','spike_rate','engel']], left_on = ['x','y','z'], right_on = ['vox_x','vox_y','vox_z'],  how = "inner")
    merge_finallabel = merge_finallabel[['key_0','name','x','y','z','final_label','soz','resected','spike_rate','engel']]

    #find the labels in which the SOZ is located on
    SOZrows = merge_finallabel[(merge_finallabel['soz'] == True) & (merge_finallabel['final_label'] != 'EmptyLabel')]
    SOZlabels = SOZrows['final_label'].unique()

    nonSOZrows= pd.DataFrame()
    for label in SOZlabels:
        rows = merge_finallabel[(merge_finallabel['soz'] == False) & (merge_finallabel['final_label'] == label)]
        nonSOZrows = nonSOZrows.append(rows)

    if SOZrows.empty | nonSOZrows.empty:
        all_vals = None
        all_idx_roich = None
        all_chs = None
        all_select_oi = None
        print('no SOZ or non-SOZ electrodes in this patient')
        return all_vals, all_idx_roich, all_chs, all_select_oi
    
    #get the names of the electrodes of interest
    SOZnames = np.array(SOZrows['name'])
    nonSOZnames = np.array(nonSOZrows['name'])
    
    all_vals = []
    all_idx_roich = []
    all_chs = []
    all_select_oi = []

    for roi, roi_chlist in enumerate([SOZnames, nonSOZnames]):

        idx_roich = []
        for i in range(len(spike.chlabels[0])):
            idx_holder = []
            for ch in roi_chlist:
                x = np.where(spike.chlabels[0][i] == ch)[0]
                idx_holder.append(x)
            idx_holder = [x for x in idx_holder for x in x]
            idx_roich.append(idx_holder)

        counts,chs = hifreq_ch_spike(spike.select)

        select_oi = []

        for i, list_of_idx in enumerate(idx_roich):
            if chs[i]-1 in list_of_idx:
                select_oi.append(i)

        values_oi = []
        if np.size(select_oi) == 0:
            print("NO MATCHES in roi {}".format(roi))
            all_vals = None
            all_idx_roich = None
            all_chs = None
            all_select_oi = None
            return all_vals, all_idx_roich, all_chs, all_select_oi

        else:
            for soi in select_oi:
                y = spike.values[soi]
                if len(y) > 2001:
                    spike_down = downsample_to_2001(y)
                else:
                    spike_down = y
                values_oi.append(spike_down)

        based_values = values_oi

        all_vals.append(based_values)
        all_chs.append(chs)
        all_idx_roich.append(idx_roich)
        all_select_oi.append(select_oi)

    return all_vals, all_idx_roich, all_chs, all_select_oi

def value_basis_interSOZ_v2(ptname, data_directory):
    """
    NEED TO UPDATE

    input: ptname, data_directory
    output: all_vals - a list containing [SOZ, non-SOZ] values for a single patient
            all_idx_roich - a list containing [SOZ, non-SOZ] channel indices for a single patient
            all_chs - a list containing [SOZ, non-SOZ] randomly selected channels, matlab-indexed, for a single patient
            all_select_oi - a list containing [SOZ, non-SOZ] indices for a single patient
    explanation: In this function we are looking to compare SOZ electrodes in a region vs. non-SOZ electrodes in the same region.
                 example - "middle temporal gyrus" SOZ electrodes vs. Left Neocortical non-SOZ electrodes
    """
    #load patient
    spike, brain_df, onsetzone, ids = load_ptall(ptname, data_directory)

    #stops the function if there is "no image"
    if isinstance(brain_df, pd.DataFrame) == False:
        all_vals = None
        all_idx_roich = None
        all_chs = None
        all_select_oi = None
        print('no image -- skip')
        return all_vals, all_idx_roich, all_chs, all_select_oi
    
    #load master file with all elecs + SOZ's
    master_elecs = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/master_elecs.csv')
    #clean up
    pt_slice_df = master_elecs[master_elecs['rid'] == ids[1]].reset_index().drop(columns = 'index')
    #merge brain_df and the master file slice to get a clean DF.
    merge_finallabel = brain_df.merge(pt_slice_df[['vox_x','vox_y','vox_z','label','soz','resected','spike_rate','engel']], left_on = ['x','y','z'], right_on = ['vox_x','vox_y','vox_z'],  how = "inner")
    merge_finallabel = merge_finallabel[['key_0','name','x','y','z','final_label','soz','resected','spike_rate','engel']]

    #find the labels in which the SOZ is located on
    SOZrows = merge_finallabel[merge_finallabel['soz'] == True]
    SOZlabels = SOZrows['final_label'].unique()

    nonSOZrows= pd.DataFrame()
    for label in SOZlabels:
        if label in roiL_mesial:
            roi = roiL_mesial
        elif label in roiL_lateral:
            roi = roiL_lateral
        elif label in roiR_mesial:
            roi = roiR_mesial
        elif label in roiR_lateral:
            roi = roiR_lateral 
        elif label in L_OC:
            roi = L_OC
        elif label in R_OC:
            roi = R_OC
        else:
            roi = emptylabel
        
        for x in roi:
            rows = merge_finallabel[(merge_finallabel['soz'] == False) & (merge_finallabel['final_label'] == x)]
            nonSOZrows = nonSOZrows.append(rows)
    
    nonSOZrows = nonSOZrows.drop_duplicates()

    if SOZrows.empty | nonSOZrows.empty:
        all_vals = None
        all_idx_roich = None
        all_chs = None
        all_select_oi = None
        print('no SOZ or non-SOZ electrodes in this patient')
        return all_vals, all_idx_roich, all_chs, all_select_oi
        
    #get the names of the electrodes of interest
    SOZnames = np.array(SOZrows['name'])
    nonSOZnames = np.array(nonSOZrows['name'])
    
    all_vals = []
    all_idx_roich = []
    all_chs = []
    all_select_oi = []

    for roi, roi_chlist in enumerate([SOZnames, nonSOZnames]):

        idx_roich = []
        for i in range(len(spike.chlabels[0])):
            idx_holder = []
            for ch in roi_chlist:
                x = np.where(spike.chlabels[0][i] == ch)[0]
                idx_holder.append(x)
            idx_holder = [x for x in idx_holder for x in x]
            idx_roich.append(idx_holder)

        counts,chs = hifreq_ch_spike(spike.select)

        select_oi = []

        for i, list_of_idx in enumerate(idx_roich):
            if chs[i]-1 in list_of_idx:
                select_oi.append(i)

        values_oi = []
        if np.size(select_oi) == 0:
            values_oi = 0
            print("NO MATCHES in roi {}".format(roi))
        else:
            for soi in select_oi:
                y = spike.values[soi]
                if len(y) > 2001:
                    spike_down = downsample_to_2001(y)
                else:
                    spike_down = y
                values_oi.append(spike_down)

        based_values = values_oi

        all_vals.append(based_values)
        all_chs.append(chs)
        all_idx_roich.append(idx_roich)
        all_select_oi.append(select_oi)

    return all_vals, all_idx_roich, all_chs, all_select_oi

def totalavg_roiwave(values, chs, select_oi):
    """
    input: values, chs, select_oi
    output: avg_waveform, abs_avg_waveform, all_chs_stacked
    explanation: this function is to get the average waveform for a single ROI in a patient. Also, it returns all the spikes
                    in a single ROI in a patient.

    """
    perroi_mean = [] #the mean of the stacked spikewaves
    perroi = [] #2 element list with all the spikewaves stacked
    
    for j, roi in enumerate(values):
        spikewaves = []

        if not roi: 
            perroi.append(np.nan)
            perroi_mean.append(np.nan)
            continue

        for l, xs in enumerate(roi):
            val_want = np.transpose(xs)
            val_want = val_want[chs[j][select_oi[j][l]]-1]
            spikewaves.append(val_want)

        perroi.append(spikewaves)
        perroi_mean.append(np.nanmean(spikewaves, axis = 0))

    return perroi, perroi_mean

def run_interSOZ(ptnames, data_directory):
    n = 15
    lists_ptnames = (divide_chunks(ptnames, n))

    for Z, ptlist in enumerate(lists_ptnames):
        print('Running Patient List: {}'.format(Z))
        #RESET variables to not crash
        perpt_all = []
        perpt_mean = []
        SOZ_all_chs_stacked = []
        nonSOZ_all_chs_stacked = []
        SOZ_all_chs_stacked_DF = pd.DataFrame()
        nonSOZ_all_chs_stacked_DF = pd.DataFrame()
        SOZ_average_waveform_DF = pd.DataFrame()
        nonSOZ_average_waveform_DF = pd.DataFrame()
        id_df = pd.DataFrame()
        SOZ_avgs = []
        nonSOZ_avgs = []
        id = []

        #get vals and perroi values for each patient in a ptlist.
        for ptname in ptlist:
            print('Running Patient: {}'.format(ptname))

            #get the values for the SOZ and non-SOZ electrodes
            vals, idx_roich, chs, select_oi = value_basis_interSOZ(ptname, data_directory)
            #vals2, idx_roich2, chs2, select_oi2 = value_basis_interSOZ_v2(ptname, data_directory)

            #get the average waveform for the SOZ and non-SOZ electrodes
            if vals != None: 
                perroi, perroi_mean = totalavg_roiwave(vals, chs, select_oi)
                perpt_all.append(perroi)
                perpt_mean.append(perroi_mean)
                id.append([ptname, np.shape(perroi[0]), np.shape(perroi[1])])


        #add them to a dataframe
        for pt in perpt_all:
            SOZ_all_chs_stacked.append(pt[0])
            nonSOZ_all_chs_stacked.append(pt[1])

        for pt in perpt_mean:
            SOZ_avgs.append([pt[0]])
            nonSOZ_avgs.append([pt[1]])

        #flatten the list
        SOZ_all_chs_stacked = [x for x in SOZ_all_chs_stacked for x in x]
        nonSOZ_all_chs_stacked = [x for x in nonSOZ_all_chs_stacked for x in x]

        SOZ_avgs = [x for x in SOZ_avgs for x in x]
        nonSOZ_avgs = [x for x in nonSOZ_avgs for x in x]

        #append a new lines to dataframe
        SOZ_all_chs_stacked_DF = SOZ_all_chs_stacked_DF.append(SOZ_all_chs_stacked)
        nonSOZ_all_chs_stacked_DF = nonSOZ_all_chs_stacked_DF.append(nonSOZ_all_chs_stacked)
        SOZ_average_waveform_DF = SOZ_average_waveform_DF.append(SOZ_avgs)
        nonSOZ_average_waveform_DF = nonSOZ_average_waveform_DF.append(nonSOZ_avgs)
        id_df = id_df.append(id)


        #save updated dataframes
        if Z == 0:
            SOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/SOZ_all_chs_stacked_DF.csv', index = False)
            nonSOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nonSOZ_all_chs_stacked_DF.csv',index = False)
            SOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/SOZ_average_waveform_DF.csv',index = False)
            nonSOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nonSOZ_average_waveform_DF.csv',index = False)
            id_df.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/id_df.csv', index = False)
        if Z != 0:
            SOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/SOZ_all_chs_stacked_DF.csv', mode = 'a', index = False, header = False)
            nonSOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nonSOZ_all_chs_stacked_DF.csv', mode = 'a', index = False, header = False)
            SOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/SOZ_average_waveform_DF.csv', mode = 'a', index = False, header = False)
            nonSOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nonSOZ_average_waveform_DF.csv', mode = 'a', index = False, header = False)
            id_df.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/id_df.csv', mode = 'a', index = False, header = False)


    SOZ_all_chs_stacked_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/SOZ_all_chs_stacked_DF.csv')
    nonSOZ_all_chs_stacked_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nonSOZ_all_chs_stacked_DF.csv')
    SOZ_average_waveform_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/SOZ_average_waveform_DF.csv')
    nonSOZ_average_waveform_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/nonSOZ_average_waveform_DF.csv')
    id_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/id_df.csv')

    return SOZ_all_chs_stacked_DF, nonSOZ_all_chs_stacked_DF, SOZ_average_waveform_DF, nonSOZ_average_waveform_DF, id_df
# %%
SOZ_all_chs_stacked_DF, nonSOZ_all_chs_stacked_DF, SOZ_average_waveform_DF, nonSOZ_average_waveform_DF, id_df = run_interSOZ(ptnames, data_directory)