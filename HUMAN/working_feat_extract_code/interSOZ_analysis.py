#%%
#set up environment
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
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
#pd.set_option('display.max_rows', None)


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

#%% get ROI's from akash's list:
mni_dkt = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/dkt_mni_v2.xlsx', sheet_name = 'Sheet1')
mni_dkt = mni_dkt[['Label 0: background', 'name']].dropna().reset_index(drop = True).rename(columns = {"Label 0: background":"DKT", "name":"MNI"})
mni_dkt["DKT"] = mni_dkt['DKT'].apply(lambda x: x.split(':')[-1] + ' ')
mni_labels = np.unique(mni_dkt['MNI'].to_numpy())
#get the DKT labels for each MNI labels
dkt_labels = []
for i in mni_labels:
    subdf = mni_dkt.loc[mni_dkt['MNI'] == i]
    dkt_labels.append(subdf['DKT'].to_list())
#pass dkt_labels as roilist if you want 40 ROIs, that correspond to the names in the mni_labels

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

def run_interSOZ(ptnames, data_directory, load = True):

    if load == True:
        print('loading data')
        SOZ_all_chs_stacked_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_all_chs_stacked_DF.csv')
        nonSOZ_all_chs_stacked_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_all_chs_stacked_DF.csv')
        SOZ_average_waveform_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_average_waveform_DF.csv')
        nonSOZ_average_waveform_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_average_waveform_DF.csv')
        id_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/id_df.csv')

    else:
        print('getting data')
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
                SOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_all_chs_stacked_DF.csv', index = False)
                nonSOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_all_chs_stacked_DF.csv',index = False)
                SOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_average_waveform_DF.csv',index = False)
                nonSOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_average_waveform_DF.csv',index = False)
                id_df.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/id_df.csv', index = False)
            if Z != 0:
                SOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_all_chs_stacked_DF.csv', mode = 'a', index = False, header = False)
                nonSOZ_all_chs_stacked_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_all_chs_stacked_DF.csv', mode = 'a', index = False, header = False)
                SOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_average_waveform_DF.csv', mode = 'a', index = False, header = False)
                nonSOZ_average_waveform_DF.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_average_waveform_DF.csv', mode = 'a', index = False, header = False)
                id_df.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/id_df.csv', mode = 'a', index = False, header = False)


        SOZ_all_chs_stacked_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_all_chs_stacked_DF.csv')
        nonSOZ_all_chs_stacked_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_all_chs_stacked_DF.csv')
        SOZ_average_waveform_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_average_waveform_DF.csv')
        nonSOZ_average_waveform_DF = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_average_waveform_DF.csv')
        id_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/id_df.csv')

    return SOZ_all_chs_stacked_DF, nonSOZ_all_chs_stacked_DF, SOZ_average_waveform_DF, nonSOZ_average_waveform_DF, id_df
# %%
SOZ_all_chs_stacked_DF, nonSOZ_all_chs_stacked_DF, SOZ_average_waveform_DF, nonSOZ_average_waveform_DF, id_df = run_interSOZ(ptnames, data_directory, load = True)

#%% function to fix id_df if it ever has to be remade
def fix_id_df(id_df):
    id_df.columns = ['id', '# SOZ', '# nonSOZ']
    #in the column # SOZ, get the number after first paranthesis and before the first comma
    id_df['# SOZ'] = id_df['# SOZ'].str.extract(r'\((.*?)\,').astype(int)
    #in the column # nonSOZ, get the number after first paranthesis and before the first comma
    id_df['# nonSOZ'] = id_df['# nonSOZ'].str.extract(r'\((.*?)\,').astype(int)

    id_df['cumsum SOZ'] = id_df['# SOZ'].cumsum()
    id_df['cumsum nonSOZ'] = id_df['# nonSOZ'].cumsum()
    return id_df

#%%
#find patients to remove:
id_df_cleaned = id_df[(id_df['# SOZ'] > 10) & (id_df['# nonSOZ'] > 10)]

#find the opposite of id_df_cleaned
id_df_removed = id_df[(id_df['# SOZ'] <= 10) | (id_df['# nonSOZ'] <= 10)]

# %% 
#function that removes rows from a dataframe based on a range of indices
def remove_rows_ALL(SOZ_df, nonSOZ_df, df_to_remove):
    """
    DF is a dataframe with all the values
    df_to_remove is a dataframe with the patient name, and sum of SOZ and nonSOZ spikes
    """
    num_SOZ = df_to_remove['# SOZ'].to_list()
    num_nonSOZ = df_to_remove['# nonSOZ'].to_list()
    cumsum_SOZ = df_to_remove['cumsum SOZ'].to_list()
    cumsum_nonSOZ = df_to_remove['cumsum nonSOZ'].to_list()

    #find the start indices for those to remove.
    startSOZ = [x - y for x, y in zip(cumsum_SOZ, num_SOZ)]
    startnonSOZ = [x - y for x, y in zip(cumsum_nonSOZ, num_nonSOZ)]
    #adjust for python indexing.
    #startSOZ = [0 if x <= 0 else x - 1 for x in startSOZ]
    #startnonSOZ = [0 if x <= 0 else x - 1 for x in startnonSOZ]

    #find the end indices for those to remove.
    endSOZ = cumsum_SOZ
    endnonSOZ = cumsum_nonSOZ

    #drop the rows from 
    DF1 = pd.DataFrame()
    for start, end in zip(startSOZ, endSOZ):
        DF1 = DF1.append(SOZ_df.iloc[start:end])

    DF2 = pd.DataFrame()
    for start, end in zip(startnonSOZ, endnonSOZ):
        DF2 = DF2.append(nonSOZ_df.iloc[start:end])
    
    DF1_idx_to_remove = DF1.index.to_list()
    DF2_idx_to_remove = DF2.index.to_list()

    SOZ_df_cleaned = SOZ_df.drop(index = DF1_idx_to_remove)
    nonSOZ_df_cleaned = nonSOZ_df.drop(index = DF2_idx_to_remove)

    return SOZ_df_cleaned, nonSOZ_df_cleaned

def remove_rows_avg(SOZ_df, nonSOZ_df, df_to_remove):
    """
    DF is a dataframe with all the values
    df_to_remove is a dataframe with the patient name, and sum of SOZ and nonSOZ spikes
    """
    index_to_remove = df_to_remove.index.to_list()
    SOZ_df_cleaned = SOZ_df.drop(index = index_to_remove)
    nonSOZ_df_cleaned = nonSOZ_df.drop(index = index_to_remove)

    return SOZ_df_cleaned, nonSOZ_df_cleaned

#%%
#cleaned dataframes (removed low counts)
SOZ_all_chs_stacked_DF_cleaned, nonSOZ_all_chs_stacked_DF_cleaned = remove_rows_ALL(SOZ_all_chs_stacked_DF,nonSOZ_all_chs_stacked_DF, id_df_removed)
SOZ_average_waveform_DF_cleaned, nonSOZ_average_waveform_DF_cleaned = remove_rows_avg(SOZ_average_waveform_DF, nonSOZ_average_waveform_DF, id_df_removed)

#%% functions to take index from cleaned ID, and calculate a feature for each patient
def get_feat_per_pt(SOZ_df, nonSOZ_df, id_df_cleaned):

    num_SOZ = id_df_cleaned['# SOZ'].to_list()
    num_nonSOZ = id_df_cleaned['# nonSOZ'].to_list()
    cumsum_SOZ = id_df_cleaned['cumsum SOZ'].to_list()
    cumsum_nonSOZ = id_df_cleaned['cumsum nonSOZ'].to_list()
    ids = id_df_cleaned['id'].to_list()

    #find the start indices for those to remove.
    startSOZ = [x - y for x, y in zip(cumsum_SOZ, num_SOZ)]
    startnonSOZ = [x - y for x, y in zip(cumsum_nonSOZ, num_nonSOZ)]

    #adjust for python indexing. (guess not needed)
    #startSOZ = [0 if x <= 0 else x - 1 for x in startSOZ]
    #startnonSOZ = [0 if x <= 0 else x - 1 for x in startnonSOZ]

    #find the end indices for those to remove.
    endSOZ = cumsum_SOZ
    endnonSOZ = cumsum_nonSOZ

    #drop the rows from 
    DF1 = pd.DataFrame()
    for start, end, id in zip(startSOZ, endSOZ, ids):
        sub_df = SOZ_df.iloc[start:end]
        sub_df['id'] = id
        DF1 = DF1.append(sub_df)


    DF2 = pd.DataFrame()
    for start, end, id in zip(startnonSOZ, endnonSOZ, ids):
        sub_df = nonSOZ_df.iloc[start:end]
        sub_df['id'] = id
        DF2 = DF2.append(sub_df)

    SOZ_feats = pd.DataFrame()
    nonSOZ_feats = pd.DataFrame()
    #SOZ_feats['amp'] = DF1.groupby('id').max(axis = 0)
    #nonSOZ_feats['amp'] = DF2.groupby('id').max(axis = 0)

    return DF1, DF2

soz_w_label, nonsoz_w_label = get_feat_per_pt(SOZ_all_chs_stacked_DF, nonSOZ_all_chs_stacked_DF, id_df_cleaned)
# %% prepare data for plotting

#time
time = np.linspace(0,4,2001)

#average cohort waveform
avgcohort_SOZ = SOZ_all_chs_stacked_DF_cleaned.mean(axis = 0).to_list()
avgcohort_nonSOZ = nonSOZ_all_chs_stacked_DF_cleaned.mean(axis = 0).to_list()

stdcohort_SOZ = SOZ_all_chs_stacked_DF_cleaned.std(axis = 0).to_list()
stdcohort_nonSOZ = nonSOZ_all_chs_stacked_DF_cleaned.std(axis = 0).to_list()

#get per_pt average waveforms
SOZ_avgs = SOZ_average_waveform_DF_cleaned.T
nonSOZ_avgs = nonSOZ_average_waveform_DF_cleaned.T

# append all the values of a column into a list
SOZ_avgs_list = []
nonSOZ_avgs_list = []
for col in SOZ_avgs.columns:
    SOZ_avgs_list.append(SOZ_avgs[col].to_list())
for col in nonSOZ_avgs.columns:
    nonSOZ_avgs_list.append(nonSOZ_avgs[col].to_list())


# %% PLOT

fig, ax = plt.subplots(1,2,figsize = (15,7))

for SOZ, nonSOZ in zip(SOZ_avgs_list, nonSOZ_avgs_list):
    ax[0].plot(time, SOZ, color = 'r', alpha = 0.3, linewidth=0.7)
    ax[1].plot(time, nonSOZ, color = 'r', alpha = 0.3, linewidth=0.7)

ax[0].plot(time, avgcohort_SOZ, color = 'k', linewidth=2)
ax[1].plot(time, avgcohort_nonSOZ, color = 'k', linewidth=2)

ax[0].fill_between(time, np.add(avgcohort_SOZ, stdcohort_SOZ), np.subtract(avgcohort_SOZ, stdcohort_SOZ), color = 'k', linewidth=0.5, alpha = 0.3, linestyle = '--')

ax[1].fill_between(time, np.add(avgcohort_nonSOZ, stdcohort_nonSOZ),np.subtract(avgcohort_nonSOZ, stdcohort_nonSOZ), color = 'k', linewidth=0.5, alpha = 0.3, linestyle = '--')

ax[0].set_title('SOZ')
ax[1].set_title('nonSOZ')
ax[0].set_xlabel('time (s)')
ax[1].set_xlabel('time (s)')
ax[0].set_ylabel('Amplitude (mV)')
ax[1].set_ylabel('Amplitude (mV)')
ax[0].set_ylim(-900,700)
ax[1].set_ylim(-900,700)

# %%
#find the max amplitude across each row (only using the middle quartile of columns)
SOZ_feats = pd.DataFrame()
nonSOZ_feats = pd.DataFrame()

#subtract mean_SOZ_amp from each row
demeaned_SOZ = SOZ_all_chs_stacked_DF_cleaned.sub(SOZ_all_chs_stacked_DF_cleaned.mean(axis = 1), axis = 0)
demeaned_nonSOZ = nonSOZ_all_chs_stacked_DF_cleaned.sub(nonSOZ_all_chs_stacked_DF_cleaned.mean(axis = 1), axis = 0)

#find the absolute amplitude
SOZ_feats['abs amp'] = demeaned_SOZ.iloc[:, int(2001/2 -200):int(2001/2 + 200)].abs().max(axis = 1)
nonSOZ_feats['abs amp'] = demeaned_nonSOZ.iloc[:, int(2001/2 -200):int(2001/2 + 200)].abs().max(axis = 1)

#find the linelength of the demeaned singal
SOZ_feats['linelength'] = demeaned_SOZ.iloc[:, int(2001/2 -200):int(2001/2 + 200)].apply(LL, axis = 1)
nonSOZ_feats['linelength'] = demeaned_nonSOZ.iloc[:, int(2001/2 -200):int(2001/2 + 200)].apply(LL, axis = 1)


#%% use soz_w_labels and nonsoz_w_labels to merge id's with max and min amplitudes
SOZ_feats['id'] = soz_w_label['id']
nonSOZ_feats['id'] = nonsoz_w_label['id']

#%% median values for each patient feat

SOZ_median_feats= SOZ_feats.groupby('id').median()
SOZ_median_feats = SOZ_median_feats.rename(columns = {'abs amp': 'SOZ abs amp', 'linelength': 'SOZ LL'})

nonSOZ_median_feats = nonSOZ_feats.groupby('id').median()
nonSOZ_median_feats = nonSOZ_median_feats.rename(columns = {'abs amp': 'nonSOZ abs amp', 'linelength': 'nonSOZ LL'})

median_feats = pd.concat([SOZ_median_feats, nonSOZ_median_feats], axis = 1)
median_feats = median_feats.dropna()

median_feats['color_amp'] = median_feats['SOZ abs amp'] - median_feats['nonSOZ abs amp']
median_feats['color_LL'] = median_feats['SOZ LL'] - median_feats['nonSOZ LL']

median_feats['color_amp'] = median_feats['color_amp'].apply(lambda x: True if x > 0 else False).astype(int)
median_feats['color_LL'] = median_feats['color_LL'].apply(lambda x: True if x > 0 else False).astype(int)

#%% 
#create a paired plot of SOZ abs amp vs. nonSOZ abs amp
fig, ax = plt.subplots(1,1, figsize = (7,7))
ax.scatter(median_feats[median_feats['color_amp'] == 1]['SOZ abs amp'], median_feats[median_feats['color_amp'] == 1]['nonSOZ abs amp'], color = 'r', label = 'Patients w/ SOZ > ({})'.format(len(median_feats[median_feats['color_amp'] == 1])))
ax.scatter(median_feats[median_feats['color_amp'] == 0]['SOZ abs amp'], median_feats[median_feats['color_amp'] == 0]['nonSOZ abs amp'], color = 'b', label = 'Patients w/ nonSOZ > ({})'.format(len(median_feats[median_feats['color_amp'] == 0])))
ax.set_xlabel('SOZ abs amp')
ax.set_ylabel('nonSOZ abs amp')
ax.set_title('SOZ vs. nonSOZ abs amp')
ax.plot(np.linspace(0,1200,100), np.linspace(0,1200,100), color = 'k', linestyle = '--')
ax.set_xlim(0,1200)
ax.set_ylim(0,1200)
ax.legend()

#create a paired plot of SOZ LL vs. nonSOZ LL
fig, ax = plt.subplots(1,1, figsize = (7,7))
ax.scatter(median_feats[median_feats['color_LL'] == 1]['SOZ LL'], median_feats[median_feats['color_LL'] == 1]['nonSOZ LL'], color = 'r', label = "Patients w/ SOZ > ({})".format(len(median_feats[median_feats['color_LL'] == 1])))
ax.scatter(median_feats[median_feats['color_LL'] == 0]['SOZ LL'], median_feats[median_feats['color_LL'] == 0]['nonSOZ LL'], color = 'b', label = "Patients w/ nonSOZ > ({})".format(len(median_feats[median_feats['color_LL'] == 0])))
ax.set_xlabel('SOZ LL')
ax.set_ylabel('nonSOZ LL')
ax.set_title('SOZ vs. nonSOZ LL')
ax.plot(np.linspace(0,6500,100), np.linspace(0,6500,100), color = 'k', linestyle = '--')
ax.set_xlim(0,6500)
ax.set_ylim(0,6500)
ax.legend()

# %%
SOZ_feats_new = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_feats.csv', index_col = 0)
nonSOZ_feats_new = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/nonSOZ_feats.csv', index_col = 0)
# %%
SOZ_median_feats = SOZ_feats_new.groupby('id').median()
nonSOZ_median_feats = nonSOZ_feats_new.groupby('id').median()

SOZ_median_feats = SOZ_median_feats.rename(columns = {'rise_amp': 'SOZ rise amp', 'decay_amp': 'SOZ decay amp', 'slow_width':'SOZ slow width', 'slow_amp': 'SOZ slow amp', 'rise_slope': 'SOZ rise slope', 'decay_slope': 'SOZ decay slope', 'linelen': 'SOZ LL', 'average_amp': 'SOZ avg amp'})
nonSOZ_median_feats = nonSOZ_median_feats.rename(columns = {'rise_amp': 'nonSOZ rise amp', 'decay_amp': 'nonSOZ decay amp', 'slow_width':'nonSOZ slow width', 'slow_amp': 'nonSOZ slow amp', 'rise_slope': 'nonSOZ rise slope', 'decay_slope': 'nonSOZ decay slope', 'linelen': 'nonSOZ LL', 'average_amp': 'nonSOZ avg amp'})

median_feats = pd.concat([SOZ_median_feats, nonSOZ_median_feats], axis = 1)

#%%
SOZ_columns = SOZ_median_feats.columns.to_list()
nonSOZ_columns = nonSOZ_median_feats.columns.to_list()
newcolumns = ['color_riseamp', 'color_decayamp', 'color_slowwidth', 'color_slowamp', 'color_riseslope', 'color_decayslope', 'color_avgamp', 'color_LL']

for i, (SOZ, nonSOZ) in enumerate(zip(SOZ_columns, nonSOZ_columns)):
    median_feats[newcolumns[i]] = median_feats[SOZ] - median_feats[nonSOZ]
    median_feats[newcolumns[i]] = median_feats[newcolumns[i]].apply(lambda x: True if x > 0 else False).astype(int)

# %%
#create paired plots
title = ['Rise Amp', 'Decay Amp', 'Slow Width', 'Slow Amp', 'Rise Slope', 'Decay Slope', 'Avg Amp', 'LL']
for i in range(len(newcolumns)):
    fig, ax = plt.subplots(1,1, figsize = (7,7))
    ax.scatter(median_feats[median_feats[newcolumns[i]] == 1][SOZ_columns[i]], median_feats[median_feats[newcolumns[i]] == 1][nonSOZ_columns[i]], color = 'r', label = "Patients w/ SOZ > ({})".format(len(median_feats[median_feats[newcolumns[i]] == 1])))
    ax.scatter(median_feats[median_feats[newcolumns[i]] == 0][SOZ_columns[i]], median_feats[median_feats[newcolumns[i]] == 0][nonSOZ_columns[i]], color = 'b', label = "Patients w/ nonSOZ > ({})".format(len(median_feats[median_feats[newcolumns[i]] == 0])))
    ax.set_xlabel(SOZ_columns[i])
    ax.set_ylabel(nonSOZ_columns[i])
    ax.set_title('SOZ vs. nonSOZ {}'.format(title[i]))
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()

# %% normalcy test
for SOZfeat, nonSOZfeat in zip(SOZ_columns, nonSOZ_columns):
    shapiro = stats.shapiro(median_feats[SOZfeat] - median_feats[nonSOZfeat])
    print(SOZfeat, nonSOZfeat)
    print(shapiro)

#%% wilcoxon test
for SOZfeat, nonSOZfeat in zip(SOZ_columns, nonSOZ_columns):
    wilcoxon = stats.wilcoxon(median_feats[SOZfeat], median_feats[nonSOZfeat])
    print(SOZfeat, nonSOZfeat)
    print(wilcoxon)

#%% paired t-test
for SOZfeat, nonSOZfeat in zip(SOZ_columns, nonSOZ_columns):
    ttest = stats.ttest_rel(median_feats[SOZfeat], median_feats[nonSOZfeat])
    print(SOZfeat, nonSOZfeat)
    print(ttest)
# %%
