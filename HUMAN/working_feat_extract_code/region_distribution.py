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

# %%
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

# %%
# want to create a heat map with the zones representing the distribution of spikes. 
# (each row is a pt, each column shows the count)
brain_dfs = pd.DataFrame()
count = 0
for pt in ptnames:
    rid,DF = load_cleaned_braindf(pt,data_directory)
    if isinstance(DF, pd.DataFrame) == False:
        continue
    else:
        brain_dfs = brain_dfs.append(DF)
        count += 1

display(count)

#save file for the concat'd dataframe
#filepath = '/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/concat_brain_df.csv'
#brain_dfs.to_csv(filepath)


#%%
""" commented out, because I saved this dataframe as a csv, and added region labels manually. 
region_count = brain_dfs.groupby('final_label').count()['name'].reset_index()
region_count = region_count.sort_values(by='name',ascending=False)
display(region_count)
"""

#%%
filepath = '/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/allcounts_brain_df.csv'
region_count = pd.read_csv(filepath)
#region_count.to_csv(filepath)

#%%
trunc_count = region_count[region_count['name'] > 20]
trunc_count = trunc_count.groupby('region').sum('name')['name'].reset_index().sort_values(by='name',ascending = False)
display(trunc_count)

"""
There was 3 brain stem???
30/25 for Cerebellum Total of 50 channels across our 77 patients. 
"""

# %% Create Heat Map that shows % of spikes in a region vs the SOZ localization. (we can start with lateralization first then exact)
def find_soz_region(SOZ, brain_df):
    """ returns a list with all the regions of the seizure onset zone """
    brain_df['SOZ'] = brain_df['name'].apply(lambda x: 1 if x in SOZ else 0)
    region = brain_df['final_label'][brain_df['SOZ'] == 1].to_list()
    return region

def biglist_roi(ptnames, roi):
    roiLlat_values = []
    roiLlat_idxch = []
    spike_soz = []
    count = 0
    for pt in ptnames:
        print(pt)
        spike, brain_df, ids = load_ptall(pt, data_directory)
        if isinstance(brain_df, pd.DataFrame) == False: #checks if RID exists
            count += 1
            continue
        if spike.fs[0][-1] < 500:
            count += 1
            print("low fs - skip")
            continue
        vals, chnum, idxch = value_basis_multiroi(spike, brain_df, roi)
        roiLlat_values.append(vals)
        roiLlat_idxch.append(idxch)    
        region = find_soz_region(spike.soz, brain_df)
        spike_soz.append([spike.soz, region]) #should get skipped if there isn't any 
        if vals == 0: #checks if there is no overlap
            count += 1
            continue
    return roiLlat_values, roiLlat_idxch, spike_soz

#update code to get count for each.
def spike_count_perregion(all_values, roilist):
    spike_count_perpt = []
    count = []
    roi_count_perpt = []
    totalcount_perpt = []
    for pt in values:
        roi_count_perpt = []
        for roi in pt:
            count = []
            for x in roi:
                if np.shape(roi) == (1,1):
                    idv_count = 0
                    count.append(idv_count)
                    continue
                else:
                    count.append(len(x))
            roi_count_perpt.append(count) #returns a list of the counts of each pt, per channel in ROI
        spike_count_perpt.append(roi_count_perpt)

    totalcount_perpt = []
    for pt in spike_count_perpt:
        sum_perroi = []
        for roi in pt:
            sum_perroi.append(np.sum(roi))
        totalcount_perpt.append(sum_perroi)

    fullcount_perroi = np.sum(totalcount_perpt, axis=0)
    return spike_count_perpt, totalcount_perpt, fullcount_perroi

#%% multiroi approach:
values, idxch, spike_soz = biglist_roi(ptnames, roilist)

#%% individual pull (using the old version of the code) This is just based_values substituted.
#left mesial
"""
NEW FUNCTION TAKES CARE OF THIS (BIGLIST_ROI + multipt based values)
roiLmesi_values, roiLmesi_idxch = biglist_roi(ptnames, roiL_mesial)
counts_Lmesi, count_Lmesi_perpt, fullcount_Lmesi = spike_count_perregion(roiLmesi_values) #returns a list of the counts of each pt, per channel in ROI

#left lateral
roiLlat_values,roiLlat_idxch = biglist_roi(ptnames, roiL_lateral)
counts_Llat, count_Llat_perpt, fullcount_Llat = spike_count_perregion(roiLlat_values) #returns a list of the counts of each pt, per channel in ROI

#right mesial
roiRmesi_values, roiRmesi_idxch = biglist_roi(ptnames, roiR_mesial)
counts_Rmesi, count_Rmesi_perpt, fullcount_Rmesi = spike_count_perregion(roiRmesi_values) #returns a list of the counts of each pt, per channel in ROI

#right lateral
roiRlat_values, roiRlat_idxch = biglist_roi(ptnames, roiR_lateral)
counts_Rlat, count_Rlat_perpt, fullcount_Rlat = spike_count_perregion(roiRlat_values) #returns a list of the counts of each pt, per channel in ROI

#L OC
roiLOC_values, roiLOC_idxch = biglist_roi(ptnames, L_OC)
counts_LOC, count_LOC_perpt, fullcount_LOC = spike_count_perregion(roiLOC_values)

#R OC
roiROC_values, roiROC_idxch = biglist_roi(ptnames, R_OC)
counts_ROC, count_ROC_perpt, fullcount_ROC = spike_count_perregion(roiROC_values)
#emptylabel
#roiNAN_values, roiNAN_idxch = biglist_roi(ptnames, emptylabel)
#counts_NAN, count_NAN_perpt, fullcount_NAN = spike_count_perregion(roiNAN_values)
"""
# %% Plot a bar plot showing the counts
totalcounts= np.sum([fullcount_Rlat,fullcount_Rmesi,fullcount_Llat,fullcount_Lmesi])#,fullcount_ROC, fullcount_LOC)
listofcounts = [fullcount_Rlat,fullcount_Rmesi,fullcount_Llat,fullcount_Lmesi]#,fullcount_ROC, fullcount_LOC]
percentofcount = np.divide(listofcounts,totalcounts)

fig = plt.figure(figsize=(10,10))
plt.bar(['R L','R M','L L', 'L M'], percentofcount)
plt.title('Percent of Spikes per Region')
plt.ylabel('%')
plt.xlabel('region')
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/spike_counts1") #save as jpg


#%% New plots with all the counts.



#%% Create average feature value (normalized) per spike region vs localization (or lateralization)

