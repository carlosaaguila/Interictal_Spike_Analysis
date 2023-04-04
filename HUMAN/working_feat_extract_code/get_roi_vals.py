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



#functions we will use
def find_soz_region(SOZ, brain_df):
    """ returns a list with all the regions of the seizure onset zone """
    brain_df['SOZ'] = brain_df['name'].apply(lambda x: 1 if x in SOZ else 0)
    region = brain_df['final_label'][brain_df['SOZ'] == 1].to_list()
    return region

def biglist_roi(ptnames, roi):
    """
    generates a big list of all relative values based on region of interest.
    takes in multiple regions, thanks to value_basis_multiroi > the results are indexed in order of the roi list we use as the import. *track this*
    """
    roiLlat_values = []
    roiLlat_idxch = []
    spike_soz = []
    count = 0
    for pt in ptnames:
        print(pt)
        spike, brain_df, _ = load_ptall(pt, data_directory)
        if isinstance(brain_df, pd.DataFrame) == False: #checks if RID exists
            count += 1
            continue
        if spike.fs[0][-1] < 500:
            count += 1
            print("low fs - skip")
            continue
        vals, _, idxch = value_basis_multiroi(spike, brain_df, roi)
        roiLlat_values.append(vals)
        roiLlat_idxch.append(idxch)    
        region = find_soz_region(spike.soz, brain_df)
        spike_soz.append([spike.soz, region]) #should get skipped if there isn't any 
        if vals == 0: #checks if there is no overlap
            count += 1
            continue
    return roiLlat_values, roiLlat_idxch, spike_soz

#update code to get count for each.
def spike_count_perregion(values): #could be generalized to multiple features easily. 
    """
    returns a list of the counts of each pt, per channel in ROI
    """
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
                    count.append(len(x)) #here you replace with a feature maker
            roi_count_perpt.append(count)
        spike_count_perpt.append(roi_count_perpt)

    #clean up, and sort into pt > roi > values (counts in this case)
    totalcount_perpt = []
    for pt in spike_count_perpt:
        sum_perroi = []
        for roi in pt:
            sum_perroi.append(np.sum(roi))
        totalcount_perpt.append(sum_perroi)

    fullcount_perroi = np.sum(totalcount_perpt, axis=0)
    return spike_count_perpt, totalcount_perpt, fullcount_perroi


#
values, idxch, spike_soz = biglist_roi(ptnames, roilist)

SAVE = [values, idxch, spike_soz]
with open('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/ROI_VALS/values.pkl', 'wb') as F:
    pkl.dump(SAVE, F)