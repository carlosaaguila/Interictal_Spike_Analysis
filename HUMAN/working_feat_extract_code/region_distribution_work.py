#set up environment
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


# functions to use:
def find_soz_region(SOZ, brain_df):
    """ returns a list with all the regions of the seizure onset zone """
    brain_df['SOZ'] = brain_df['name'].apply(lambda x: 1 if x in SOZ else 0)
    region = brain_df['final_label'][brain_df['SOZ'] == 1].to_list()
    return region

def biglist_roi(ptnames, roi = roilist):
    """
    generates a big list of all relative values based on region of interest.
    takes in multiple regions, thanks to value_basis_multiroi > the results are indexed in order of the roi list we use as the import. *track this*
    """
    roiLlat_values = []
    roiLlat_idxch = []
    infer_spike_soz = []
    clinic_soz = []
    count = 0
    for pt in ptnames:
        print(pt)
        spike, brain_df, soz_region, _  = load_ptall(pt, data_directory)
        if isinstance(brain_df, pd.DataFrame) == False: #checks if RID exists
            count += 1
            continue
        if spike.fs[0][-1] < 500:
            count += 1
            print("low fs - skip")
            continue
        vals, _, idxch = value_basis_multiroi(spike, brain_df, roi)
        #here instead of storing the val, you want to grab the feature of interest. 
        roiLlat_values.append(vals)
        roiLlat_idxch.append(idxch)    
        region = find_soz_region(spike.soz, brain_df)
        infer_spike_soz.append([spike.soz, region]) #should get skipped if there isn't any 
        clinic_soz.append(soz_region)
        if vals == 0: #checks if there is no overlap
            count += 1
            continue
    return roiLlat_values, roiLlat_idxch, infer_spike_soz, clinic_soz

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

def spike_amplitude_perregion(values, idxch): #could be generalized to multiple features easily. 
    """
    returns a list of the features of interest of each pt, per channel in ROI
    """
    perpt = []
    perpt_mean = []
    for i, pt in enumerate(values):
        perroi_mean = []
        perroi = []
        for j, roi in enumerate(pt):
            spikefeat = []
            for l, xs in enumerate(roi):
                if np.shape(roi) == (1,1):
                    feat = np.nan
                    spikefeat.append(feat)
                    continue
                else:
                    for x in xs:
                        val_want = np.transpose(x)
                        val_want = val_want[idxch[i][j][l]]
                        feat = np.max(np.absolute(val_want[750:1251]))
                        spikefeat.append(feat)
            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))
        perpt.append(perroi)
        perpt_mean.append(perroi_mean)

    return perpt, perpt_mean

def spike_LL_perregion(values, idxch): #could be generalized to multiple features easily. 
    """
    returns a list of the features of interest of each pt, per channel in ROI
    """
    perpt = []
    perpt_mean = []
    for i, pt in enumerate(values):
        perroi_mean = []
        perroi = []
        for j, roi in enumerate(pt):
            spikefeat = []
            for l, xs in enumerate(roi):
                if np.shape(roi) == (1,1):
                    feat = np.nan
                    spikefeat.append(feat)
                    continue
                else:
                    for x in xs:
                        val_want = np.transpose(x)
                        val_want = val_want[idxch[i][j][l]]
                        feat = LL(val_want[750:1500]) #added a constraint to hopefully capture the spike
                        spikefeat.append(feat)
            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))
        perpt.append(perroi)
        perpt_mean.append(perroi_mean)

    return perpt, perpt_mean

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def feat_extract(lists_ptnames):
    clinic_soz_all = []
    Aperpt_mean_all = []
    totalcount_perpt_all = []
    LLperpt_mean_all = []

    for list in lists_ptnames:
        #clear at the start to reduce memory load
        values = []
        idxch = []
        infer_spike_soz = []
        print('cleared + new pt list')

        #values
        values, idxch, infer_spike_soz, clinic_soz = biglist_roi(list, roilist)
        clinic_soz_all.append(clinic_soz)

        #features
        Aperpt, Aperpt_mean = spike_amplitude_perregion(values, idxch)
        spike_count_perpt, totalcount_perpt, fullcount_perroi = spike_count_perregion(values)
        LLperpt, LLperpt_mean = spike_LL_perregion(values, idxch)

        Aperpt_mean_all.append(Aperpt_mean)
        totalcount_perpt_all.append(totalcount_perpt)
        LLperpt_mean_all.append(LLperpt_mean)
    

    Aperpt_mean = [x for x in Aperpt_mean_all for x in x]
    LLperpt_mean = [x for x in LLperpt_mean_all for x in x]
    totalcount_perpt = [x for x in totalcount_perpt_all for x in x]
    clinic_soz = [x for x in clinic_soz_all for x in x]

    return Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz

# run biglist_roi >> gives you all the values/roi, idxch/roi, the inferred spike soz (based off electrodes), and the SOZ determined by the clinician. 

n=15
lists_ptnames = divide_chunks(ptnames, n)
Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz = feat_extract(lists_ptnames)


"""
#save the values. 
SAVE = [values, idxch, infer_spike_soz, clinic_soz]
with open('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/ROI_VALS/values.pkl', 'wb') as F:
    pkl.dump(SAVE, F)


#calculate spike rates (total count) for patient per ROI
spike_count_perpt, totalcount_perpt, fullcount_perroi = spike_count_perregion(values)

#give a label to the electrode
spike_soz2 = []
for x in spike_soz:
    roi_name = []
    for y in x[1]:
        if y in roiL_mesial:
            roi_name.append('L_mesial')
        if y in roiL_lateral:
            roi_name.append('L_lateral')
        if y in roiR_mesial:
            roi_name.append('R_mesial')
        if y in roiR_lateral:
            roi_name.append('R_lateral')
        if y in L_OC:
            roi_name.append('LOC')
        if y in R_OC:
            roi_name.append('ROC')
        if y in emptylabel:
            roi_name.append('no label')
    spike_soz2.append(roi_name)

from collections import Counter

#function to get the K most frequent
def top_k(numbers, k=2):
    
    #The counter.most_common([k]) method works
    #in the following way:
    #>>> Counter('abracadabra').most_common(3)  
    #[('a', 5), ('r', 2), ('b', 2)]
    

    c = Counter(numbers)
    most_common = [key for key, val in c.most_common(k)]

    return most_common

#find top 2 regions based on electrode (for SOZ column)
spike_soz_top2 = []
for pt in spike_soz2:
    if len(pt) == 0:
        spike_soz_top2.append(["no label"])
    else:
        spike_soz_top2.append(top_k(pt,k=2))

#find the 1 SOZ region > that isnt no label
spike_soz_onlyone = []
for pt in spike_soz_top2:
    if (pt[0] == "no label"):
        spike_soz_onlyone.append(pt[-1])
    else:
        spike_soz_onlyone.append(pt[0])

#put into pretty format for heatmap
test_df = pd.DataFrame(data = totalcount_perpt)
test_df['soz'] = spike_soz_onlyone
test_df2 = test_df.groupby(by='soz').sum()
test_df_droplabels = test_df2.drop(columns=['no label'])
test_df_percentage = test_df_droplabels.div(test_df_droplabels.sum(axis=1), axis = 0)
test_df_percentage = test_df_droplabels.mul(100)

#plot feature
test_df = pd.DataFrame(data = LLperptmean)
test_df['soz'] = spike_soz_onlyone
test_df = test_df.rename(columns={0:'L Mesial', 1:'L Lateral', 2:'R Mesial', 3:'R Lateral', 4:'L OC', 5:'R OC', 6:'no label'})
test_df_droplabels = test_df[test_df['soz'] != 'no label']
test_df2 = test_df_droplabels.groupby(by="soz").median()
sns.heatmap(test_df2, cmap='crest')
plt.title('Line Length across all 21 patients')
plt.xlabel('Brain Region')
plt.ylabel('SOZ')
"""

#CLASS LIST COMBINATION
to_combine = ['bilateral - diffuse', 'bilateral - mesial temporal', 'bilateral - multifocal' , 'bilateral - temporal multifocal','diffuse - diffuse', 'left - diffuse' ,'left - multifocal', 'right - multifocal']
to_remove = ['temporal', 'frontal']

#amplitude
test_df = pd.DataFrame(data = Aperpt_mean)
soz_df  = pd.DataFrame(data = clinic_soz)
test_df = test_df.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df = test_df.drop(columns = 'Empty Label')
soz_df = soz_df.rename(columns = {0:'region', 1:'lateralization'})
amp_df_combine =  pd.concat([test_df, soz_df], axis = 1)

for l in to_remove: #remove to_remove variables
    amp_df_combine = amp_df_combine[amp_df_combine['region'] != l]

amp_df_drop1 = amp_df_combine
amp_df_drop1['soz'] = amp_df_combine['lateralization'] + " - " + amp_df_combine['region'] 

amp_df = amp_df_drop1.drop(columns = ['region','lateralization'])
amp_df['soz'] = amp_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)
amp_df = amp_df.groupby(by='soz').median()

fig = plt.figure(figsize=(8,8))
sns.heatmap(amp_df.transpose(), cmap='crest', cbar_kws = {'label':'Median Amplitude'})
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Amplitude across all Patients')
plt.yticks(rotation = 0)
fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/amp_persoz3", bbox_inches = 'tight')

#LL
test_df = pd.DataFrame(data = LLperpt_mean)
soz_df  = pd.DataFrame(data = clinic_soz)
test_df = test_df.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df = test_df.drop(columns = 'Empty Label')
soz_df = soz_df.rename(columns = {0:'region', 1:'lateralization'})
amp_df_combine =  pd.concat([test_df, soz_df], axis = 1)

for l in to_remove: #remove to_remove variables
    amp_df_combine = amp_df_combine[amp_df_combine['region'] != l]

amp_df_drop1 = amp_df_combine
amp_df_drop1['soz'] = amp_df_combine['lateralization'] + " - " + amp_df_combine['region'] 

ll_df = amp_df_drop1.drop(columns = ['region','lateralization'])
ll_df['soz'] = ll_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)
ll_df = ll_df.groupby(by='soz').median()

fig2 = plt.figure(figsize=(8,8))
sns.heatmap(ll_df.transpose(), cmap='crest', cbar_kws = {'label':'Median Linelength'})
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Linelength across all Patients')
plt.yticks(rotation = 0)
fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/ll_persoz_croppedwindow3", bbox_inches = 'tight')

#spikecount
test_df = pd.DataFrame(data = totalcount_perpt)
soz_df  = pd.DataFrame(data = clinic_soz)
test_df = test_df.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
soz_df = soz_df.rename(columns = {0:'region', 1:'lateralization'})
amp_df_combine =  pd.concat([test_df, soz_df], axis = 1)

for l in to_remove: #remove to_remove variables
    amp_df_combine = amp_df_combine[amp_df_combine['region'] != l]

amp_df_drop1 = amp_df_combine
amp_df_drop1['soz'] = amp_df_combine['lateralization'] + " - " + amp_df_combine['region'] 

#WANT TO FIND THE PERCENTAGE BREAKDOWN PER PT BEFORE GROUPING. 
count_df = amp_df_drop1.drop(columns = ['region','lateralization'])
count_df['soz'] = count_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

count_percent_perpt = count_df[['L_Mesial', 'L_Lateral', 'R_Mesial', 'R_Lateral', 'L_OtherCortex', 'R_OtherCortex', 'Empty Label']].div(count_df.sum(axis=1), axis =0).mul(100)
count_percent_perpt['soz'] = count_df['soz']
count_percent_perpt = count_percent_perpt.groupby(by='soz').median()

fig3 = plt.figure(figsize=(8,8))
sns.heatmap(count_percent_perpt.transpose(), cmap='crest', cbar_kws = {'label':'Spike Percentage'})
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Spike Count Percentage per SOZ across all Patients')
plt.yticks(rotation = 0)
fig3.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/countpercentage_persoz2", bbox_inches = 'tight')


#spikecount NO emptylabels

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

fig4 = plt.figure(figsize=(8,8))
sns.heatmap(count_percent_perpt.transpose(), cmap='crest', cbar_kws = {'label':'Spike Percentage'})
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Spike Count Percentage per SOZ across all Patients')
plt.yticks(rotation = 0)
fig4.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/countpercentage_persoz3", bbox_inches = 'tight')

