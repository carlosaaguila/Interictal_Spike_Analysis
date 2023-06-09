#%% 
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


#%%
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
    roiLlat_chs = []
    roiLlat_selectoi = []
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
        vals, idxch, chs, select_oi = value_basis_multiroi(spike, brain_df, roi)
        #here instead of storing the val, you want to grab the feature of interest. 
        roiLlat_values.append(vals)
        roiLlat_idxch.append(idxch)
        roiLlat_chs.append(chs)
        roiLlat_selectoi.append(select_oi)
        region = find_soz_region(spike.soz, brain_df)
        infer_spike_soz.append([spike.soz, region]) #should get skipped if there isn't any 
        clinic_soz.append(soz_region)
        if vals == 0: #checks if there is no overlap
            count += 1
            continue
    return roiLlat_values, roiLlat_idxch,roiLlat_chs, roiLlat_selectoi, infer_spike_soz, clinic_soz

#update code to get count for each.
def spike_count_perregion(select_oi): #could be generalized to multiple features easily. 

    total_count_perpt = []
    count_roi = []
    count_perpt = []
    for pt in select_oi:
        count_roi = []
        for roi in pt:
            count_roi.append(np.size(roi))
        count_perpt.append(count_roi)

    total_count_perpt = np.sum(count_perpt, axis=1)
        
    return count_perpt, total_count_perpt

def spike_amplitude_perregion(values, chs, select_oi):

    perpt = []
    perpt_mean = []
    for i, pt in enumerate(values):
        #print('new pt {}'.format(i))
        perroi_mean = []
        perroi = []

        for j, roi in enumerate(pt):
            #print('roi {}'.format(j))
            spikefeat = []

            if not roi:
                perroi.append(np.nan)
                perroi_mean.append(np.nan)
                continue

            for l, xs in enumerate(roi):

                val_want = np.transpose(xs)
                val_want = val_want[chs[i][j][select_oi[i][j][l]]-1]
                feat = np.max(np.absolute(val_want[750:1251]))
                spikefeat.append(feat)

            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))
        perpt.append(perroi)
        perpt_mean.append(perroi_mean)  


    return perpt, perpt_mean

def spike_LL_perregion(values, chs, select_oi): #could be generalized to multiple features easily. 
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

            if not roi:
                perroi.append(np.nan)
                perroi_mean.append(np.nan)
                continue
            for l, xs in enumerate(roi):
                    val_want = np.transpose(xs)
                    val_want = val_want[chs[i][j][select_oi[i][j][l]]-1]
                    feat = LL(val_want[750:1500]) #added a constraint to hopefully capture the spike
                    spikefeat.append(feat)
            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))
        perpt.append(perroi)
        perpt_mean.append(perroi_mean)

    return perpt, perpt_mean

def divide_chunks(l, n):
    # looping till length l
    biglist = []
    for i in range(0, len(l), n):
        list = l[i:i + n]
        biglist.append(list)
    return biglist

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
        values, idxch, chs, select_oi, infer_spike_soz, clinic_soz = biglist_roi(list, roilist)
        clinic_soz_all.append(clinic_soz)

        #features
        Aperpt, Aperpt_mean = spike_amplitude_perregion(values, chs, select_oi)
        count_perpt, total_count_perpt = spike_count_perregion(select_oi)
        LLperpt, LLperpt_mean = spike_LL_perregion(values, chs, select_oi)

        Aperpt_mean_all.append(Aperpt_mean)
        totalcount_perpt_all.append(count_perpt)
        LLperpt_mean_all.append(LLperpt_mean)
    

    Aperpt_mean = [x for x in Aperpt_mean_all for x in x]
    LLperpt_mean = [x for x in LLperpt_mean_all for x in x]
    totalcount_perpt = [x for x in totalcount_perpt_all for x in x]
    clinic_soz = [x for x in clinic_soz_all for x in x]

    return Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz

# run biglist_roi >> gives you all the values/roi, idxch/roi, the inferred spike soz (based off electrodes), and the SOZ determined by the clinician. 

#%% call em. 
n=15
lists_ptnames = (divide_chunks(ptnames, n))
Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz = feat_extract(lists_ptnames)#, roilist, data_directory)

#%%
#CLASS LIST COMBINATION
to_combine = ['bilateral - diffuse', 'bilateral - mesial temporal', 'bilateral - multifocal' , 'bilateral - temporal multifocal','diffuse - diffuse', 'left - diffuse' ,'left - multifocal', 'right - multifocal']
to_remove = ['temporal', 'frontal']

#%% amplitude
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
sns.heatmap(amp_df.transpose(), cmap='Purples', cbar_kws = {'label':'Median Amplitude'})
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Amplitude across all Patients')
plt.yticks(rotation = 0)
#fig.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/amp_persoz3", bbox_inches = 'tight')
fig.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/NEW/feature maps/amplitude_persoz', bbox_inches = 'tight')

#%% LL
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
sns.heatmap(ll_df.transpose(), cmap='Purples', cbar_kws = {'label':'Median Linelength'})
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Linelength across all Patients')
plt.yticks(rotation = 0)
#fig2.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/distribution/ll_persoz_croppedwindow3", bbox_inches = 'tight')
fig2.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/NEW/feature maps/linelength_persoz', bbox_inches = 'tight')


#%% spikecount NO emptylabels

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
fig4.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/NEW/feature maps/countperc_persoz', bbox_inches = 'tight')

# %% exmaple of how to plot paired plots w/ different colors for a single feature

ll_df_combine

#left
left_pts_LL = ll_df_combine[ll_df_combine['lateralization'] == 'left'] 
left_mesial_LL = left_pts_LL[left_pts_LL['region'] == 'mesial temporal']
left_neocort_LL = left_pts_LL[left_pts_LL['region'] == 'temporal neocortical']

#right
right_pts_LL = ll_df_combine[ll_df_combine['lateralization'] == 'right'] 
right_mesial_LL = right_pts_LL[right_pts_LL['region'] == 'mesial temporal']
right_neocort_LL = right_pts_LL[right_pts_LL['region'] == 'temporal neocortical']

#plot left_mesial_first
clean_left_mesial_amp = left_mesial_LL[['L_Mesial', 'R_Mesial']].dropna()
clean_left_mesial_amp_DOWN = clean_left_mesial_amp[clean_left_mesial_amp['L_Mesial'] > clean_left_mesial_amp['R_Mesial']]
clean_left_mesial_amp_UP = clean_left_mesial_amp[clean_left_mesial_amp['L_Mesial'] < clean_left_mesial_amp['R_Mesial']]

plt.plot(clean_left_mesial_amp_DOWN['L_Mesial'].to_list(), clean_left_mesial_amp_DOWN['R_Mesial'].to_list(), 'x', color = 'r')
plt.plot(clean_left_mesial_amp_UP['L_Mesial'].to_list(), clean_left_mesial_amp_UP['R_Mesial'].to_list(), 'x', color = 'b')

plt.plot(range(0,10000), range(0,10000), 'k')
plt.ylim([0,10000])
plt.xlim([0,10000])
plt.ylabel('Right Mesial')
plt.xlabel('Left Mesial')
plt.title('Line Length - Left-Mesial Pts, n = {}'.format(len(clean_left_mesial_amp['L_Mesial'].to_list())))
plt.show()

#plot Left Neocortical
clean_left_mesial_amp = left_neocort_LL[['L_Lateral', 'R_Lateral']].dropna()
clean_left_mesial_amp_DOWN = clean_left_mesial_amp[clean_left_mesial_amp['L_Lateral'] > clean_left_mesial_amp['R_Lateral']]
clean_left_mesial_amp_UP = clean_left_mesial_amp[clean_left_mesial_amp['L_Lateral'] < clean_left_mesial_amp['R_Lateral']]

plt.plot(clean_left_mesial_amp_DOWN['L_Lateral'].to_list(), clean_left_mesial_amp_DOWN['R_Lateral'].to_list(), 'x', color = 'r')
plt.plot(clean_left_mesial_amp_UP['L_Lateral'].to_list(), clean_left_mesial_amp_UP['R_Lateral'].to_list(), 'x', color = 'b')

plt.plot(range(0,10000), range(0,10000), 'k')
plt.ylim([0,8000])
plt.xlim([0,8000])
plt.ylabel('Right Lateral')
plt.xlabel('Left Lateral')
plt.title('Line Length - Left Neocortical Temporal Pts, n = {}'.format(len(clean_left_mesial_amp['L_Lateral'].to_list())))
plt.show()

#plot Right Neocortical
clean_left_mesial_amp = right_neocort_LL[['L_Lateral', 'R_Lateral']].dropna()
clean_left_mesial_amp_DOWN = clean_left_mesial_amp[clean_left_mesial_amp['L_Lateral'] < clean_left_mesial_amp['R_Lateral']]
clean_left_mesial_amp_UP = clean_left_mesial_amp[clean_left_mesial_amp['L_Lateral'] > clean_left_mesial_amp['R_Lateral']]

plt.plot(clean_left_mesial_amp_DOWN['L_Lateral'].to_list(), clean_left_mesial_amp_DOWN['R_Lateral'].to_list(), 'x', color = 'b')
plt.plot(clean_left_mesial_amp_UP['L_Lateral'].to_list(), clean_left_mesial_amp_UP['R_Lateral'].to_list(), 'x', color = 'r')

plt.plot(range(0,4500), range(0,4500), 'k')
plt.ylim([0,4500])
plt.xlim([0,4500])
plt.xlabel('Right Lateral')
plt.ylabel('Left Lateral')
plt.title('Line Length - Right Neocortical Temporal Pts, n = {}'.format(len(clean_left_mesial_amp['L_Lateral'].to_list())))
plt.show()

#plot Right Mesial
clean_left_mesial_amp = right_mesial_LL[['L_Mesial', 'R_Mesial']].dropna()
clean_left_mesial_amp_DOWN = clean_left_mesial_amp[clean_left_mesial_amp['L_Mesial'] < clean_left_mesial_amp['R_Mesial']]
clean_left_mesial_amp_UP = clean_left_mesial_amp[clean_left_mesial_amp['L_Mesial'] > clean_left_mesial_amp['R_Mesial']]

plt.plot(clean_left_mesial_amp_DOWN['L_Mesial'].to_list(), clean_left_mesial_amp_DOWN['R_Mesial'].to_list(), 'x', color = 'b')
plt.plot(clean_left_mesial_amp_UP['L_Mesial'].to_list(), clean_left_mesial_amp_UP['R_Mesial'].to_list(), 'x', color = 'r')

plt.plot(range(0,7500), range(0,7500), 'k')
plt.ylim([0,7500])
plt.xlim([0,7500])
plt.xlabel('Right Mesial')
plt.ylabel('Left Mesial')
plt.title('Line Length - Right Mesial Temporal Pts, n = {}'.format(len(clean_left_mesial_amp['L_Mesial'].to_list())))
plt.show()


# %%  Normative ampltiude heatmap

#laod amplitude data

#create sozlist
sozlist = amp_df_combine['soz'].to_list()
norm_amp_df = amp_df_combine

#create normative dataframe
for index, row in norm_amp_df.iterrows():
    #mesial soz
    if sozlist[index] == "left - mesial temporal":
        norm_amp_df.at[index, 'L_Mesial'] = None
    if sozlist[index] == "right - mesial temporal":
        norm_amp_df.at[index, 'R_Mesial'] = None
    #neo cortical soz
    if sozlist[index] == "left - temporal neocortical":
        norm_amp_df.at[index, 'L_Lateral'] = None
    if sozlist[index] == "right - temporal neocortical":
        norm_amp_df.at[index, 'R_Lateral'] = None
    #other soz
    if sozlist[index] == "right - other cortex":
        norm_amp_df.at[index, 'R_OtherCortex'] = None
    if sozlist[index] == "left - other cortex":
        norm_amp_df.at[index, 'L_OtherCortex'] = None

#clean up
for l in to_remove: #remove to_remove variables
    norm_amp_df = norm_amp_df[norm_amp_df['region'] != l]

norm_amp_df = norm_amp_df.drop(columns = ['region','lateralization'])
norm_amp_df['soz'] = norm_amp_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)
norm_amp_df = norm_amp_df.groupby(by='soz').median()

#plot heatmap
fig = plt.figure(figsize=(8,8))
sns.heatmap(norm_amp_df.transpose(), cmap='Purples', cbar_kws = {'label':'Median Amplitude'}, vmax = 600, annot = True, fmt = '.1f')
plt.xlabel('Clinically Defined SOZ')
plt.ylabel('Spiking Brain Region')
plt.title('Normative Amplitude across all Patients')
plt.yticks(rotation = 0)