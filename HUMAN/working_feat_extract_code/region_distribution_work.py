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
        soz_region.append(pt)
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
                perroi.append([np.nan])
                perroi_mean.append([np.nan])
                continue

            for l, xs in enumerate(roi):

                val_want = np.transpose(xs)
                val_want = val_want[chs[i][j][select_oi[i][j][l]]-1]
                demean_val = val_want - np.nanmean(val_want)
                feat = np.max(np.absolute(demean_val[750:1251]))
                spikefeat.append(feat)

            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))

        perpt.append(perroi)
        perpt_mean.append(perroi_mean)  


    return perpt, perpt_mean

def morphology_perregion(values, chs, select_oi, list_num):
    print('listnum {}'.format(list_num))
    perpt_riseamp = []
    perpt_decayamp = []
    perpt_slowwidth = []
    perpt_slowamp = []
    perpt_riseslope = []
    perpt_decayslope = []
    perpt_averageamp = []
    perpt_linelen = []

    for i, pt in enumerate(values):
        print('new pt {}'.format(i))
        perroi_riseamp= []
        perroi_decayamp = []
        perroi_slowwidth = []
        perroi_slowamp = []
        perroi_riseslope = []
        perroi_decayslope = []
        perroi_averageamp = []
        perroi_linelen = []

        for j, roi in enumerate(pt):
            print('roi {}'.format(j))
            feat_rise_amp = []
            feat_decay_amp = []
            feat_slow_width = []
            feat_slow_amp = []
            feat_rise_slope = []
            feat_decay_slope = []
            feat_average_amp = []
            feat_linelen = []

            if not roi:
                perroi_riseamp.append([np.nan])
                perroi_decayamp.append([np.nan])
                perroi_slowwidth.append([np.nan])
                perroi_slowamp.append([np.nan])
                perroi_riseslope.append([np.nan])
                perroi_decayslope.append([np.nan])
                perroi_averageamp.append([np.nan])
                perroi_linelen.append([np.nan])
                continue

            for l, xs in enumerate(roi):
                if (list_num >= 4): #debugmode
                    print('spike {}'.format(l))

                val_want = np.transpose(xs)
                val_want = val_want[chs[i][j][select_oi[i][j][l]]-1]

                #remove the spikes that are not working (its not alot)
                if (list_num == 0) & (i == 0) & (j == 6) & (l == 80):
                    continue
                elif (list_num == 0) & (i == 4) & (j == 1) & (l == 23):
                    continue
                elif (list_num == 1) & (i == 9) & (j == 5) & (l == 12):
                    continue
                elif (list_num == 1) & (i == 9) & (j == 6) & (l == 254):
                    continue
                elif (list_num == 3) & (i == 0) & (j == 6) & (l == 123):
                    continue
                elif (list_num == 3) & (i == 0) & (j == 6) & (l == 526):
                    continue
                elif (list_num == 4) & (i == 7) & (j == 5) & (l == 83):
                    continue
                elif (list_num == 4) & (i == 7) & (j == 5) & (l == 341):
                    continue
                elif (list_num == 5) & (i == 9) & (j == 5) & (l == 25):
                    continue
                else:
                    rise_amp, decay_amp, slow_width, slow_amp, rise_slope, decay_slope, average_amp, linelen = morphology_feats_v1(val_want)

                feat_rise_amp.append(rise_amp)
                feat_decay_amp.append(decay_amp)
                feat_slow_width.append(slow_width)
                feat_slow_amp.append(slow_amp)
                feat_rise_slope.append(rise_slope)
                feat_decay_slope.append(decay_slope)
                feat_average_amp.append(average_amp)
                feat_linelen.append(linelen)

            perroi_riseamp.append(feat_rise_amp)
            perroi_decayamp.append(feat_decay_amp)
            perroi_slowwidth.append(feat_slow_width)
            perroi_slowamp.append(feat_slow_amp)
            perroi_riseslope.append(feat_rise_slope)
            perroi_decayslope.append(feat_decay_slope)
            perroi_averageamp.append(feat_average_amp)
            perroi_linelen.append(feat_linelen)

        perpt_riseamp.append(perroi_riseamp)
        perpt_decayamp.append(perroi_decayamp)
        perpt_slowwidth.append(perroi_slowwidth)
        perpt_slowamp.append(perroi_slowamp)
        perpt_riseslope.append(perroi_riseslope)
        perpt_decayslope.append(perroi_decayslope)
        perpt_averageamp.append(perroi_averageamp)
        perpt_linelen.append(perroi_linelen)


    return perpt_riseamp, perpt_decayamp, perpt_slowwidth, perpt_slowamp, perpt_riseslope, perpt_decayslope, perpt_averageamp, perpt_linelen

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
                perroi.append([np.nan])
                perroi_mean.append([np.nan])
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
    list_num = 0  
    riseamp_perpt_all = []
    decayamp_perpt_all = []
    slowwidth_perpt_all = []
    slowamp_perpt_all = []
    riseslope_perpt_all = []
    decayslope_perpt_all = []
    averageamp_perpt_all = []
    linelen_perpt_all = []

    for list in lists_ptnames:
        #clear at the start to reduce memory load
        values = []
        print('cleared + new pt list')

        #values
        values, idxch, chs, select_oi, infer_spike_soz, clinic_soz = biglist_roi(list, roilist)
        clinic_soz_all.append(clinic_soz)

        #features
        perpt_riseamp, perpt_decayamp, perpt_slowwidth, perpt_slowamp, perpt_riseslope, perpt_decayslope, perpt_averageamp, perpt_linelen = morphology_perregion(values, chs, select_oi, list_num)

        riseamp_perpt_all.append(perpt_riseamp)
        decayamp_perpt_all.append(perpt_decayamp)
        slowwidth_perpt_all.append(perpt_slowwidth)
        slowamp_perpt_all.append(perpt_slowamp)
        riseslope_perpt_all.append(perpt_riseslope)
        decayslope_perpt_all.append(perpt_decayslope)
        averageamp_perpt_all.append(perpt_averageamp)
        linelen_perpt_all.append(perpt_linelen)

        list_num +=1

    clinic_soz = [x for x in clinic_soz_all for x in x]
    riseamp_perpt = [x for x in riseamp_perpt_all for x in x]
    decayamp_perpt = [x for x in decayamp_perpt_all for x in x]
    slowwidth_perpt = [x for x in slowwidth_perpt_all for x in x]
    slowamp_perpt = [x for x in slowamp_perpt_all for x in x]
    riseslope_perpt = [x for x in riseslope_perpt_all for x in x]
    decayslope_perpt = [x for x in decayslope_perpt_all for x in x]
    averageamp_perpt = [x for x in averageamp_perpt_all for x in x]
    linelen_perpt = [x for x in linelen_perpt_all for x in x]

    return clinic_soz, riseamp_perpt, decayamp_perpt, slowwidth_perpt, slowamp_perpt, riseslope_perpt, decayslope_perpt, averageamp_perpt, linelen_perpt

# run biglist_roi >> gives you all the values/roi, idxch/roi, the inferred spike soz (based off electrodes), and the SOZ determined by the clinician. 
#%% call em. 
n=15
lists_ptnames = (divide_chunks(ptnames, n))
#Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz, Aperpt, LLperpt = feat_extract(lists_ptnames[0:2])#, roilist, data_directory)
#clinic_soz, Aperpt, LLperpt, totalcount_perpt = feat_extract(lists_ptnames)
clinic_soz, riseamp_perpt, decayamp_perpt, slowwidth_perpt, slowamp_perpt, riseslope_perpt, decayslope_perpt, averageamp_perpt, linelen_perpt = feat_extract(lists_ptnames)


#%%
#CLASS LIST COMBINATION
to_combine = ['bilateral - diffuse', 'bilateral - mesial temporal', 'bilateral - multifocal' , 'bilateral - temporal multifocal','diffuse - diffuse', 'left - diffuse' ,'left - multifocal', 'right - multifocal']
to_remove = ['temporal', 'frontal']

#%% amplitude
test_df = pd.DataFrame(data = Aperpt_mean)
soz_df  = pd.DataFrame(data = clinic_soz)
test_df = test_df.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df = test_df.drop(columns = 'Empty Label')
soz_df = soz_df.rename(columns = {0:'region', 1:'lateralization', 2:'pt'})
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
#fig.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/NEW/feature maps/amplitude_persoz', bbox_inches = 'tight')

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
#fig2.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/NEW/feature maps/linelength_persoz', bbox_inches = 'tight')


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
#fig4.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/spike figures/NEW/feature maps/countperc_persoz', bbox_inches = 'tight')

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

#%% CHECKER
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

def checker(ptname):
        
    data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
    spike, brain_df, soz, id = load_ptall(ptname, data_directory)
    print(id)
    print(soz)

    print(np.unique(spike.select[:,3]))
    #establish ROI's
    roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
    roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
    roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
    roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
    emptylabel = ['EmptyLabel','NaN']
    L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
    R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']

    #label name check
    roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]
    pd.options.display.max_rows = 999
    display(brain_df)

    #plot check
    X = brain_df['x'].to_list()
    Y = brain_df['y'].to_list()
    Z = brain_df['z'].to_list()
 
    fig = figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')

    X = brain_df['x'].to_list()
    Y = brain_df['y'].to_list()
    Z = brain_df['z'].to_list()
    names = brain_df['key_0'].to_list()

    for i in range(len(X)): #plot each point + it's index as text above
        ax.scatter(X[i],Y[i],Z[i],color='b') 
        ax.text(X[i],Y[i],Z[i],  '%s' % (str(names[i])), size=10, zorder=1,  
        color='k') 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    pyplot.show()

    #value check
    vals, idxch, chs, select_oi = value_basis_multiroi(spike, brain_df, roilist)
    basis_results = [vals, idxch, chs, select_oi]
    count_roi = []
    for roi in select_oi:
        count_roi.append(np.size(roi))
    print('L Mesial, L lateral, R Mesial, R Lateral, L OC, R OC, Empty')
    print(count_roi)

    return count_roi, brain_df, spike, basis_results

# %%

# for a list of 7 lists with varying sizes, append NaN's to the end of each list so that they are all the same size
def append_nan(list_of_lists):
    max_length = max([np.size(l) for l in list_of_lists])
    for l in list_of_lists:
        while np.size(l) < max_length:
            l.append(np.nan)
    return list_of_lists


for pt in riseamp_perpt:
    pt = append_nan(pt)
for pt in decayamp_perpt:
    pt = append_nan(pt)
for pt in slowwidth_perpt:
    pt = append_nan(pt)
for pt in slowamp_perpt:
    pt = append_nan(pt)
for pt in riseslope_perpt:
    pt = append_nan(pt)
for pt in decayslope_perpt:
    pt = append_nan(pt)
for pt in averageamp_perpt:
    pt = append_nan(pt)
for pt in linelen_perpt:
    pt = append_nan(pt)

# %% Make a large dataframe for mixed effects model
#CLASS LIST COMBINATION
to_combine = ['bilateral - diffuse', 'bilateral - mesial temporal', 'bilateral - multifocal' , 'bilateral - temporal multifocal','diffuse - diffuse', 'left - diffuse' ,'left - multifocal', 'right - multifocal']
to_remove = ['temporal', 'frontal']

test_df = pd.DataFrame(data = riseamp_perpt)
soz_df  = pd.DataFrame(data = clinic_soz)
test_df = test_df.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df = test_df.drop(columns = 'Empty Label')
labels = test_df.columns
soz_df = soz_df.rename(columns = {0:'region', 1:'lateralization', 2:'pt'})
riseamp_df_combine =  pd.concat([test_df, soz_df], axis = 1)

test_df2 = pd.DataFrame(data = decayamp_perpt)
test_df2 = test_df2.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df2 = test_df2.drop(columns = 'Empty Label')
decayamp_df_combine = pd.concat([test_df2, soz_df], axis = 1)

test_df3 = pd.DataFrame(data = slowwidth_perpt)
test_df3 = test_df3.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df3 = test_df3.drop(columns = 'Empty Label')
slowwidth_df_combine = pd.concat([test_df3, soz_df], axis = 1)

test_df4 = pd.DataFrame(data = slowamp_perpt)
test_df4 = test_df4.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df4 = test_df4.drop(columns = 'Empty Label')
slowamp_df_combine = pd.concat([test_df4, soz_df], axis = 1)

test_df5 = pd.DataFrame(data = riseslope_perpt)
test_df5 = test_df5.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df5 = test_df5.drop(columns = 'Empty Label')
riseslope_df_combine = pd.concat([test_df5, soz_df], axis = 1)

test_df6 = pd.DataFrame(data = decayslope_perpt)
test_df6 = test_df6.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df6 = test_df6.drop(columns = 'Empty Label')
decayslope_df_combine = pd.concat([test_df6, soz_df], axis = 1)

test_df7 = pd.DataFrame(data = averageamp_perpt)
test_df7 = test_df7.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df7 = test_df7.drop(columns = 'Empty Label')
averageamp_df_combine = pd.concat([test_df7, soz_df], axis = 1)

test_df8 = pd.DataFrame(data = linelen_perpt)
test_df8 = test_df8.rename(columns={0:'L_Mesial', 1:'L_Lateral', 2:'R_Mesial', 3:'R_Lateral',4:'L_OtherCortex', 5:'R_OtherCortex', 6:'Empty Label'})
test_df8 = test_df8.drop(columns = 'Empty Label')
LL_df_combine = pd.concat([test_df8, soz_df], axis = 1)



for l in to_remove: #remove to_remove variables
    riseamp_df_combine = riseamp_df_combine[riseamp_df_combine['region'] != l]
    decayamp_df_combine = decayamp_df_combine[decayamp_df_combine['region'] != l]
    slowwidth_df_combine = slowwidth_df_combine[slowwidth_df_combine['region'] != l]
    slowamp_df_combine = slowamp_df_combine[slowamp_df_combine['region'] != l]
    riseslope_df_combine = riseslope_df_combine[riseslope_df_combine['region'] != l]
    decayslope_df_combine = decayslope_df_combine[decayslope_df_combine['region'] != l]
    averageamp_df_combine = averageamp_df_combine[averageamp_df_combine['region'] != l]
    LL_df_combine = LL_df_combine[LL_df_combine['region'] != l]

#riseamp cleanup
riseamp_df_drop1 = riseamp_df_combine
riseamp_df_drop1['soz'] = riseamp_df_combine['lateralization'] + " - " + riseamp_df_combine['region']
riseamp_df = riseamp_df_drop1.drop(columns = ['region','lateralization'])
riseamp_df['soz'] = riseamp_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#decayamp cleanup
decayamp_df_drop1 = decayamp_df_combine
decayamp_df_drop1['soz'] = decayamp_df_combine['lateralization'] + " - " + decayamp_df_combine['region']
decayamp_df = decayamp_df_drop1.drop(columns = ['region','lateralization'])
decayamp_df['soz'] = decayamp_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#slowwidth cleanup
slowwidth_df_drop1 = slowwidth_df_combine
slowwidth_df_drop1['soz'] = slowwidth_df_combine['lateralization'] + " - " + slowwidth_df_combine['region']
slowwidth_df = slowwidth_df_drop1.drop(columns = ['region','lateralization'])
slowwidth_df['soz'] = slowwidth_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#slowamp cleanup
slowamp_df_drop1 = slowamp_df_combine
slowamp_df_drop1['soz'] = slowamp_df_combine['lateralization'] + " - " + slowamp_df_combine['region']
slowamp_df = slowamp_df_drop1.drop(columns = ['region','lateralization'])
slowamp_df['soz'] = slowamp_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#riseslope cleanup
riseslope_df_drop1 = riseslope_df_combine
riseslope_df_drop1['soz'] = riseslope_df_combine['lateralization'] + " - " + riseslope_df_combine['region']
riseslope_df = riseslope_df_drop1.drop(columns = ['region','lateralization'])
riseslope_df['soz'] = riseslope_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#decayslope cleanup
decayslope_df_drop1 = decayslope_df_combine
decayslope_df_drop1['soz'] = decayslope_df_combine['lateralization'] + " - " + decayslope_df_combine['region']
decayslope_df = decayslope_df_drop1.drop(columns = ['region','lateralization'])
decayslope_df['soz'] = decayslope_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#averageamp cleanup
averageamp_df_drop1 = averageamp_df_combine
averageamp_df_drop1['soz'] = averageamp_df_combine['lateralization'] + " - " + averageamp_df_combine['region']
averageamp_df = averageamp_df_drop1.drop(columns = ['region','lateralization'])
averageamp_df['soz'] = averageamp_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

#LL cleanup
LL_df_drop1 = LL_df_combine
LL_df_drop1['soz'] = LL_df_combine['lateralization'] + " - " + LL_df_combine['region']
LL_df = LL_df_drop1.drop(columns = ['region','lateralization'])
LL_df['soz'] = LL_df['soz'].apply(lambda x: "bilateral" if x in to_combine else x)

# %% fix the table
df_riseamp = pd.DataFrame()
df_decayamp = pd.DataFrame()
df_slowwidth = pd.DataFrame()
df_slowamp = pd.DataFrame()
df_riseslope = pd.DataFrame()
df_decayslope = pd.DataFrame()
df_averageamp = pd.DataFrame()
df_LL = pd.DataFrame()
for label in labels:
    #fix riseamp table
    df = riseamp_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'amp'})
    df_riseamp = pd.concat([df_riseamp, df], axis = 0)

    #fix decayamp table
    df = decayamp_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'amp'})
    df_decayamp = pd.concat([df_decayamp, df], axis = 0)

    #fix slowwidth table
    df = slowwidth_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'width'})
    df_slowwidth = pd.concat([df_slowwidth, df], axis = 0)

    #fix slowamp table
    df = slowamp_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'amp'})
    df_slowamp = pd.concat([df_slowamp, df], axis = 0)

    #fix riseslope table
    df = riseslope_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'slope'})
    df_riseslope = pd.concat([df_riseslope, df], axis = 0)

    #fix decayslope table
    df = decayslope_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'slope'})
    df_decayslope = pd.concat([df_decayslope, df], axis = 0)

    #fix averageamp table
    df = averageamp_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'amp'})
    df_averageamp = pd.concat([df_averageamp, df], axis = 0)
    
    #fix LL table
    df = LL_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'LL'})
    df_LL = pd.concat([df_LL, df], axis = 0)


#%%
"""
df_count = pd.DataFrame()
df_LL = pd.DataFrame()
for label in labels:
    #fix amplitude table
    df = amp_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'amp'})
    df_amp = pd.concat([df_amp, df], axis = 0)

    #fix spike count table
    df = spike_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'count'})
    df_count = pd.concat([df_count, df], axis = 0)

    #fix LL table
    df = LL_df[[label, 'soz', 'pt']]
    df = df.explode(label)
    df['roi'] = label
    df = df.rename(columns={label:'LL'})
    df_LL = pd.concat([df_LL, df], axis = 0)
    """