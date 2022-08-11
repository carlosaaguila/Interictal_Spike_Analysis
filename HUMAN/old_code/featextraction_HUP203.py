#establishing environment
!pip install -U numpy
!pip install -U scipy

import numpy as np
import scipy
from scipy.io import loadmat, savemat


split_1 = loadmat('/gdrive/public/USERS/aguilac/Projects/FC_toolbox/results/mat_output/split_1.mat')
fs = 512;

def mean_max(sequence_split, values):
    seq_0_concat = np.concatenate(sequence_split)
    seq_0_concat = np.concatenate(seq_0_concat) #double concatenate so that everything is in a array corresponding - basically turns into GDF
    ch_uniq = np.unique(seq_0_concat[:,0]) #finds the unique channels in each run_time
    #finds the mean of the max of each channel's spike index.
    meanmax_per_ch = [];
    all_max = []
    for ch in ch_uniq:
        x = np.where(seq_0_concat[:,0] == ch)[0] #index where all spikes are per channel per run_time
        max_in_ch = []
        chs = []
        for i in x:
            val = values[seq_0_concat[i,1]-20:seq_0_concat[i,1]+20, ch] #finding value at the spike and channel from x
            val_max = np.max(np.abs(val))
            max_in_ch.append(val_max)
            chs.append(np.unique(ch))
        meanmax_per_ch.append(np.mean(max_in_ch))
        all_max.append([np.concatenate(chs),max_in_ch])
        
    chs2 = (np.concatenate(np.transpose(all_max)[0]))
    maxs2 = (np.concatenate(np.transpose(all_max)[1]))
    all_max_2 = [chs2,maxs2] #reshape of maxs in all channels.
    return meanmax_per_ch, ch_uniq, all_max_2

def ALL_mean_max(split_1): #input would be the complete matrix assuming 'values_all' and 'seqs_all' are the base names
    mean_max_ALL = []
    ch_uniq_ALL = []
    max_I_ALL = []
    
    for I in range(len(split_1['values_all'][0])):
        values_gdf_I = split_1['values_all'][0,I]
        seq_I = split_1['seqs_all'][0,I]
        mean_max_seq_I, ch_uniq_I, all_max_I = mean_max(seq_I,values_gdf_I)
        mean_max_ALL.append(mean_max_seq_I)
        ch_uniq_ALL.append(ch_uniq_I)
        max_I_ALL.append(all_max_I)
    
    ch_uniq_AL_C = np.concatenate(ch_uniq_ALL)
    mean_max_ALL_C = np.concatenate(mean_max_ALL)
    ALL_CH = []
    ALL_maxvalues = []
    for s in range(len(max_I_ALL)):
        ALL_CH.append((max_I_ALL[s][0]))
        ALL_maxvalues.append((max_I_ALL[s][1]))
    ALL_CH = np.concatenate(ALL_CH)
    ALL_maxvalues = np.concatenate(ALL_maxvalues)
    max_I_ALL = [ALL_CH,ALL_maxvalues]
    ALL_mean_max_with_ch = [ch_uniq_AL_C,mean_max_ALL_C]
    return ALL_mean_max_with_ch, max_I_ALL

#creates a compilation of every channel in all gdfs and there respective means of max absolute peak values 
def meanofmeanmax_per_ch(split_1): #input is the complete split file.
    all_mean_max, max_I_all = ALL_mean_max(split_1) #uses ALL_mean_max function to get you a complete list of concatenated mean max's

    #code to get means of means per channel.
    all_mean_max = np.transpose(all_mean_max)
    max_I_all = np.transpose(max_I_all)
    popmean= np.nanmean(max_I_all[:,1])
    popstd = np.nanstd(max_I_all[:,1])
    ch_uniq = np.unique(all_mean_max[:,0]) #finds the unique channels in concatenated list
    ch_uniq_ALL = np.unique(max_I_all[:,0])
    
    means = []
    means_from_all_maxes = []
    std_from_all_maxes = []
    for ch in ch_uniq:
        x = np.where(all_mean_max[:,0] == ch)[0]#index where all spikes are per channel per run_time
        means.append(np.mean(all_mean_max[x,1]))
    for ch2 in ch_uniq_ALL:
        x2 = np.where(max_I_all[:,0] == ch2)[0]
        means_from_all_maxes.append(np.mean(max_I_all[x2,1]))
        std_from_all_maxes.append(np.std(max_I_all[x2,1]))
    meanofmeanmax = [ch_uniq,means]    
    stats_per_ch = [ch_uniq_ALL, means_from_all_maxes, std_from_all_maxes]
    
    return np.transpose(meanofmeanmax), np.transpose(stats_per_ch), popmean, popstd # ['channel','mean of mean max'] ['channel', 'mean of maxes for all channels', 'std of maxes for all channels']

#test for significance
from scipy import stats as st

#perform 1-sample t test
def stu_ttest_per_chn(split_1):
    _ , max_I_all = ALL_mean_max(split_1)
    _ , stats_ch_ALL, popmean, popstd = meanofmeanmax_per_ch(split_1)
    max_I_all = np.transpose(max_I_all)
    ch_uniq_ALL = np.unique(max_I_all[:,0])
    stats_per_chn = []
    for ch2 in ch_uniq_ALL:
        x2 = np.where(max_I_all[:,0] == ch2)[0]
        stats = st.ttest_1samp(a=max_I_all[x2,1], popmean=popmean)
        stats_per_chn.append(stats)
    stats_per_chn_labeled = [ch_uniq_ALL, stats_per_chn];
    return np.transpose(stats_per_chn_labeled)


x = stu_ttest_per_chn(split_1)

np.save('/gdrive/public/USERS/aguilac/Projects/FC_toolbox/results/mat_output/t_test_output.npy',x)