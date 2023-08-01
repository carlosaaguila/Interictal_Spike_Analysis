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
import random

#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *


#%% 
#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)

#%%
#all the spikes in the SOZ
SOZ_spikes = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/intra_SOZ_v1/SOZ_all_chs_stacked_DF.csv')

# %% 
# find a series of minima and maxima for a spike
#gives indices

"""
randomlist = random.sample(range(1, 8000), 20)

for idx in randomlist:
    myspike = SOZ_spikes.iloc[idx].to_numpy()
    print(idx)
    print(np.mean(myspike))
    myspike = myspike - np.mean(myspike)
    fig = plt.figure(figsize=(5,4))
    plt.plot(myspike[700:1300])
    plt.plot(300, myspike[1000], 'o')

#%%
maxima = sig.argrelextrema(myspike, np.greater)[0]
#find the 
minima = sig.argrelextrema(myspike, np.less)[0]

amplitude = np.argmax((myspike[750:1250])) #get the maximum amplitude of the spike

maxima2 = np.argwhere(myspike[maxima] > 0.9 * myspike[750+amplitude]) #find the maxima that are greater than 1.5 times the mean of the maximas

#%% plots the spike
# then we plot the max maximas
plt.plot(myspike)
plt.plot(maxima[maxima2], myspike[maxima[maxima2]], 'o')
"""
# %% function to find a feature
myspike = SOZ_all_chs_stacked_DF_cleaned.iloc[160].to_numpy() #using a different dataframe

def morphology_feats_v1(myspike):
    """
    function to find the morphological features of a spike
    major assumption - that the peak is closest to the spike detection

    input: myspike - the single spike to be analyzed

    output: rise_amp - the amplitude of the spike
            fall_amp - the amplitude of the spike
            rise_slope - the slope of the lienar fitted line leading to the spike peak
            fall_slope - the slope of the linear fitted line leading to the spike trough
            slow_width - the width of the slow wave
            slow_height - the height of the slow wave
    """
    #detrend the spike (make the mean = 0)
    myspike = myspike - np.mean(myspike)

    #find the peak closest to the spike detection (this will be our main reference point)

    allmaxima = sig.argrelextrema(myspike, np.greater)[0] #find all the maximas
    allminima = sig.argrelextrema(myspike, np.less)[0] #find all the minimas

    stndev = np.std(myspike)
    peaks_pos = peaks_pos = sig.find_peaks(myspike[1000-50:1000+50], height = stndev)[0]
    peaks_neg = sig.find_peaks(-1 * myspike[1000-50:1000+50], height = stndev)[0]
    peaks_pos = peaks_pos +  950
    peaks_neg = peaks_neg + 950
    combined_peaks = [peaks_pos, peaks_neg]
    combined_peaks = [x for x in combined_peaks for x in x]

    for peaks in combined_peaks:
        if (myspike[peaks] > myspike[peaks-3]) & (myspike[peaks] < myspike[peaks+3]):
            combined_peaks.remove(peaks)
        if (myspike[peaks] < myspike[peaks-3]) & (myspike[peaks] > myspike[peaks+3]):
            combined_peaks.remove(peaks)
    
    if not combined_peaks:
        peak = None
        left_point = None
        right_point = None
        slow_end = None
        slow_max = None

    else:
        if np.size(combined_peaks) > 1:
            peak_from_mid = [x - 1000 for x in combined_peaks]
            peak_idx = np.argmin(np.abs(peak_from_mid))
            peak = combined_peaks[peak_idx]
        else:
            peak_idx = np.argmax(np.abs(myspike[combined_peaks]))
            peak = combined_peaks[peak_idx]

        '''
        potentialpeak_max = np.argmin(np.abs(maxima))
        potentialpeak_min = np.argmin(np.abs(minima))
        closest_max = maxima[potentialpeak_max]
        closest_min = minima[potentialpeak_min]
        if np.abs(closest_max) > np.abs(closest_min):
            peak = closest_min+midpoint
        if np.abs(closest_min) > np.abs(closest_max):
            peak = closest_max+midpoint
        if np.abs(closest_min) == np.abs(closest_max):
            if myspike[closest_min+midpoint] > myspike[closest_max+midpoint]:
                peak = closest_min+midpoint
            else:
                peak = closest_max+midpoint
        '''

        #find the left and right points

        #from peak we will navigate to either baseline or the next minima/maxima
        #here we will trim down the potential left/right peaks/troughs to the 5 closest to the peak
        if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
            left_points_trim = allmaxima[allmaxima < peak][-3::]
            right_points_trim = allmaxima[allmaxima > peak][0:3]
        if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
            left_points_trim = allminima[allminima < peak][-3::]
            right_points_trim = allminima[allminima > peak][0:3]

        left_points_trim2 = []
        right_points_trim2 = []
        for i, (left, right) in enumerate(zip(left_points_trim, right_points_trim)):
            if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
                if myspike[left] > 0.5 * myspike[peak]:
                    left_points_trim2.append(left)
                if myspike[right] > 0.5 * myspike[peak]:
                    right_points_trim2.append(right)
            if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
                if myspike[left] < 0.5 * myspike[peak]:
                    left_points_trim2.append(left)
                if myspike[right] < 0.5 * myspike[peak]:
                    right_points_trim2.append(right)

        if not left_points_trim2:
            left_points_trim2 = [x for x in left_points_trim]
        if not right_points_trim2:
            right_points_trim2 = [x for x in right_points_trim]

        # find the closest spike with the greatest amplitude difference? try to balance this?
        left_point = []
        right_point = []
        if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
            dist_from_peak_left = (left_points_trim2 - peak)
            dist_from_peak_right = (right_points_trim2 - peak)
            #restrict what we are looking at by looking at the cloesest to the peak (50 samples from peak)
            left_points_trim2 = [x+peak for x in dist_from_peak_left if (x <= 50) & (x >= -50)]
            right_points_trim2 = [x+peak for x in dist_from_peak_right if (x <= 50) & (x >= -50)]

            #backup if it doesn't find any (e.g. wide spike)
            if not left_points_trim2:
                left_points_trim2 = [x + peak for x in dist_from_peak_left if (x <= 100) & (x >= -100)]
                if not left_points_trim2:
                    left_points_trim2 = [x for x in left_points_trim]
            if not right_points_trim2:
                right_points_trim2 = [x + peak for x in dist_from_peak_right if (x <= 100) & (x >= -100)]
                if not right_points_trim2:
                    right_points_trim2 = [x for x in right_points_trim]

            if not left_points_trim2:
                left_point = None
                right_point = None
                
            if not right_points_trim2:
                right_point = None
                left_point = None

            value_leftpoints = myspike[left_points_trim2]
            value_rightpoints = myspike[right_points_trim2]
            left_value_oi = np.argmax(value_leftpoints)
            right_value_oi = np.argmax(value_rightpoints)
            left_point = left_points_trim2[left_value_oi]
            right_point = right_points_trim2[right_value_oi]

        if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
            dist_from_peak_left = (left_points_trim2 - peak)
            dist_from_peak_right = (right_points_trim2 - peak)
            #restrict what we are looking at by looking at the cloesest to the peak (50 samples from peak)
            left_points_trim2 = [x+peak for x in dist_from_peak_left if (x <= 50) & (x >= -50)]
            right_points_trim2 = [x+peak for x in dist_from_peak_right if (x <= 50) & (x >= -50)]

            #backup if it doesn't find any (e.g. wide spike)
            if not left_points_trim2:
                left_points_trim2 = [x + peak for x in dist_from_peak_left if (x <= 100) & (x >= -100)]
                if not left_points_trim2:
                    left_points_trim2 = [x for x in left_points_trim]
            if not right_points_trim2:
                right_points_trim2 = [x + peak for x in dist_from_peak_right if (x <= 100) & (x >= -100)]
                if not right_points_trim2:
                    right_points_trim2 = [x for x in right_points_trim]

            if not left_points_trim2:
                left_point = None
                right_point = None
                
            if not right_points_trim2:
                right_point = None
                left_point = None
                
            else: 
                value_leftpoints = myspike[left_points_trim2]
                value_rightpoints = myspike[right_points_trim2]
                left_value_oi = np.argmin(value_leftpoints)
                right_value_oi = np.argmin(value_rightpoints)
                left_point = left_points_trim2[left_value_oi]
                right_point = right_points_trim2[right_value_oi]


        #now we will look for the start and end of the aftergoing slow wave.
        #for positive peaks
        counter = 0
        if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
            right_of_right_peaks = [x for x in allmaxima if x > right_point]
            right_of_right_troughs = [x for x in allminima if x > right_point]
            slow_start = right_point

            slow_end = []
            for peaks, troughs in zip(right_of_right_peaks, right_of_right_troughs):
                if ZX(myspike[right_point:peaks]) >= 1:
                    counter += 1
                if (counter >= 1) | (np.abs(myspike[right_point]) >= 100):
                    if (((myspike[troughs] < 0) | (myspike[troughs] < myspike[right_point])) & (troughs - right_point >= 50)):
                        slow_end = troughs 
                        break

        #for negative peaks
        if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
            right_of_right_peaks = [x for x in allmaxima if x > right_point]
            right_of_right_troughs = [x for x in allminima if x > right_point]
            slow_start = right_point

            slow_end = []
            for peaks, troughs in zip(right_of_right_peaks, right_of_right_troughs):
                if ZX(myspike[right_point:peaks]) >= 1:
                    counter += 1
                if (counter >= 1) | (np.abs(myspike[right_point]) >= 100):
                    if (((myspike[peaks] > 0) | (myspike[peaks] > myspike[right_point])) & (peaks - right_point >= 50)):
                        slow_end = peaks
                        break
        
        #find slow wave peak
        if slow_end:
            #added the positive/negative bias to get the right peak of slow wave, but it seems that it doesn't work well for all spikes
            #using spike 86 as an example it doesn't work well so get back to it.
            """
            if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
                slow_len = slow_end - right_point
                local_maxes_idx = sig.argrelextrema(myspike[right_point + int(slow_len * 0.3):slow_end - int(slow_len * 0.3)], np.greater)[0]
                it = np.argmax(np.abs(myspike[right_point + int(slow_len * 0.3) + local_maxes_idx]))
                slow_max = local_maxes_idx[it] + right_point

            if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
                slow_len = slow_end - right_point
                local_mins_idx = sig.argrelextrema(myspike[right_point+ int(slow_len * 0.3):slow_end- int(slow_len * 0.3)], np.less)[0]
                it = np.argmax(np.abs(myspike[right_point + int(slow_len * 0.3) + local_mins_idx]))
                slow_max = local_mins_idx[it] + right_point

            if not slow_end:
                local_maxes_idx = sig.argrelextrema(myspike[right_point:slow_end], np.greater)[0]
                local_mins_idx = sig.argrelextrema(myspike[right_point:slow_end], np.less)[0]
                combined = np.concatenate((local_maxes_idx, local_mins_idx))
                it = np.argmax(np.abs(myspike[right_point + combined]))
                slow_max = combined[it] + right_point
            """

            #find the vertical distance between the lowest point in the slowwave and the highest
            slow_max_idx = np.argmax(myspike[right_point:slow_end]) + right_point
            slow_min_idx = np.argmin(myspike[right_point:slow_end]) + right_point
            slow_max = myspike[slow_max_idx] - myspike[slow_min_idx]
        

        if not slow_end:
            slow_end = None
            slow_max = None

        """
        In a scenario like in spike 4907 - the slow wave is not well defined yet it gives a shallow slow_end. There isn't necessarily a good deefined slow wave in this case, but we get something.
        Potential solution - could be to check that a point has crossed the right_point (thus you get maybe a counter?) 
        if the counter is set to 1, then you can start adding in values. then if it doesn't grab, then theres either no slow wave or theres another criteria?
        """
    return peak, left_point, right_point, slow_end, slow_max#, slow_max_idx, slow_min_idx

peak, left_point, right_point, slow_end, slow_max = morphology_feats_v1(myspike)

#%% code to plot single spike
plt.plot(myspike)
plt.plot(peak, myspike[peak],'x')
plt.plot(left_point, myspike[left_point], 'o')
plt.plot(right_point, myspike[right_point], 'o')
plt.plot(slow_end, myspike[slow_end], 'o', color = 'k')
plt.xlim(700, 1500)

#%% create feats dataframe USING interSOZ_analysis.py (load cleaned data from that file)
#SOZ
SOZ_feats = pd.DataFrame(columns = ['peak', 'left point', 'right point', 'slow end', 'slow max'])
for idx in range(len(SOZ_all_chs_stacked_DF_cleaned)):
    myspike = SOZ_all_chs_stacked_DF_cleaned.iloc[idx].to_numpy()
    peak, left_point, right_point, slow_end, slow_max = morphology_feats_v1(myspike)
    SOZ_feats.loc[idx] = [peak, left_point, right_point, slow_end, slow_max]

#%% create feats dataframe USING interSOZ_analysis.py (load cleaned data from that file)
#NON SOZ
SOZ_feats_nonSOZ = pd.DataFrame(columns = ['peak', 'left point', 'right point', 'slow end', 'slow max'])
for idx in range(len(nonSOZ_all_chs_stacked_DF_cleaned)):
    myspike = nonSOZ_all_chs_stacked_DF_cleaned.iloc[idx].to_numpy()
    peak, left_point, right_point, slow_end, slow_max = morphology_feats_v1(myspike)
    SOZ_feats_nonSOZ.loc[idx] = [peak, left_point, right_point, slow_end, slow_max]

# %% check on my random spikes
randitest_spikes = [2786, 1090, 900, 478, 2906, 5204, 7302, 1094, 4907]
for randi in randitest_spikes:
    myspike = SOZ_spikes.iloc[randi].to_numpy()
    peak, left_point, right_point, slow_end, slow_max = morphology_feats_v1(myspike)
    plt.figure(figsize=(7,7))
    plt.plot(myspike)
    plt.plot(peak, myspike[peak],'x')
    plt.plot(left_point, myspike[left_point], 'o')
    plt.plot(right_point, myspike[right_point], 'o')
    plt.plot(slow_end, myspike[slow_end], 'o', color = 'k')
    plt.xlim(700, 1500)

# %%
#run 10 random points from 0 to 10000 and see if it works
randi = np.random.randint(0, 10000, 10)
for randi in randi:
    myspike = SOZ_spikes.iloc[randi].to_numpy()
    peak, left_point, right_point, slow_end, slow_max = morphology_feats_v1(myspike)
    plt.figure(figsize=(7,7))
    plt.plot(myspike)
    plt.plot(peak, myspike[peak],'x')
    plt.plot(left_point, myspike[left_point], 'o')
    plt.plot(right_point, myspike[right_point], 'o')
    plt.plot(slow_end, myspike[slow_end], 'o', color = 'k')
    plt.xlim(700, 1500)
# %%
