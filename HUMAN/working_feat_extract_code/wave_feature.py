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

# %% function to find a feature
spike_idx = 2906
myspike = SOZ_spikes.iloc[spike_idx].to_numpy()
testspike = SOZ_spikes.iloc[1000].to_numpy()

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
    detected_val = myspike[len(myspike)//2] #the value of the spike at the middle of the spike
    allmaxima = sig.argrelextrema(myspike, np.greater)[0] #find all the maximas
    allminima = sig.argrelextrema(myspike, np.less)[0] #find all the minimas

    midpoint = len(myspike)//2 #midpoint of spike
    maxima_idx_from_mid = allmaxima - midpoint #find the index of the maxima from the middle of the spike
    minima_idx_from_mid = allminima - midpoint #find the index of the minima from the middle of the spike

    maxima = [max for max in maxima_idx_from_mid if (max > -100) & (max < 100)]
    minima = [min for min in minima_idx_from_mid if (min > -100) & (min < 100)]
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

    #find the rise and fall amplitudes
    #from peak we will navigate to either baseline or the next minima/maxima
    #here we will trim down the potential left/right peaks/troughs to the 5 closest to the peak
    if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
        left_points_trim = allmaxima[allmaxima < peak][-5::]
        right_points_trim = allmaxima[allmaxima > peak][0:5]
    if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
        left_points_trim = allminima[allminima < peak][-5::]
        right_points_trim = allminima[allminima > peak][0:5]

    left_points_trim2 = []
    right_points_trim2 = []
    for i, (left, right) in enumerate(zip(left_points_trim, right_points_trim)):
        if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
            if myspike[left] > 0.25 * myspike[peak]:
                left_points_trim2.append(left)
            if myspike[right] > 0.25 * myspike[peak]:
                right_points_trim2.append(right)
        if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
            if myspike[left] < 0.25 * myspike[peak]:
                left_points_trim2.append(left)
            if myspike[right] < 0.25 * myspike[peak]:
                right_points_trim2.append(right)

    # find the closest spike with the greatest amplitude difference? try to balance this?
    left_point = []
    right_point = []
    if (myspike[peak + 3] > myspike[peak]) & (myspike[peak - 3] > myspike[peak]): #negative peak
        dist_from_peak_left = np.abs(left_points_trim2 - peak)
        dist_from_peak_right = np.abs(right_points_trim2 - peak)
        value_leftpoints = myspike[left_points_trim2]
        value_rightpoints = myspike[right_points_trim2]
        left_value_oi = np.argmax(value_leftpoints)
        right_value_oi = np.argmax(value_rightpoints)
        left_point = left_points_trim2[left_value_oi]
        right_point = right_points_trim2[right_value_oi]

    if (myspike[peak + 3] < myspike[peak]) & (myspike[peak - 3] < myspike[peak]): #positive peak
        dist_from_peak_left = np.abs(left_points_trim2 - peak)
        dist_from_peak_right = np.abs(right_points_trim2 - peak)
        value_leftpoints = myspike[left_points_trim2]
        value_rightpoints = myspike[right_points_trim2]
        left_value_oi = np.argmin(value_leftpoints)
        right_value_oi = np.argmin(value_rightpoints)
        left_point = left_points_trim2[left_value_oi]
        right_point = right_points_trim2[right_value_oi]


    return left_points_trim2, right_points_trim2, dist_from_peak_left, dist_from_peak_right, value_leftpoints, value_rightpoints, peak, left_point, right_point

 #rise_amp, fall_amp, rise_slope, fall_slope, slow_width, slow_height
left_points_trim2, right_points_trim2, dist_from_peak_left, dist_from_peak_right, value_leftpoints, value_rightpoints, peak, left_point, right_point = morphology_feats_v1(testspike)
print(left_points_trim2
      , right_points_trim2
      , dist_from_peak_left
      , dist_from_peak_right
      , value_leftpoints
      , value_rightpoints
      , peak
      )

# %%

plt.plot(testspike)
plt.plot(1000, testspike[1000],'x')
plt.plot(left_point, testspike[left_point], 'o')
plt.plot(right_point, testspike[right_point], 'o')
# %%
