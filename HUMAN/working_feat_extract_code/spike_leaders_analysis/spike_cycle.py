#%%
import pandas as pd
import numpy as np
import pywt
from pywt._doc_utils import boundary_mode_subplot
import matplotlib.pyplot as plt
from scipy import signal as sig

#change pathway to Volumes if you use that.
stim_spikecounts = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/spike_leaders/stim_pts/stim_counts_perinterval.csv')

unique_pts = stim_spikecounts['filename'].unique()
def z_score_normalization(data):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Z-score normalization
    normalized_data = (data - mean) / std_dev
    return normalized_data


for pt in unique_pts:
    subset = stim_spikecounts[stim_spikecounts['filename'] == pt]
    subset = subset.sort_values('interval_number', ascending = True)
    data = subset['total_count'].to_list()

    norm_data = z_score_normalization(data)

    time = np.linspace(0, len(norm_data)/(6*24), len(norm_data))

    fs = 1/(60*10) #Hz 1 / (60 seconds * 10 mins)
    dt = 1/fs
    scales = range(1, int(len(norm_data)/4)+1)
    wavelet = 'morl'
    sampling_frequencies = 1
    coefficients, frequencies = pywt.cwt(norm_data, scales = scales, wavelet = wavelet, sampling_period = dt)

    freq_to_hour = 1/frequencies * (1/60) * (1/60)
    ind = np.where((freq_to_hour <= 24+8) & (freq_to_hour >= 24-8))[0]


    band_mean = np.mean(coefficients[ind,:], axis = 0)

    hilb = sig.hilbert(coefficients[ind,:])
    hilb = np.mean(hilb, axis = 0)
    phase = np.angle((hilb))
    amp = (np.abs(hilb))


    plt.figure(figsize=(10, 5))
    plt.title(f'pt: {pt}')
    plt.plot(time, norm_data, 'k', label = 'real signal')
    # plt.plot(time, amp, 'r', alpha = 0.8, label = '24hr $\pm$ 3hr')
    plt.plot(time, phase, 'b', linestyle = '--', alpha = 0.8, label = 'phase')
    plt.plot(time, hilb, 'r', alpha = 0.8, label = '24hr $\pm$ 3hr')
    plt.legend()
    plt.show()

# %%
