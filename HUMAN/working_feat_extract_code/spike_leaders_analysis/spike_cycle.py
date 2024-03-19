#%%
import pandas as pd
import numpy as np
import pywt
from pywt._doc_utils import boundary_mode_subplot
import matplotlib.pyplot as plt

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

stim_spikecounts = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/spike_leaders/stim_pts/stim_counts_perinterval.csv')
unique_pts = stim_spikecounts['filename'].unique()

for pt in unique_pts:
    subset = stim_spikecounts[stim_spikecounts['filename'] == pt]
    subset = subset.sort_values('interval_number', ascending = True)
    data = subset['total_count'].to_list()

    time = range(1, len(data)+1)
    widths = range(1, int(len(data)/4))
    sampling_frequencies = 1
    wavelet = 'morl'
    coefficients, frequencies = pywt.cwt(data, widths, wavelet, sampling_period=sampling_frequencies)
    fig = plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    plt.plot(time, data)
    plt.subplot(3,1,2)
    plt.imshow(np.abs(coefficients), extent=[min(time), max(time), min(frequencies), max(frequencies)], cmap='jet', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.subplot(3,1,3)

    power_spectrum = (np.abs(coefficients)) ** 2
    # Compute derivative of the periodogram
    periodogram_derivative = np.gradient(power_spectrum)
    # Find peaks based on positive-to-negative zero crossings
    peaks_indices = np.where(np.diff(np.sign(periodogram_derivative)))[0]

    reconstructed_signal = pywt.waverec(coefficients, wavelet)
    plt.plot(time, reconstructed_signal)





 # %%
