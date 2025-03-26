#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import custom functions
import sys, os
code_v2_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/')
sys.path.append(code_v2_path)
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *

code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

musc_gammaspikes = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/full_musc_spikes_with_gamma.csv', index_col=0)
musc_gammaspikes = musc_gammaspikes.rename(columns={'region': 'SOZ'})

hup_gammaspikes = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/full_hup_spikes_with_gamma.csv', index_col=0)
#lets put the SOZ's into number format
def hup_soz(row):
    if row['SOZ'] == 'mesial temporal':
        return int(1)
    elif row['SOZ'] == 'temporal neocortical':
        return int(2)
    elif row['SOZ'] == 'frontal':
        return int(3)
    elif row['SOZ'] == 'other cortex':
        return int(3)
    else:
        return None
hup_gammaspikes['SOZ'] = hup_gammaspikes.apply(hup_soz, axis=1) 
hup_gammaspikes = hup_gammaspikes.dropna(subset=['SOZ'])

gammaspikes = pd.concat([musc_gammaspikes, hup_gammaspikes], axis=0).drop(columns=['Right','Left','Neo','Other','Temporal','MTL','lateralization'])

#Look for rows in gammaspikes where max_gamma_power is not 0
gammaspikes = gammaspikes[gammaspikes['max_gamma_power'] != 0]
gammaspikes = gammaspikes[gammaspikes['max_gamma_power'] != 9999.5555]

#%%
# #Side note: look for duration across pt_ids
# with open("/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin", "r") as f:
#     session = Session("aguilac", f.read())
# duration_df = pd.DataFrame() 
# for filename in gammaspikes.filename.unique():
#     dataset = session.open_dataset(filename)
#     chlabels = dataset.get_channel_labels()
#     timeseries = dataset.get_time_series_details(chlabels[0])
#     eegduration = timeseries.duration/1e6 
#     id = gammaspikes[gammaspikes['filename'] == filename]['pt_id'].iloc[0]
#     df = pd.DataFrame({'pt_id': [id], 'filename': [filename], 'dur': [eegduration]})
#     duration_df = pd.concat([duration_df, df], ignore_index=True)

    

# #sum durations for each pt_id
# total_duration_per_pt = duration_df.groupby('pt_id')['dur'].sum()

# #calculate median and quartiles across pt_ids
# median_duration = total_duration_per_pt.median()
# q1_duration = total_duration_per_pt.quantile(0.25)
# q3_duration = total_duration_per_pt.quantile(0.75)

# print(f"Median duration across patients: {median_duration/3600:.2f} hours")
# print(f"25th percentile duration: {q1_duration/3600:.2f} hours") 
# print(f"75th percentile duration: {q3_duration/3600:.2f} hours")

# %%
#plot a histogram of row counts per pt_id in gammaspikes
spike_counts = gammaspikes['filename'].value_counts()
# Plot a histogram of row counts
# Plotting a bar chart
plt.figure(figsize=(8, 6))
spike_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Spike Counts Per Patient (After Filtering)', fontsize=14)
plt.xlabel('Patient ID', fontsize=12)
plt.ylabel('Number of Spikes', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
#grab the median spike_counts
median_spike_count = spike_counts.median()
#25th percentile
q1_spike_count = spike_counts.quantile(0.25)
#75th percentile
q3_spike_count = spike_counts.quantile(0.75)
print(median_spike_count, q1_spike_count, q3_spike_count)

# %%

list_of_feats = ['spike_rate','rise_amp','decay_amp','sharpness','linelen','recruitment_latency_thresh', 'spike_width','slow_width','slow_amp']
#strip letters from the channel labels
gammaspikes.channel_label = gammaspikes.channel_label.str.replace('L|R|D|A|B|C|P|H', '', regex = True)

all_spikes_avg = []
for Feat_of_interest in list_of_feats:
    gammaspikes.Feat_of_interest = abs(gammaspikes[Feat_of_interest])
    gammaspikes_avg = gammaspikes.groupby(['pt_id','channel_label','SOZ'])[Feat_of_interest].mean().reset_index()
    all_spikes_avg.append(gammaspikes_avg)

all_gammaspike_feats = all_spikes_avg[0].copy()
for i, df in enumerate(all_spikes_avg[1:], start=2):
    # Rename the 'feature' column to avoid conflicts
    df = df.rename(columns={'feature': f'feature_{i}'})
    
    # Merge the DataFrames on 'pt_id', 'channel_label', and 'SOZ'
    all_gammaspike_feats = pd.merge(all_gammaspike_feats, df, 
                         on=['pt_id', 'channel_label', 'SOZ'], 
                         how='outer')

merged_df = all_gammaspike_feats.copy()
soz_to_remove = ['temporal']

pearson_df = pd.DataFrame()
corr_df = pd.DataFrame()
slope_df = pd.DataFrame()

for Feat_of_interest in list_of_feats:
    all_spikes_avg = merged_df.pivot_table(index=['pt_id','SOZ'], columns='channel_label', values=Feat_of_interest)
    if '' in all_spikes_avg.columns:
        all_spikes_avg.drop(columns=[''], inplace=True)
    all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

    def remove_rows_by_index(pivot_table, index_values_to_remove):
        """
        Remove rows from a pivot table based on index values.

        Args:
        - pivot_table (DataFrame): The pivot table to filter.
        - index_values_to_remove (str or list): Index value(s) to remove.

        Returns:
        - DataFrame: The filtered pivot table.
        """
        if isinstance(index_values_to_remove, str):
            index_values_to_remove = [index_values_to_remove]

        mask = ~pivot_table.index.get_level_values('SOZ').isin(index_values_to_remove)
        return pivot_table[mask]

    #look to move some SOZ's around:
    all_spikes_avg = remove_rows_by_index(all_spikes_avg, soz_to_remove)

    #probably change the Frontal SOZ to other? 
    if 'frontal' in all_spikes_avg.index.get_level_values(1):
        all_spikes_avg.rename(index={'frontal': 'other cortex'}, inplace=True, level = 'SOZ')

    #reorder all_spikes_avg, so that is_mesial is decesending
    all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])

    ####################
    # 3. Plot Heatmaps #
    ####################

    sns.set_style('ticks')
    plt.figure(figsize=(10,10))

    sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)
    # sns.heatmap(all_spikes_avg, cmap = 'rocket', alpha = 1)
    plt.xlabel('Channel Number', fontsize=20)
    plt.ylabel('Patient ID', fontsize=20)
    plt.title(f'Average {Feat_of_interest} by Channel and Patient', fontsize=24)
    #change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
    plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0), fontsize=13)

    #in all_spikes_avg, get the number of 'temporal neocortical' patients
    temp_neocort_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 2])
    #in all_spikes_avg, get the number of 'temporal' patients
    temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal'])
    #same for other cortex
    other_cortex_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 3])
    #same for mesial temporal
    mesial_temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 1])

    plt.axhline(mesial_temp_pts, color='w', linewidth=2.5)
    plt.axhline(mesial_temp_pts+other_cortex_pts, color='w', linewidth=1.5, linestyle = '--')
    plt.axhline(mesial_temp_pts+other_cortex_pts+temp_pts, color='w', linewidth=1.5, linestyle = '--')
    #create a list of 48 colors
    colors = ['#E64B35FF']*mesial_temp_pts + ['#7E6148FF']*other_cortex_pts + ['#00A087FF']*temp_pts + ['#3C5488FF']*temp_neocort_pts
    for ytick, color in zip(plt.gca().get_yticklabels(), colors):
        ytick.set_color(color)

    #add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
    import matplotlib.patches as mpatches
    mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
    other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
    # temporal_patch = mpatches.Patch(color='#00A087FF', label='Temporal Patients')
    neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')

    plt.legend(handles=[mesial_patch, other_patch, neocort_patch], loc='upper right')
    sns.despine()
    # plt.close()
    
    #########################
    # Generate Correlations #
    #########################

    #find the spearman correlation of each row in all_spikes_avg
    #initialize a list to store the spearman correlation
    channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
    channel_labels = [int(x) for x in channel_labels]
    spearman_corr = []
    label = []
    for row in range(len(all_spikes_avg)):
        # #if the row has less than 8 channels, omit from analysis
        # if len(all_spikes_avg.iloc[row].dropna()) < 8:
        #     continue
        spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
        label.append(all_spikes_avg.index[row]) 

    df = pd.DataFrame(spearman_corr, columns=[f'{Feat_of_interest}_correlation', 'p-value'])
    corr_df[f'{Feat_of_interest}_corr'] = df[[f'{Feat_of_interest}_correlation']]
    corr_df['SOZ'] = [x[1] for x in label]
    corr_df['pt_id'] = [x[0] for x in label]

    # find the pearson correlation of each row in all_spikes_avg
    # initialize a list to store the spearman correlation
    pearson_corr = []
    p_label = []
    for row in range(len(all_spikes_avg)):
        # #if the row has less than 8 channels, omit from analysis
        # if len(all_spikes_avg.iloc[row].dropna()) < 8:
        #     continue
        gradient = all_spikes_avg.iloc[row].to_list()
        channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
        channel_labels = [int(x) for x in channel_labels]
        # for each nan in the graident list, remove the corresponding channel_labels
        list_to_remove = []
        for i in range(len(channel_labels)):
            if np.isnan(gradient[i]):
                list_to_remove.append(i)

        #remove list_to_remove from channel_labels and gradient
        channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
        gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]

        pearson_corr.append(stats.pearsonr(channel_labels,gradient))
        p_label.append(all_spikes_avg.index[row])

    df = pd.DataFrame(pearson_corr, columns=[f'{Feat_of_interest}_correlation', 'p-value'])
    pearson_df[f'{Feat_of_interest}_corr'] = df[[f'{Feat_of_interest}_correlation']]
    pearson_df['SOZ'] = [x[1] for x in label]
    pearson_df['pt_id'] = [x[0] for x in label]

    ### New METRIC
    # coeff_5 = []
    # coeff_10 = []
    # one_to_four = []
    # one_to_three = []
    one_to_two = []
    # one = []
    # one_to_five = []
    m_label = []

    for row in range(len(all_spikes_avg)):
        # #if the row has less than 8 channels, omit from analysis
        # if len(all_spikes_avg.iloc[row].dropna()) < 8:
        #     continue
        gradient = all_spikes_avg.iloc[row].to_list()
        channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
        channel_labels = [int(x) for x in channel_labels]
        # for each nan in the graident list, remove the corresponding channel_labels
        list_to_remove = []
        for i in range(len(channel_labels)):
            if np.isnan(gradient[i]):
                list_to_remove.append(i)

        #remove list_to_remove from channel_labels and gradient
        channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
        gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]
        m_label.append(all_spikes_avg.index[row])

        # coeff_5.append((gradient[4]-gradient[0])/len(gradient[0:5]))
        # coeff_10.append((gradient[-1]-gradient[0])/len(gradient))
        # one_to_five.append(np.mean([gradient[0],gradient[1], gradient[2], gradient[3], gradient[4]]))
        # one_to_four.append(np.mean([gradient[0],gradient[1], gradient[2], gradient[3]]))
        # one_to_three.append(np.mean([gradient[0],gradient[1], gradient[2]]))
        one_to_two.append(np.mean([np.abs(gradient[0]),np.abs(gradient[1])]))
        # one.append(np.mean([gradient[0]]))

    # df = pd.DataFrame(data = one_to_two, columns = ['one_to_two'])
    # slope_df[f'{Feat_of_interest}_coef10'] = coeff_10
    # slope_df[f'{Feat_of_interest}_four_mean'] = one_to_four
    # slope_df[f'{Feat_of_interest}_three_mean'] = one_to_three
    df = pd.DataFrame(data = one_to_two, columns = ['one_to_two'])
    slope_df[f'{Feat_of_interest}_two_mean'] = one_to_two
    # slope_df[f'{Feat_of_interest}_first'] = one
    # slope_df[f'{Feat_of_interest}_five_mean'] = one_to_five

    slope_df['SOZ'] = [x[1] for x in m_label]
    slope_df['pt_id'] = [x[0] for x in m_label]

    #remove the temporal patients for this plot (corr_df and pearson_df)
    # corr_df = corr_df[corr_df['SOZ'] != 'temporal']
    # pearson_df = pearson_df[pearson_df['SOZ'] != 'temporal']

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

from plot_soz_correlations import *
#%%
plt.rcParams['font.family'] = 'Arial'
effect_sizes = plot_soz_correlations(
    pearson_df,
    ['spike_rate_corr'],
    x_labels=[''],
    title='Spike Rate Directionality w/ Gamma Spikes',
    figsize = (10,6),
    path = '/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/figures/gamma-spikes/spikerate_corr.pdf'
)
print("Effect Sizes for spike rate:\n", effect_sizes)
print('Cliffs D', effect_sizes[1])

#%%
effect_sizes = plot_soz_correlations(
    pearson_df,
    ['recruitment_latency_thresh_corr'],
    x_labels=[''],
    title='Timing Directionality w/ Gamma Spikes',
    figsize = (10,6),
    path = '/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/figures/gamma-spikes/timing-corr.pdf'
    )
print("Effect Sizes for timing:\n", effect_sizes)
print('Cliffs D', effect_sizes[1])

#%%
effect_sizes = plot_soz_correlations(
    pearson_df,
    ['rise_amp_corr','sharpness_corr','spike_width_corr'],
    x_labels=['Rise Amp','Sharpness','Spike Width'],
    title='Morphology Directionality w/ Gamma Spikes',
    figsize = (10,6),
    path = '/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/figures/gamma-spikes/base-morph.pdf'
)
print("Effect Sizes:\n", effect_sizes[0])
print('Cliffs D', effect_sizes[1])


print(pearson_df.groupby('SOZ')['rise_amp_corr'].describe())
print(pearson_df.groupby('SOZ')['spike_width_corr'].describe())
#%%
# Perform Kruskal-Wallis test for each morphology feature
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Only test morphology features
# features = ['rise_amp_corr', 'sharpness_corr', 'spike_width_corr']
features = ['spike_rate_corr','recruitment_latency_thresh_corr']
print("\nKruskal-Wallis Test Results:")
print("-" * 50)

# Store p-values
p_values = []

# First pass to collect p-values
for feature in features:
    # Create groups based on SOZ
    mtle = pearson_df[pearson_df['SOZ'] == 1][feature]
    neo = pearson_df[pearson_df['SOZ'] == 2][feature]
    other = pearson_df[pearson_df['SOZ'] == 3][feature]
    
    # Perform Kruskal-Wallis H-test
    h_stat, p_val = stats.kruskal(mtle, neo, other)
    p_values.append(p_val)

# Correct p-values using Benjamini-Hochberg
rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# Print results with corrected p-values
for i, feature in enumerate(features):
    mtle = pearson_df[pearson_df['SOZ'] == 1][feature]
    neo = pearson_df[pearson_df['SOZ'] == 2][feature]
    other = pearson_df[pearson_df['SOZ'] == 3][feature]
    
    h_stat, _ = stats.kruskal(mtle, neo, other)
    
    print(f"\n{feature}:")
    print(f"H-statistic: {h_stat:.3f}")
    print(f"Original p-value: {p_values[i]:.3e}")
    print(f"Corrected p-value: {p_corrected[i]:.3e}")
    print(f"Significant: {rejected[i]}")

#%%
effect_sizes = plot_soz_correlations(
    pearson_df,
    ['decay_amp_corr','linelen_corr','slow_width_corr','slow_amp_corr'],
    # x_labels=['Decay Amp','Line Length','Slow Width','Slow Amp','Rise Slope','Decay Slope','Average Amp','Rise Duration','Decay Duration'],
    title='Morphology Directionality w/ Gamma Spikes',
    figsize = (15,7),
    comparisons_correction=None,
    rotation = 45
)
print("Effect Sizes:\n", effect_sizes)

# %%
#plot some spikes
from ieeg.auth import Session

musc_gammaspikes = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/musc_spikes_with_gamma.csv', index_col=0)
musc_gammaspikes = musc_gammaspikes.rename(columns={'region': 'SOZ'})

hup_gammaspikes = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_spikes_with_gamma.csv', index_col=0)
#lets put the SOZ's into number format
def hup_soz(row):
    if row['SOZ'] == 'mesial temporal':
        return int(1)
    elif row['SOZ'] == 'temporal neocortical':
        return int(2)
    elif row['SOZ'] == 'frontal':
        return int(3)
    elif row['SOZ'] == 'other cortex':
        return int(3)
    else:
        return None
hup_gammaspikes['SOZ'] = hup_gammaspikes.apply(hup_soz, axis=1) 
hup_gammaspikes = hup_gammaspikes.dropna(subset=['SOZ'])

gammaspikes = pd.concat([musc_gammaspikes, hup_gammaspikes], axis=0).drop(columns=['Right','Left','Neo','Other','Temporal','MTL','lateralization'])

#Look for rows in gammaspikes where max_gamma_power is not 0
gammaspikes = gammaspikes[gammaspikes['max_gamma_power'] != 0]

#%%



train = gammaspikes.sample(n=1)
yo = gammaspikes[gammaspikes.filename == 'HUP126_phaseII_D02']
check = yo[yo.channel_label == 'LDA1']
for X in range(len(check)):
    print(X)
    train = pd.DataFrame(check.iloc[X]).T
    # train = pd.DataFrame(gammaspikes.iloc[129]).T
    filename = train['filename'].iloc[0]


    #Load in the spike.
    with open("/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin", "r") as f:
        session = Session("aguilac", f.read())

    dataset = session.open_dataset(filename)

    all_channel_labels = np.array(dataset.get_channel_labels())

    #change the sequence_index == X for a different peak_index
    ch_labels = all_channel_labels[electrode_selection(all_channel_labels)]

    fs = int(dataset.get_time_series_details(dataset.ch_labels[0]).sample_rate)  # get sample rate

    #find a minute of data around the spike train we want.
    lower_bound = (train['peak_time_usec'])
    upper_bound = (train['peak_time_usec'])

    ieeg_data, fs = get_iEEG_data(
                                "aguilac",
                                "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin",
                                filename,
                                (lower_bound) - (2 * 1e6),
                                (upper_bound) + (2 * 1e6),
                                ch_labels
                            )

    fs = int(fs)

    #look for bad channels
    good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
    good_channel_indicies = good_channels_res[0]
    good_channel_labels = ch_labels[good_channel_indicies]
    ieeg_data = ieeg_data[good_channel_labels]
    good_channel_labels = [decompose_labels(x, train['pt_id'].iloc[0]) for x in good_channel_labels]
    ieeg_data.columns = good_channel_labels
    signal = ieeg_data[train['channel_label']]

    #apply bandpass filter
    ieeg_data = notch_filter(signal, 60, fs)
    signal_filtered = bandpass_filter((ieeg_data), 1, 100, fs, order=4)

    try:
        gamma_filt = bandpass_filter((ieeg_data), 100,500,fs,order = 4)
    except:
        gamma_filt = bandpass_filter(ieeg_data, 100, (fs/2) -1, fs,order = 4)
    plt.figure(figsize = (10,5))
    plt.plot(signal_filtered,'k')
    plt.plot(gamma_filt+200, 'r')
    plt.show()  

#good locs: 
# 949286
# 22, 
# %%
