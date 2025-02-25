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

list_of_feats = ['spike_rate', 'rise_amp','decay_amp','sharpness','linelen','recruitment_latency_thresh','spike_width','slow_width','slow_amp', 'rise_slope','decay_slope','average_amp','rise_duration','decay_duration']
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

def plot_soz_correlations(df, value_columns, soz_column='SOZ', 
                         x_labels=None, title='Distribution of Pearson Correlation by SOZ Type',
                         figsize=(15,6), palette=None, y_label='Correlation Coef.',
                         comparisons_correction='Benjamini-Hochberg', rotation=0):
    """
    Creates boxplots with statistical annotations and calculates effect sizes for SOZ type comparisons
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    value_columns (list): List of column names to plot
    soz_column (str): Name of SOZ column (default: 'SOZ')
    x_labels (list): Custom x-axis labels (default: None)
    title (str): Plot title
    figsize (tuple): Figure size
    palette (dict): Color palette for SOZ categories
    y_label (str): Y-axis label
    
    Returns:
    pd.DataFrame: DataFrame containing effect sizes for all comparisons
    """
    
    # Set default palette if not provided
    if not palette:
        palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
    
    # Prepare data
    df = df.copy()
    df[soz_column] = df[soz_column].astype('category')
    melted_df = df.melt(id_vars=soz_column, 
                        value_vars=value_columns,
                        var_name='Metric', 
                        value_name='Value')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    plot_params = {
        'x': 'Metric',
        'y': 'Value',
        'hue': soz_column,
        'data': melted_df,
        'order': value_columns,
        'hue_order': sorted(df[soz_column].cat.categories)
    }

    # Generate significance comparisons
    significance_comparisons = []
    for metric in value_columns:
        for pair in [(1,3), (1,2), (2,3)]:
            significance_comparisons.append(((metric, pair[0]), (metric, pair[1])))

    # Create plots
    sns.boxplot(ax=ax, showfliers=False, palette=palette, **plot_params)
    sns.stripplot(ax=ax, color='k', alpha=0.5, dodge=True, 
                  jitter=True, size=5, **plot_params)

    # Add statistical annotations
    annotator = Annotator(ax=ax, pairs=significance_comparisons, **plot_params)
    annotator.configure(
        test='Mann-Whitney',
        comparisons_correction=comparisons_correction,
        text_format='star',
        loc='inside',
        verbose=True,
        hide_non_significant=False
    )
    annotator.apply_and_annotate()

    # Set labels and formatting
    plt.title(title, fontsize=28)
    ax.set_xticklabels(x_labels if x_labels else value_columns, 
                      fontsize=20, ha='center',rotation=rotation)
    plt.ylabel(y_label, fontsize=20)
    ax.set(xlabel=None)
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    legend_labels = ['mTLE', 'Neo', 'Other']  # Update these if needed
    ax.legend(handles[:3], legend_labels, loc='center left', fontsize=20, bbox_to_anchor=(1.05, 0.5), frameon=False)

    # Final touches
    sns.despine()
    plt.axhline(y=0, color='k', linestyle='--')
    plt.tight_layout()
    
    # Calculate effect sizes
    effect_sizes = []
    for comparison in significance_comparisons:
        metric, soz1 = comparison[0]
        _, soz2 = comparison[1]
        group1 = melted_df[(melted_df['Metric'] == metric) & 
                          (melted_df[soz_column] == soz1)]['Value']
        group2 = melted_df[(melted_df['Metric'] == metric) & 
                          (melted_df[soz_column] == soz2)]['Value']
        effect_size = cohend(group1, group2)
        effect_sizes.append({
            'Metric': metric,
            'Group1': soz1,
            'Group2': soz2,
            'Effect_Size': effect_size
        })

    effect_df = pd.DataFrame(effect_sizes)
    
    plt.show()
    
    return effect_df

#%%
plt.rcParams['font.family'] = 'Arial'
effect_sizes = plot_soz_correlations(
    pearson_df,
    ['spike_rate_corr'],
    x_labels=[''],
    title='Spike Rate Directionality w/ Gamma Spikes',
    figsize = (10,6)
)
print("Effect Sizes for spike rate:\n", effect_sizes)

effect_sizes = plot_soz_correlations(
    pearson_df,
    ['recruitment_latency_thresh_corr'],
    x_labels=[''],
    title='Timing Directionality w/ Gamma Spikes',
    figsize = (10,6)
)
print("Effect Sizes for timing:\n", effect_sizes)

effect_sizes = plot_soz_correlations(
    pearson_df,
    ['rise_amp_corr','sharpness_corr','spike_width_corr'],
    x_labels=['Rise Amp','Sharpness','Spike Width'],
    title='Morphology Directionality w/ Gamma Spikes',
    figsize = (10,6)
)
print("Effect Sizes:\n", effect_sizes)

effect_sizes = plot_soz_correlations(
    pearson_df,
    ['decay_amp_corr','linelen_corr','slow_width_corr','slow_amp_corr','rise_slope_corr','decay_slope_corr','average_amp_corr','rise_duration_corr','decay_duration_corr'],
    x_labels=['Decay Amp','Line Length','Slow Width','Slow Amp','Rise Slope','Decay Slope','Average Amp','Rise Duration','Decay Duration'],
    title='Morphology Directionality w/ Gamma Spikes',
    figsize = (15,7),
    comparisons_correction=None,
    rotation = 45
)
print("Effect Sizes:\n", effect_sizes)

# %%
