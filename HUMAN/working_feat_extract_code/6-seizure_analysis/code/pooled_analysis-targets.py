#%%
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
import ast

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


#%%
#Lets reorganize the pearson_all_df
pearson_6 = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/supplement-HFER/HFER-6-pearson.csv', index_col = 0)
pearson_11 = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/supplement-HFER/HFER-11-pearson.csv', index_col = 0)
pearson_16 = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/supplement-HFER/HFER-16-pearson.csv', index_col = 0)
pearson_21 = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/supplement-HFER/HFER-21-pearson.csv', index_col = 0)

#%%
# Function to prepare each dataframe
def prepare_df(df, time):
    return df[['SOZ', 'pt_id', 'correlation']].rename(columns={'correlation': f'correlation-{time}s'})

# Prepare each dataframe
df_6 = prepare_df(pearson_6, 6)
df_11 = prepare_df(pearson_11, 11)
df_16 = prepare_df(pearson_16, 16)
df_21 = prepare_df(pearson_21, 21)

# Merge all dataframes
merged_df = df_6.merge(df_11, on=['SOZ', 'pt_id'], how='outer')\
               .merge(df_16, on=['SOZ', 'pt_id'], how='outer')\
               .merge(df_21, on=['SOZ', 'pt_id'], how='outer')

#%%
#PEARSON PLOTS MORPHOLOGY
plt.rcParams['font.family'] = 'Arial'
merged_df['SOZ'] = merged_df['SOZ'].astype('category')

new_metrics = ['correlation-6s', 'correlation-11s', 'correlation-16s','correlation-21s']
# Melt the dataframe to long format
melted_pearson_df = merged_df.melt(id_vars='SOZ', 
                              value_vars=new_metrics,
                              var_name='Metric', value_name='Value')

# Set up the matplotlib figure
fig, ax = plt.subplots(1,1, figsize=(10,6))

my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
fig_args = {'x':'Metric',
            'y':'Value',
            'hue':'SOZ',
            'data':melted_pearson_df,
            'order': new_metrics,
            'hue_order':[1,2,3]}

# Generate significance comparisons for all pairs of SOZ values for each metric
significanceComparisons = []
for metric in new_metrics:
    significanceComparisons.extend([
        ((metric, 1), (metric, 3)),
        ((metric, 1), (metric, 2)),
        ((metric, 2), (metric, 3))
    ])


sns.boxplot(ax=ax, showfliers = False, palette=my_palette, **fig_args)
sns.stripplot(ax =ax, color = 'k', alpha = 0.5, dodge=True, jitter=True, size=5, **fig_args)

annotator = Annotator(ax=ax, pairs=significanceComparisons,
                    **fig_args, plot='boxplot')

# Assign Mann-Whitney U test p-values to the annotator
test = 'Mann-Whitney'
comp = 'BH' #benjamani hochberg correction
configuration = {'test':test,
                    'comparisons_correction':None,
                    'text_format':'star',
                    'loc':'inside',
                    'verbose':True,
                    'hide_non_significant':True}
annotator.configure(**configuration)
annotator.apply_and_annotate()

# Set plot title and labels
plt.title('Distribution of Pearson Correlation by SOZ Type', fontsize = 30)

new_labels = ['6s', '11s', '16s','21s']
ax.set_xticklabels(new_labels, fontsize=20)
plt.ylabel('Correlation Coef.', fontsize =20)
ax.set(xlabel=None)

# Update the legend to prevent duplication
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], ['mTLE', 'Neo', 'Other'], loc='upper right', fontsize=20, bbox_to_anchor=(1.05, 1))

# Show the plot
sns.despine()
plt.axhline(y=0, color='k', linestyle='--')
plt.savefig("/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/figures/supplement/HFER-allwindows.pdf")
plt.show()

all_effect_szs = []
for comparison in significanceComparisons:
    print(comparison)
    metric, soz1 = comparison[0]
    _, soz2 = comparison[1]
    group1 = melted_pearson_df[(melted_pearson_df['Metric'] == metric) & (melted_pearson_df['SOZ'] == soz1)]['Value']
    group2 = melted_pearson_df[(melted_pearson_df['Metric'] == metric) & (melted_pearson_df['SOZ'] == soz2)]['Value']
    all_effect_szs.append([metric, soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

#%%
from scipy.stats import f_oneway, levene, shapiro, kruskal

for yo in new_metrics:
    x = melted_pearson_df[melted_pearson_df['Metric'] == yo]
    k1,tp1 = kruskal(x[x['SOZ'] == 1]['Value'],x[x['SOZ'] == 2]['Value'],x[x['SOZ'] == 3]['Value'])
    print(f'{yo} p:',tp1)
    print('K = ', k1)
    print('---------------------')


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import scipy.stats as stats
import pandas as pd
import numpy as np

def cohend(d1, d2):
    # Calculating Cohen's d
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(d1) - np.mean(d2)) / s

# Prepare data
time_windows = ['6s', '11s', '16s', '21s']
data_for_plot = []

for time in time_windows:
    temp_df = merged_df[['SOZ', f'correlation-{time}']].copy()
    temp_df['Time Window'] = time
    temp_df.rename(columns={f'correlation-{time}': 'correlation'}, inplace=True)
    data_for_plot.append(temp_df)

plot_df = pd.concat(data_for_plot)
plot_df = plot_df.dropna()  # Remove NaN values

# Set up the plot
plt.figure(figsize=(16, 10))
plt.rcParams['font.family'] = 'Arial'

my_palette = {1: '#E64B35FF', 3: '#7E6148FF', 2: '#3C5488FF'}
pairs = [(1, 2), (1, 3), (2, 3)]
order = [1, 2, 3]

# Create boxplot
ax = sns.boxplot(x='SOZ', y='correlation', hue='Time Window', data=plot_df, palette='Set2', 
                 order=order, showfliers=False)
sns.stripplot(x='SOZ', y='correlation', hue='Time Window', data=plot_df, dodge=True, alpha=0.3, 
              jitter=True, color='black')

plt.axhline(y=0, color='k', linestyle='--')

# Adjust labels and title
plt.xlabel('SOZ Type', fontsize=14)
plt.ylabel('Pearson Correlation', fontsize=14)
plt.title('EI Directionality - All Time Windows', fontsize=18)
plt.xticks(range(3), ['Mesial Temporal', 'Neocortical', 'Other Cortex'], fontsize=12)
plt.yticks(np.arange(-1, 1.1, 0.5), fontsize=12)
plt.ylim([-1, 1])

# Adjust legend
plt.legend(title='Time Window', fontsize=12, title_fontsize=14)

sns.despine()

# Calculate effect sizes and perform statistical tests
for time in time_windows:
    print(f"\nAnalysis for {time}:")
    time_df = plot_df[plot_df['Time Window'] == time]
    
    all_effect_szs = []
    for comparison in pairs:
        soz1, soz2 = comparison
        group1 = time_df[time_df['SOZ'] == soz1]['correlation']
        group2 = time_df[time_df['SOZ'] == soz2]['correlation']
        effect_size = cohend(group1, group2)
        all_effect_szs.append([f'Pearson Corr EI {time}', soz1, soz2, effect_size])
    
    print("Effect sizes:")
    print(all_effect_szs)
    
    # Perform Kruskal-Wallis H-test and one-way ANOVA
    groups = [time_df[time_df['SOZ'] == soz]['correlation'] for soz in order]
    kruskal_result = stats.kruskal(*groups)
    anova_result = stats.f_oneway(*groups)
    
    print("Statistical tests:")
    print("Kruskal-Wallis H-test:", kruskal_result)
    print("One-way ANOVA:", anova_result)

plt.tight_layout()
# plt.savefig('EI_Pearson_correlation_combined.pdf', bbox_inches='tight')
plt.show()

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming merged_df is your dataframe with columns: SOZ, pt_id, correlation-6s, correlation-11s, correlation-16s, correlation-21s

# Extract correlation columns
correlation_columns = ['correlation-6s', 'correlation-11s', 'correlation-16s', 'correlation-21s']
correlation_data = merged_df[correlation_columns]

# Calculate Pearson correlation between correlation columns
correlation_matrix = correlation_data.corr(method='pearson')

# Plot heatmap of the correlation matrix with all values displayed
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,  # This ensures all numbers are displayed
            cmap='viridis', 
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.2f',  # Display 2 decimal places
            square=True,  # Make sure the cells are square
            cbar_kws={'label': 'Pearson Correlation'})

plt.title('Pearson Correlation Between Correlation Values of Different Time Windows', fontsize=30)
plt.tight_layout()
# plt.savefig('full_correlation_of_correlations_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.show()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

print("\nP-values:")
print(p_values)