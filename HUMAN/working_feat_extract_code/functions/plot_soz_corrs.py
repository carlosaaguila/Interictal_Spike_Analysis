import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
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

import numpy as np
from scipy.stats import mannwhitneyu
from pandas import Series

def cliffsd(d1: Series, d2: Series) -> float:
    """
    Takes 2 series from pandas and returns Cliff's delta
    """
    # Convert Series to numpy arrays and remove NaN values
    x = d1.dropna().to_numpy()
    y = d2.dropna().to_numpy()
    
    # Calculate the size of samples
    nx, ny = len(x), len(y)
    
    # Calculate Mann-Whitney U
    U, _ = mannwhitneyu(x, y, alternative='two-sided')
    
    # Calculate Cliff's delta
    delta = ((2 * U) / (nx * ny)) - 1
    
    return delta

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
    effect_sizes2 = []
    for comparison in significance_comparisons:
        metric, soz1 = comparison[0]
        _, soz2 = comparison[1]
        group1 = melted_df[(melted_df['Metric'] == metric) & 
                          (melted_df[soz_column] == soz1)]['Value']
        group2 = melted_df[(melted_df['Metric'] == metric) & 
                          (melted_df[soz_column] == soz2)]['Value']
        effect_size = cohend(group1, group2)
        cliffs_d = cliffsd(group1, group2)
        effect_sizes.append({
            'Metric': metric,
            'Group1': soz1,
            'Group2': soz2,
            'Effect_Size': effect_size
        })
        effect_sizes2.append({
            'Metric': metric,
            'Group1': soz1,
            'Group2': soz2,
            'Effect_Size': cliffs_d
        })
    effect_df = pd.DataFrame(effect_sizes)
    cliff_effect = pd.DataFrame(effect_sizes2)
    
    plt.show()
    
    return effect_df, cliff_effect