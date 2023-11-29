#%%
# initialize packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats

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

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

#%%
# Load in the data
all_spikes = pd.read_csv('dataset/intra_mtle/all_spikes.csv')

# %%
spike_tolookat = all_spikes[~all_spikes['final_label'].isna()]
spikes_tolookat = spike_tolookat[spike_tolookat['final_label'].str.contains('hippo|amyg|rhinal')]
spikes_tolookat = spikes_tolookat[~spikes_tolookat['soz'].isna()]

# %%

median_feats_per_region = spikes_tolookat.groupby(['pt_id', 'soz']).median().reset_index()

# %%
#create box plot looking at the difference between False/True for soz
import seaborn as sns
import matplotlib.pyplot as plt
feature = 'sharpness'
#arial font
plt.rcParams['font.family'] = "Arial"
#size 12
plt.rcParams['font.size'] = 12
sns.set(style="whitegrid")
plt.figure(figsize=(10, 10))
median_feats_per_region['sharpness'] = median_feats_per_region['sharpness'].abs()
#make rise_slope and decay_slope absolute values
median_feats_per_region['rise_slope'] = median_feats_per_region['rise_slope'].abs()
median_feats_per_region['decay_slope'] = median_feats_per_region['decay_slope'].abs()

my_palette = {False:'#E64B35FF', True:'#3C5488FF'}
ax = sns.boxplot(x="soz", y=feature, data=median_feats_per_region, palette=my_palette)
ax.set_title(f'{feature} in SOZ vs. non-SOZ for MTLE patients', fontsize = 16)
ax.set_ylabel(f'{feature}')
ax.set_xlabel('SOZ')

#save
# plt.savefig('figures/intra_SOZ/mtle_sharpness_boxplot.png', dpi = 300, bbox_inches = 'tight')

#%% create multi box plot

# %%
feature = 'linelen'
median_feats = median_feats_per_region[['pt_id','soz',f'{feature}']]
SOZ_feats = median_feats[median_feats['soz'] == True]
nonSOZ_feats = median_feats[median_feats['soz'] == False]
#drop soz column
SOZ_feats = SOZ_feats.drop(columns = ['soz'])
nonSOZ_feats = nonSOZ_feats.drop(columns = ['soz'])
#rename columns
SOZ_feats = SOZ_feats.rename(columns = {f'{feature}':'SOZ'})
nonSOZ_feats = nonSOZ_feats.rename(columns = {f'{feature}':'non-SOZ'})
#merge
median_feats = pd.merge(SOZ_feats, nonSOZ_feats, on = 'pt_id')
#create a 'color' column, that is the difference between SOZ and non-SOZ, then change it to a 1 if it's >0 and a 0 if it's <0
median_feats['color'] = median_feats['SOZ'] - median_feats['non-SOZ']
median_feats['color'] = np.where(median_feats['color'] > 0, 1, 0)

fig, ax = plt.subplots(1,1, figsize = (10,10))
ax.scatter(median_feats[median_feats['color'] == 1]['SOZ'], median_feats[median_feats['color'] == 1]['non-SOZ'], color = 'r', label = "Patients w/ SOZ > ({})".format(len(median_feats[median_feats['color'] == 1])))
ax.scatter(median_feats[median_feats['color'] == 0]['SOZ'], median_feats[median_feats['color'] == 0]['non-SOZ'], color = 'b', label = "Patients w/ non-SOZ > ({})".format(len(median_feats[median_feats['color'] == 0])))
ax.set_xlabel(f'SOZ - {feature}')
ax.set_ylabel(f'nonSOZ - {feature}')
ax.set_title('SOZ vs. non-SOZ {}'.format(feature))
lims = [
np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend()

#save
plt.savefig(f'figures/intra_SOZ/MTLE/pairedplots/{feature}.png', dpi = 300, bbox_inches = 'tight')
# %%
