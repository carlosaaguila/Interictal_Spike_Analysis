
#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

#load all the filenames (long form IEEG filenames)
filenames_w_ids = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
#load the list of patients to exclude
blacklist = ['HUP101' ,'HUP112','HUP115','HUP119','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176',
             'HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071',
             'HUP072','HUP073','HUP085','HUP094', 'HUP173', 'HUP211','HUP224'] #check why HUP173/HUP211/HUP224 didn't make it through the pipeline

# remove the patients in the blacklist from filenames_w_ids
filenames_w_ids = filenames_w_ids[~filenames_w_ids['hup_id'].isin(blacklist)]
#only keep rows where the column "to use" is a 1
filenames_w_ids = filenames_w_ids[filenames_w_ids['to use'] == 1].reset_index(drop=True)

#%%
"""
all_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/spike_output_DF_leads_all_v3')
#show me where the nan's are for each patient
all_df[all_df['final_label'].isna()]['pt_id'].unique()
# which patient has the most nan's?
all_df[all_df['final_label'].isna()]['pt_id'].value_counts()

#%%
#load patient data
pt_id = 'HUP123'
pt_df = all_df[all_df['pt_id'] == pt_id]
pt_df = pt_df.reset_index(drop=True)

#find all the unique channels in channel_label
channels = pt_df['channel_label'].unique() #channels in pt_df
#using load_ptall, load this patients data
spike, brain_df, _, _ = load_ptall(pt_id, data_directory)
#try to correct
brain_df['key_0'] = brain_df['key_0'].apply(lambda x: decompose_labels(x, pt_id))
channels2 = brain_df['key_0'].unique() #channels in brain_df

#%%
#find difference between channels and channels2
#channels not in brain_df but in pt_df
missing_channels_1 = [i for i in channels if i not in channels2]
print(missing_channels_1)
#channels not in pt_df but in brain_df
missing_channels_2 = [i for i in channels2 if i not in channels]
print(missing_channels_2)
# %%
#find which channels are missing from channel_label in pt_df and key_0 in brain_df
#chanels not in brain_df but in pt_df
missing_channels_1 = [i for i in pt_df['channel_label'].unique() if i not in brain_df['key_0'].unique()]
print(missing_channels_1)
#channels not in pt_df but in brain_df
missing_channels_2 = [i for i in brain_df['key_0'].unique() if i not in pt_df['channel_label'].unique()]
print(missing_channels_2)
"""

#%%

#we want to add a isSOZ column to all_df
filenames_w_ids = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv')
#load the master elecs file
master_elecs = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/master_elecs.csv')
#add hup_id by merging on r_id to master_elecs 
all_elecs = master_elecs.merge(filenames_w_ids[['r_id', 'hup_id']], left_on='rid', right_on = 'r_id', how='left')
#drop duplicate rows
all_elecs = all_elecs.drop_duplicates(subset=['rid', 'name'])

#decompose labels to all_elecs using name and hup_id as inputs
all_elecs['clean_labels'] = all_elecs.apply(lambda x: decompose_labels(x['name'], x['hup_id']), axis=1)

# %%
all_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/spike_output_DF_leads_all_v3')

#drop all "EmptyLabel" rows in all_df
all_df = all_df[all_df['final_label'] != 'EmptyLabel']
#%%
all_elecs = all_elecs[['hup_id', 'clean_labels', 'soz', 'label']]
# %%
#merge all_df with all_elecs on pt_id and channel_label
all_df_2 = all_df.merge(all_elecs, left_on=['pt_id', 'channel_label'], right_on=['hup_id', 'clean_labels'], how='left')
all_df_2 = all_df_2.drop(columns=['hup_id', 'clean_labels','label'])
all_df_2 = all_df_2.dropna()
# %%
#for each pt_id, find which value of final_label when soz is True
SOZ_labels = all_df_2[all_df_2['soz'] == True].groupby('pt_id')['final_label'].unique().to_dict()

all_df_3 = pd.DataFrame()
for key, value in SOZ_labels.items():
    #find all the rows in all_df_2 where pt_id == key and final_label is in value
    df = all_df_2[(all_df_2['pt_id'] == key) & (all_df_2['final_label'].isin(value))]
    #drop final_label == 'Empty Label'
    df = df[df['final_label'] != 'EmptyLabel']
    all_df_3 = all_df_3.append(df)

#%%
#save this dataframe as intra_soz_leaders.csv in working features/clean_spikeleads
#all_df_3.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/clean_spikeleads/intra_soz_leaders.csv', index=False)

# %%
