#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session

# Import custom functions
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *

import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

#load all the filenames (long form IEEG filenames)
filenames_w_ids = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/filenames_w_ids.csv').drop(columns = ['whichpt'])
#load the list of patients to exclude
blacklist = ['HUP101' ,'HUP112','HUP115','HUP119','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
# remove the patients in the blacklist from filenames_w_ids
filenames_w_ids = filenames_w_ids[~filenames_w_ids['hup_id'].isin(blacklist)]

#show me all the rows that their filename ends in D02
filenames_w_ids[filenames_w_ids['filename'].str.endswith('D02')]

#take the following comment and make these changes to the large dataframe, to reduce time spent.
"""
#keep only the filenames ending in D01 and D02 for hup_id = HUP099
filenames_w_ids = filenames_w_ids[((filenames_w_ids['filename'].str.endswith('D01')) | (filenames_w_ids['filename'].str.endswith('D02'))) & (filenames_w_ids['hup_id'] == 'HUP099')]
#keep only the filenames ending in D02 for hup_id = HUP100
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP100')]
#keep only filename ending in D01 for hup_id = HUP110
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP110')]
#keep only filename ending in D01 for hup_id = HUP111
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP111')]
#keep only filename ending in D01 for hup_id = HUP113
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP113')]
#remove all rows for hup_id = HUP117
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP117']
#remove all rows for hup_id = HUP118
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP118']
#remove all rows for hup_id = HUP122
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP122']
#keep only filename ending in D01 for hup_id = HUP123
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP123')]
#keep only filename ending in D02 for hup_id = HUP126
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP126')]
#keep only filename ending in D02 for hup_id = HUP128
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP128')]
#remove all rows for hup_id = HUP129
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP129']
#remove all rows for hup_id = HUP132 -- get back to this when you talk to alfredo
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP132']
#keep only the filenames ending in D01 for hup_id = HUP134
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP134')]
#remove all rows for hup_id = HUP137
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP137']
#keep only the filenames ending in D02 for hup_id = HUP140
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP140')]
#keep only the filenames ending in D02 for hup_id = HUP148
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP148')]
#remove all rows for hup_id = HUP152
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP152']
#keep only the filenames ending in D01 for hup_id = HUP153
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP153')]
#keep only the filenames ending in D01 for hup_id = HUP156
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP156')]
#keep only the filenames ending in D01 for hup_id = HUP159
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP159')]
#keep only the filenames ending in D02 for hup_id = HUP167
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP167')]
#remove all rows for hup_id = HUP168 #no images for this patient
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP168']
#keep only the filenames ending in D01 for hup_id = HUP179
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP179')]
#keep only the filenames ending in D01 for hup_id = HUP181
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP181')]
#keep only the filenames ending in D02 for hup_id = HUP197
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D02')) & (filenames_w_ids['hup_id'] == 'HUP197')]
#remove all rows for hup_id = HUP201
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP201']
#keep only the filenames ending in D01 for hup_id = HUP209
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP209')]
#keep only the filenames ending in D01 for hup_id = HUP213
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP213')]
#remove all rows for hup_id = HUP214 - this one has electrode names but no record on my end again
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP214']
#keep only the filenames ending in D01 and D02 for hup_id = HUP215
filenames_w_ids = filenames_w_ids[((filenames_w_ids['filename'].str.endswith('D01')) | (filenames_w_ids['filename'].str.endswith('D02'))) & (filenames_w_ids['hup_id'] == 'HUP215')]
#keep only the filenames ending in D01 for hup_id = HUP218
filenames_w_ids = filenames_w_ids[(filenames_w_ids['filename'].str.endswith('D01')) & (filenames_w_ids['hup_id'] == 'HUP218')]
#remove hup098 - no images
filenames_w_ids = filenames_w_ids[filenames_w_ids['hup_id'] != 'HUP098']
"""



#split filenames_w_ids dataframe into 10 dataframes
pt_files_split = np.array_split(filenames_w_ids, 10)


#%% load the session
print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())

#%%

#pick a list of patients to run
pt_to_use = pt_files_split[0]

