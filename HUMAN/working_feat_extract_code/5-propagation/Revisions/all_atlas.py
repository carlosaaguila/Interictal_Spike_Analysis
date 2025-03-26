#%%
import pandas as pd
from glob import glob
import os
from os.path import join as ospj
import numpy as np
import seaborn as sns


# Define input paths
hup_rids_path = ospj('/users', 'aguilac', 'CNTSurgicalRepositor-CarlosUpdatedOutcome_DATA_LABELS_2024-09-04_1518.csv')
ids_we_have_path = "/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/data/mtl_all_feats.csv"
collected_elecs_path = "/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/collected_electrode2ROI/"

# Load patient information
hup_rids = pd.read_csv(hup_rids_path)[['Record ID', 'HUP Number']].dropna().rename(columns={'HUP Number': 'pt_id'})
ids_we_have = pd.read_csv(ids_we_have_path, index_col=0).pt_id.str.replace('HUP', '').str.replace(" 3T_MP00", "").str.replace('3T_MP00', '').astype(int).unique()
rids_we_want = hup_rids[hup_rids.pt_id.isin(ids_we_have)]

final_ids = np.array([ 89, 272, 238, 274, 278, 320, 294, 295, 301, 309, 382, 412, 325,
       442, 371, 440, 139, 490, 332, 522, 520, 405, 454, 530, 582, 646,
       596, 588, 589, 647, 566, 677, 695, 621, 785, 617, 700])

rids_we_want = rids_we_want[rids_we_want['Record ID'].isin(final_ids)]
# Collect atlas files
atlas_dkt = glob(ospj(collected_elecs_path, '*'))
atlas_dkt_filtered = []
kept_rids = []

# Filter atlas files based on patient IDs
for rid in rids_we_want['Record ID'].unique():
    rid_str = f"RID{str(rid).zfill(4)}"
    matching_files = [f for f in atlas_dkt if rid_str in f]
    
    # Prefer 'ieeg_recon' over 'ieeg_recon_old' files
    recon_files = [f for f in matching_files if 'ieeg_recon/' in f]
    recon_old_files = [f for f in matching_files if 'ieeg_recon_old/' in f]
    if len(recon_files) > 0:
        matching_files = recon_files
        
    if matching_files:
        kept_rids.append(rid)
        atlas_dkt_filtered.extend(matching_files)

atlas_dkt = atlas_dkt_filtered
print(f"Kept {len(kept_rids)} RIDs: {sorted(kept_rids)}")

# Combine atlas information
big_table = pd.DataFrame()

for i, atlas_file in enumerate(atlas_dkt):
    atlas_df = pd.read_csv(atlas_file)[['labels', 'roi']]
    rid_oi = kept_rids[i]
    atlas_df['Record ID'] = rid_oi
    big_table = pd.concat([big_table, atlas_df], axis=0)

# Preprocessing roi column
def preprocess_roi(roi_label):
    # Keep only part of the label after the last "-"
    last_dash_split = roi_label.split("-")[-1]
    # Remove unwanted labels
    if last_dash_split.lower() in ['matter', 'brain']:
        return None
    return last_dash_split

big_table['roi_cleaned'] = big_table['roi'].apply(preprocess_roi)
big_table = big_table.dropna(subset=['roi_cleaned'])  # Drop rows with None after cleaning
#%%
chs_tokeep = ['RA','LA','RDA','LDA','LDH','RDH','LHD', 'RHD','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']
big_table = big_table[big_table['labels'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)
big_table = big_table[~big_table['labels'].str.contains('T|F|P|RCC|RCA|RCB|Z')].reset_index(drop=True)

# Calculate median and IQR of contacts per region per patient
roi_stats = big_table.groupby(['roi_cleaned', 'Record ID']).size().reset_index(name='contact_count')
roi_summary = roi_stats.groupby('roi_cleaned').agg(
    median_contacts=('contact_count', 'median'),
    percentile_25_contacts=('contact_count', lambda x: np.percentile(x, 25)),
    percentile_75_contacts=('contact_count', lambda x: np.percentile(x, 75)),
    patient_count=('Record ID', 'nunique')
).reset_index()

# Sort and select top 10 regions
roi_summary_sorted = roi_summary.sort_values(by='median_contacts', ascending=False).head(10)

# Display the table
print("Top 10 regions with the most electrode contacts (median, IQR, and patient count):")
display(roi_summary_sorted)


# %%
