# %%
# standard imports
import os
from os.path import join as ospj
from itertools import combinations

# third party imports
import numpy as np
import pandas as pd
#from tqdm import tqdm

# local imports
#from config import CONFIG

# %%
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
emptylabel = ['EmptyLabel','NaN']
L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']

roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]

roi_list = {
    "roiL_mesial": roiL_mesial,
    "roiL_lateral": roiL_lateral,
    "roiR_mesial": roiR_mesial,
    "roiR_lateral": roiR_lateral,
    "L_OC": L_OC,
    "R_OC": R_OC,
    "emptylabel": emptylabel,
}

# strip all strings in dict of list of strings
roi_list = {k: [s.strip() for s in v] for k, v in roi_list.items()}
roi_list.pop("emptylabel")

# %%
master_elecs = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/master_elecs.csv', index_col=0
)
# %%
# get number of patients who have electrodes in each roi
pt_per_roi = {}
for roi, elecs in roi_list.items():
    pt_per_roi[roi] = master_elecs[master_elecs["label"].isin(elecs)].index.nunique()

pt_per_roi['total'] = master_elecs.index.nunique()
display(pt_per_roi)
# %%
master_elecs_soz = master_elecs[master_elecs["soz"] == True]
master_elec_nsoz = master_elecs[master_elecs["soz"] == False]

# repeat for every combination of roi
roi_combos = []
for i in range(len(roi_list.keys())):
    roi_combos += list(combinations(roi_list.keys(), i + 1))

for master, name in zip([master_elecs_soz, master_elec_nsoz, master_elecs], ["soz", "nsoz", "all"]):
    # make a dict of the number of patients who have sampling of at least one region in each roi_combo
    pt_per_roi_combo = {}
    for roi_combo in roi_combos:
        # patient must have at least one electrode in each element of roi_combo
        pts = []
        for roi in roi_combo:
            pts.append(master[master["label"].isin(roi_list[roi])].index.unique())

        # get intersection of all pts
        pts = set.intersection(*map(set, pts))
        pt_per_roi_combo[roi_combo] = len(pts)

    # make a df
    pt_per_roi_combo = pd.DataFrame.from_dict(pt_per_roi_combo, orient="index")
    # add a row for total
    pt_per_roi_combo.loc["total"] = master.index.nunique()

    # display('soz' if master is master_elecs_soz else 'nsoz')
    display(name)
    display(pt_per_roi_combo)
    print()

# %% find patients who have (roiL_mesial and roiR_mesial) and (roiL_lateral and roiR_lateral)

left = ("roiL_mesial", "roiL_lateral")
right = ("roiR_mesial", "roiR_lateral")

left_pts = []
for roi in left:
    left_pts.append(master_elecs[master_elecs["label"].isin(roi_list[roi])].index.unique())
left_pts = set.intersection(*map(set, left_pts))

right_pts = []
for roi in right:
    right_pts.append(master_elecs[master_elecs["label"].isin(roi_list[roi])].index.unique())
right_pts = set.intersection(*map(set, right_pts))

#%% same as above but we want to find out coverage of left mesial/temporal and right mesial/temporal
mesials = ("roiL_mesial", "roiR_mesial")
laterals = ("roiL_lateral", "roiR_lateral")

mesial_pts = []
for roi in mesials:
    mesial_pts.append(master_elecs[master_elecs["label"].isin(roi_list[roi])].index.unique())
mesial_pts = set.intersection(*map(set, mesial_pts))

lateral_pts = []
for roi in laterals:
    lateral_pts.append(master_elecs[master_elecs["label"].isin(roi_list[roi])].index.unique())
lateral_pts = set.intersection(*map(set, lateral_pts))

# %%
# load carlos pts and rid_hup_table
carlos_pts = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/soz_locations.csv', index_col=0)
# drop rows with na
carlos_pts.dropna(inplace=True)
display(
    carlos_pts.groupby(["region", "lateralization"]).count().index.sort_values(
        ascending=False
    ).to_frame()
)
# if index has HUP, then use hup_to_rid to convert to RID
# if index has MP, then use musc_to_rid to convert to RID
# use map function

def map_fn(x):
    if "HUP" in x:
        if x in CONFIG.hup_to_rid:
            return CONFIG.hup_to_rid[x]
        else:
            return "NA"
    elif "MP" in x:
        if x in CONFIG.musc_to_rid:
            return CONFIG.musc_to_rid[x]
        else:
            return "NA"
    else:
        return "NA"


carlos_pts['rid'] = carlos_pts.index.map(map_fn)
# how many na (check first element of tuple)
print(f"Number of NA: {sum(carlos_pts.rid == 'NA')}")

# what are the na
display(carlos_pts[carlos_pts.rid == 'NA'])

# drop na
carlos_pts = carlos_pts[carlos_pts.rid != 'NA']

# how many pts from carlos are in master elec
print(f"Number of pts from carlos in master elec: {sum(carlos_pts.rid.isin(master_elecs.index))}")

print("Pts not in master elec:")
# print pts from carlos that are not in master elec
display(carlos_pts[~carlos_pts.rid.isin(master_elecs.index)])

# drop pts from carlos that are not in master elec
carlos_pts = carlos_pts[carlos_pts.rid.isin(master_elecs.index)]

# do value counts on region and lateralization together
display(
    carlos_pts.groupby(["region", "lateralization"]).count()["rid"].sort_values(
        ascending=False
    ).to_frame()
)

# %%