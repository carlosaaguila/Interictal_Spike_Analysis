#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df_hup= pd.read_csv('/mnt/sauce/littlab/users/slavelle/iEEG_Atlas/Tables/Master_Table_EIs/Manual_validation_seizures.csv')
# df_musc =  pd.read_csv('/mnt/sauce/littlab/users/slavelle/EI/EI_Carlos/MUSC_seizure_times.csv') Actually don't need this? Those patients are already in hup
EI_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/ML_data/EI_pearson_final1.csv', index_col=0)[['correlation', 'pt_id']]
EI_df.pt_id.unique()
# Clean the MUSC IDs (they start with 3T)
EI_df_pts_clean = []
for id in EI_df.pt_id.unique():
    if '3T' in id:
        print(id)
        x = id.split('_')[1]
        EI_df_pts_clean.append(x)
    else:
        EI_df_pts_clean.append(id)
df_hup = df_hup.dropna(subset='start')
# df_musc = df_musc.dropna(subset='Onset time')
# df_hup_uniques = df_hup.Patient.unique()
# df_musc_uniques = df_musc.File.unique()
# Filter fot pts we want
df_hup_filtered = df_hup[df_hup['Patient'].isin(EI_df_pts_clean)]
# Count the number of seizures per patient
seizure_counts = df_hup_filtered['Patient'].value_counts()
# Create a histogram based on the number of seizures per patient
plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(8, 6))
seizure_counts.plot(kind='bar', color='#00A087FF')
plt.title('Number of Seizures per Patient', fontsize=14)
plt.xlabel('Patient', fontsize=12)
plt.ylabel('Number of Seizures', fontsize=12)
# plt.xticks(rotation=90)
sns.despine()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/Revisions/figures/seizure-count/number_perpt.pdf')

plt.show()
# %%
