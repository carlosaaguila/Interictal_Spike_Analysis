#%%
#Load data

import pandas as pd
import numpy as np
from scipy import stats

combined_preds = pd.read_csv('ML_results/LR_elastic/Combined_predictions.csv').rename(columns = {"Predicted_Probability":"combined_predprob"})
interictal_preds = pd.read_csv('ML_results/LR_elastic/Interictal_predictions.csv').rename(columns = {"Predicted_Probability":"interictal_predprob"})
ictal_preds = pd.read_csv('ML_results/LR_elastic/Ictal_predictions.csv').rename(columns = {"Predicted_Probability":"ictal_predprob"})

merged_preds_v1 = combined_preds.merge(interictal_preds[['Patient_ID','interictal_predprob']], on='Patient_ID', how = "left")
merged_preds = merged_preds_v1.merge(ictal_preds[['Patient_ID','ictal_predprob']], on = 'Patient_ID', how = "inner")

#NEW outcomes - look through them

pec_outcomes = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pec_trimmed_outcomes.xlsx', index_col = 0)

# %%

Engel_good = ['IA','IB','IC','ID']

def calculate_outcome(row):
    if pd.isna(row['engel_f1']):
        return row[' ILAE-v1']
    elif str(row['engel_f1']) in Engel_good:
        return 1
    else:
        return 0

# Apply the function to each row to create the outcome3 column
pec_outcomes['engel_outcomes_12m'] = pec_outcomes.apply(calculate_outcome, axis=1)

def calculate_outcome_2yr(row):
    if pd.isna(row['engel_f2']):
        return row[' ILAE-v1']
    elif str(row['engel_f2']) in Engel_good:
        return 1
    else:
        return 0

pec_outcomes['engel_outcomes_24m'] = pec_outcomes.apply(calculate_outcome_2yr, axis=1)

#%%
#merge the outcomes w/ the predprob
combined_outcomes = combined_preds.merge(pec_outcomes, left_on = 'Patient_ID',right_on = 'hup_id', how = 'right').dropna(subset = 'combined_predprob')
interictal_outcomes = interictal_preds.merge(pec_outcomes, left_on = 'Patient_ID',right_on = 'hup_id', how = 'right').dropna(subset = 'interictal_predprob')
interictal_outcomes = ictal_preds.merge(pec_outcomes, left_on = 'Patient_ID',right_on = 'hup_id', how = 'right').dropna(subset = 'ictal_predprob')

# %%
#graph for combined_outcomes
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#Pearson Correlation PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_12m', y='combined_predprob', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_12m", y="combined_predprob", data=combined_outcomes, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_12m", y="combined_predprob", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('mTLE Model Probability', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 12m', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_12m.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz1]['combined_predprob']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz2]['combined_predprob']

    all_effect_szs.append(['Outcome 12m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

#%%
#Do the same analysis for 24m
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_24m', y='combined_predprob', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_24m", y="combined_predprob", data=combined_outcomes, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_24m", y="combined_predprob", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('mTLE Model Probability', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 24m', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_24m,.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz1]['combined_predprob']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz2]['combined_predprob']

    all_effect_szs.append(['Outcome 24m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

# %%

# Model Prob vs. the true label

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='True_Label', y='combined_predprob', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="True_Label", y="combined_predprob", data=combined_outcomes, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=combined_outcomes, x="True_Label", y="combined_predprob", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('True_Label', fontsize=12)
plt.ylabel('mTLE Model Probability', fontsize=12)
#change the x-tick labels to be more readable
# plt.xticks(np.arange(2), ['Good', 'Bad'], fontsize = 12)
# plt.yticks(fontsize = 12)

#part to change
plt.title(f'Model Prob vs. the true label', fontsize=16)
sns.despine()
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['True_Label'] == soz1]['combined_predprob']
    group2 = combined_outcomes[combined_outcomes['True_Label'] == soz2]['combined_predprob']

    all_effect_szs.append(['Outcome 24m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)
# %%
# Want to plot raw correlation feature vs. outcome
def load_data():
    pearson_df = pd.read_csv('dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col=0)
    pearson_df['SOZ'] = pearson_df['SOZ'].replace({2: 0, 3: 0})
    pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)

    EI_df = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/ML_data/EI_pearson_final1.csv', index_col=0)[['correlation', 'pt_id']]
    EI_df['pt_id'] = EI_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '').astype(int)

    combined_df = pearson_df.merge(EI_df, on='pt_id')
    combined_df = combined_df.dropna(subset='correlation')

    return combined_df, EI_df, pearson_df

combined_df, EI_df, pearson_df = load_data()

combined_outcomes = combined_outcomes.merge(combined_df, left_on='hup_id', right_on = 'pt_id', how = 'left')

#%%
plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_12m', y='spike_rate_corr', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_12m", y="spike_rate_corr", data=combined_outcomes, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_12m", y="spike_rate_corr", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('spike rate', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 12m vs. Spike Rate', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_12m_vs_spikerate.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz1]['spike_rate_corr']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz2]['spike_rate_corr']

    all_effect_szs.append(['Outcome 12m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_12m', y='correlation', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_12m", y="correlation", data=combined_outcomes, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_12m", y="correlation", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('HFER', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 12m vs. HFER', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_12m_vs_hfer.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz1]['correlation']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz2]['correlation']

    all_effect_szs.append(['Outcome 12m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_12m', y='rise_amp_corr', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_12m", y="rise_amp_corr", data=combined_outcomes, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_12m", y="rise_amp_corr", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('Rise Amp', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 12m vs. Rise Amp', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_12m_vs_riseamp.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz1]['rise_amp_corr']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz2]['rise_amp_corr']

    all_effect_szs.append(['Outcome 12m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)
#%%
plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_24m', y='spike_rate_corr', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_24m", y="spike_rate_corr", data=combined_outcomes, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_24m", y="spike_rate_corr", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('spike rate', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 24m vs. Spike Rate', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_24m_vs_spikerate.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz1]['spike_rate_corr']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz2]['spike_rate_corr']

    all_effect_szs.append(['Outcome 24m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_24m', y='correlation', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_24m", y="correlation", data=combined_outcomes, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_24m", y="correlation", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('HFER', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 24m vs. HFER', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_24m_vs_hfer.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz1]['correlation']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz2]['correlation']

    all_effect_szs.append(['Outcome 24m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 2:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1)]
order = [0,1]

ax = sns.boxplot(x='engel_outcomes_24m', y='rise_amp_corr', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_24m", y="rise_amp_corr", data=combined_outcomes, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_24m", y="rise_amp_corr", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Engel Outcome', fontsize=12)
plt.ylabel('Rise Amp', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 24m vs. Rise Amp', fontsize=16)
sns.despine()
plt.savefig('ML_results/LR_elastic/outcomes_24m_vs_riseamp.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz1]['rise_amp_corr']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_24m'] == soz2]['rise_amp_corr']

    all_effect_szs.append(['Outcome 24m cohens d:', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)
