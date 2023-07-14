#%% 
#set up environment
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import loadmat, savemat
import warnings
#from Interictal_Spike_Analysis.HUMAN.working_feat_extract_code.functions.ied_fx_v3 import value_basis_multiroi
warnings.filterwarnings('ignore')
import seaborn as sns
#get all functions 
import sys, os
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *


#Setup ptnames and directory
data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']
pt = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/pkl_list.csv') #pkl list is our list of the transferred data (mat73 -> pickle)
pt = pt['pt'].to_list()
blacklist = ['HUP101' ,'HUP112','HUP115','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']
ptnames = [i for i in pt if i not in blacklist] #use only the best EEG signals (>75% visually validated)


#establish ROI's
roiL_mesial = [' left entorhinal ', ' left parahippocampal ' , ' left hippocampus ', ' left amygdala ', ' left perirhinal ']
roiL_lateral = [' left inferior temporal ', ' left superior temporal ', ' left middle temporal ', ' left fusiform '] #lingual??
roiR_mesial = [' right entorhinal ', ' right parahippocampal ', ' right hippocampus ', ' right amygdala ', ' right perirhinal ']
roiR_lateral = [' right inferior temporal ', ' right superior temporal ', ' right middle temporal ', ' right fusiform ']
emptylabel = ['EmptyLabel','NaN']
L_OC = [' left inferior parietal ', ' left postcentral ', ' left superior parietal ', ' left precentral ', ' left rostral middle frontal ', ' left pars triangularis ', ' left supramarginal ', ' left insula ', ' left caudal middle frontal ', ' left posterior cingulate ', ' left lateral orbitofrontal ', ' left lateral occipital ', ' left cuneus ']
R_OC = [' right inferior parietal ', ' right postcentral ', ' right superior parietal ', ' right precentral ', ' right rostral middle frontal ', ' right pars triangularis ', ' right supramarginal ', ' right insula ', ' right caudal middle frontal ', ' right posterior cingulate ', ' right lateral orbitofrontal ', ' right lateral occipital ', ' right cuneus ']

roilist = [roiL_mesial, roiL_lateral, roiR_mesial, roiR_lateral, L_OC, R_OC, emptylabel]

# %% load the tables
df_LL = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tables/LL.csv', index_col = 0)
df_amp = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tables/amp.csv', index_col = 0)
df_count = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/working features/mixed_effect_tables/count.csv', index_col = 0)

df_amp = df_amp.dropna().reset_index(drop = True)
df_count = df_count.dropna().reset_index(drop = True)
df_LL = df_LL.dropna().reset_index(drop = True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

# %% clean up the tables to remove bilateral and rename SOZ

df_amp_v2 = df_amp[df_amp['soz'] != 'bilateral']
df_count_v2 = df_count[df_count['soz'] != 'bilateral']
df_LL_v2 = df_LL[df_LL['soz'] != 'bilateral']

dict_soz = {'right - temporal neocortical':'R_Lateral', 'right - mesial temporal':'R_Mesial', 'left - temporal neocortical':'L_Lateral', 'left - mesial temporal':'L_Mesial','right - other cortex':'R_OtherCortex', 'left - other cortex':'L_OtherCortex'}

df_amp_v2clean = pd.DataFrame()
for soz, roi in dict_soz.items():
    subdf = df_amp_v2.loc[df_amp_v2['soz'] == soz]
    subdf = subdf.replace(to_replace = dict_soz)
    #change elements in soz to 1 or 0 if they match elements in roi
    subdf['soz2'] = subdf['soz'] == subdf['roi']
    subdf['soz2'] = subdf['soz2'].astype(int) #convert to int
    df_amp_v2clean = pd.concat([df_amp_v2clean, subdf], axis = 0)


#%% Run random intercepts LMER on Amplitude DF
# w/ interaction term 
md = smf.mixedlm("amp ~ C(soz) + C(roi) + C(soz):C(roi)", df_amp, groups="pt")
mdf = md.fit()
print(mdf.summary())

# w/out interaction term 
md2 = smf.mixedlm("amp ~ C(soz) + C(roi)", df_amp, groups="pt")
mdf2 = md2.fit()
print(mdf2.summary())

#%% RUN random intercepts LMER on Amplitude DF v2
# w/ interaction term
md3 = smf.mixedlm("amp ~ C(soz2) + C(roi) + C(soz2):C(roi)", df_amp_v2clean, groups="pt")
mdf3 = md3.fit()
print(mdf3.summary())

# w/out interaction term
md4 = smf.mixedlm("amp ~ C(soz2) + C(roi)", df_amp_v2clean, groups="pt")
mdf4 = md4.fit()
print(mdf4.summary())


#%% random intercepts+ SLOPE LMER on AMP DF

# w/ interaction term
md = smf.mixedlm("amp ~ C(soz2) + C(roi)", df_amp_v2clean, groups="pt", vc_formula = {"soz2":"C(soz2)", "roi":"C(roi)"})
mdf = md.fit()
print(mdf.summary())
"""
md = smf.mixedlm("amp ~ C(soz2) + C(roi)", df_amp_v2clean, groups="pt", re_formula = '~(C(soz2)+C(roi))')
mdf = md.fit()
print(mdf.summary())
"""
# w/out interaction term

#%% random intercepts + SLOPE LMER on amp df v2

# w/ interaction term

# w/out interaction term

#%% Normalcy test for a model
model = mdf #input model you want to evaluate

#  QQ plot
fig = plt.figure(figsize = (16, 9))
ax = fig.add_subplot(111)

sm.qqplot(model.resid, dist = stats.norm, line = 's', ax = ax)

ax.set_title("Q-Q Plot")
plt.show()

# kernal density estimate plot (check for normalcy)
fig = plt.figure(figsize = (16, 9))

ax = sns.distplot(mdf.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)

ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")
plt.show()

#shapiro_wilk test
print('shapiro_wilk test')
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(model.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)

#if test is significant, the assumption of nomality for the residuals is violated
#transofrm variables, remove outliers, use non-parametric approach, rely on central limit theorem

#%% boxplot
#sort df_amp_v2clean by soz2 and roi
boxplot = df_amp_v2clean.boxplot(["amp"], by = ["roi", "soz2"],
                     figsize = (20, 13),
                     showmeans = True,
                     notch = True,
                     showfliers = True)

boxplot.set_xlabel("Categories")
boxplot.set_ylabel("Amplitude")
plt.show()

#%% scatter plot + regression lines

scatter = df_amp_v2clean.scatter(["amp"], by = ["roi", "soz2"],
                     figsize = (20, 13))
# %% plot the actual mixedLM