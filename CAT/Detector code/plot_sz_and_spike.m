% NEW CODE TO PLOT THE WHOLE CAT DATA - THEN USE A LIMITER TO ZOOM INTO A SZ - 
% THEN SEE IF OUR DOTS (WHERE PEAKS/SPIKES OCCUR) LINE UP WITH START TIMES

% we start by importing a cat - create script to plot multiple lines

addpath(genpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/'))
addpath(genpath('/Users/carlosaguila/Desktop/SSH MODEL/Projects/FC_toolbox/toolbox/'))
%addpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/cat_seqs')
%addpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/cat_values')
%addpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/Bink_files')


load('gdf_array_cats.mat')
load('Exp1Unfilt.mat') %change for cat
load('sequences.mat')
load('catspikes.mat')

cat1 = allEventData'; %change for cat

%% adjust for the cat could have extra electrodes
to_plot = zeros(48,1);
to_plot(1:32,:) = 1; % grid only eeg

%% plot
show_eeg_and_spikes_select(cat1,gdf_array_cats{1},2000,to_plot,1)
xlim([1988339 2013308])

%go to sequences (seqs_c1) --> plot in those xlims to figure out where it
%is that you need to look at for seizures....