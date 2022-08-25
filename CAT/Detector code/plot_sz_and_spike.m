% NEW CODE TO PLOT THE WHOLE CAT DATA - THEN USE A LIMITER TO ZOOM INTO A SZ - 
% THEN SEE IF OUR DOTS (WHERE PEAKS/SPIKES OCCUR) LINE UP WITH START TIMES

% we start by importing a cat - create script to plot multiple lines

addpath(genpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/'))
addpath(genpath('/Users/carlosaguila/Desktop/SSH MODEL/Projects/FC_toolbox/toolbox/'))
%addpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/cat_seqs')
%addpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/cat_values')
%addpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/Bink_files')


load('gdf_array_cats.mat')
load('Exp5Unfilt.mat') %change for cat
load('sequences.mat')
load('catspikes.mat')

cat5 = allEventData'; %change for cat

gdf_c1 = gdf_array_cats{1};
gdf_c2 = gdf_array_cats{2};
gdf_c3 = gdf_array_cats{3};
gdf_c4 = gdf_array_cats{4};
gdf_c5 = gdf_array_cats{5};

gdf_c1=double(gdf_c1);
gdf_c2=double(gdf_c2);
gdf_c3=double(gdf_c3);
gdf_c4=double(gdf_c4);
gdf_c5=double(gdf_c5);

gdf_c1=[ gdf_c1(:,1)+1,gdf_c1(:,2) ];
gdf_c2=[ gdf_c2(:,1)+1,gdf_c2(:,2) ];
gdf_c3=[ gdf_c3(:,1)+1,gdf_c3(:,2) ];
gdf_c4=[ gdf_c4(:,1)+1,gdf_c4(:,2) ];
gdf_c5=[ gdf_c5(:,1)+1,gdf_c5(:,2) ];

gdf_c1 = sortrows(gdf_c1,2);
gdf_c2 = sortrows(gdf_c2,2);
gdf_c3 = sortrows(gdf_c3,2);
gdf_c4 = sortrows(gdf_c4,2);
gdf_c5 = sortrows(gdf_c5,2);



%% adjust for the cat could have extra electrodes
to_plot = zeros(64,1);
to_plot(1:32,:) = 1; % grid only eeg

%% plot
fs=2000;
xlimval = [2500 2700];
show_eeg_and_spikes_OG(cat5,gdf_c5,fs,xlimval)

%go to sequences (seqs_c1) --> plot in those xlims to figure out where it
%is that you need to look at for seizures....

%MY SEQUENCES —> CONTAIN THE FIRST STARTING SPIKE IN A SEIZURE ACROSS ALL
%ELECTRODES TO THE LAST STARTING SPIKE IN A SEIZURE ACROSS ALL ELECTRODES!!

%THUS THEY ARE DIFFERENT FROM REGULAR SEIZURES BECAUSE THEY DO NOT CONTAIN 
%EVERY SPIKE IN A SEIZURE ACROSS ALL ELECTRODES, JUST THE ONSET SPIKE. 

%THEREFORE WE ARE ONLY MEASURING THE SEIZURE ONSET
