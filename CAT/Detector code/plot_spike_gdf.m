addpath(genpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/'))
addpath(genpath('/Users/carlosaguila/Desktop/SSH MODEL/Projects/FC_toolbox/toolbox/'))

load('catspikes.mat')
load('Exp1Unfilt.mat')

cat1 = allEventData';

%%
cat1spikes = spikeStart{1};

gdfcat1 = zeros(48,1);
for i=1:length(cat1spikes)
    if size(cat1spikes{i},2) >0
    gdfcat1(i) = (cat1spikes{i}{1}(1));
    else 
    gdfcat1(i) = 1;
    end
end

elec = [1:1:48]';
gdf1 = [elec, gdfcat1];
%remove 0's
%%
spike = 2405881;
xlimval = [(spike/2000)-1 (spike/2000)+1];
show_eeg_and_spikes_OG(cat1,gdf1,2000,xlimval)
