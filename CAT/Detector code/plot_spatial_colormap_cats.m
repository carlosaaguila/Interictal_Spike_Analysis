addpath(genpath('/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/'))
addpath(genpath('/Users/carlosaguila/Desktop/SSH MODEL/Projects/FC_toolbox/toolbox/'))

load('catspikes.mat')
load('Exp1Unfilt.mat')

cat1 = allEventData';

%% Colormap FOR spatial arangement 
cat1_ptx_rank = [1,2,5,3,6,9,4,7,10,13,8,11,14,17,12,15,18,21,16,19,22,25,20,23,26,29,24,27,30,28,31,32];
cat2_ptx_rank = [4,3,8,2,7,12,1,6,11,16,5,10,15,20,9,14,19,24,13,18,23,28,17,22,27,32,21,26,31,25,30,29];
cat3_ptx_rank = [8,12,4,3,7,11,15,16,2,6,10,14,18,19,20,1,5,9,13,17,21,22,23,24,25,26,27,28,29,30,31,32];
cat4_ptx_rank = [2,3,1,5,6,7,8,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32];
cat4_ptx_rank = [2,3,1,5,6,7,8,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32];
colormap = winter(32);

%% Colormap FOR amplitude or whatever feature
spike = 1332759;
values = cat1(spike-(2*2000):spike+(2*2000),:);

feat1 = zeros(32,1);
for i =1:32
    feat1(i) = round(abs(max(values(:,i)-mean(values(:,i)))));
end

cvars = feat1;
n = 32;
cmap = winter(n);
cvars_map = (cvars - min(cvars))/(max(cvars)-min(cvars)) * (n-1) + 1;
% Now find the color of i-th cvars

colormap = zeros(32,3);
for i = 1:32           
colormap(i,:) = cmap(round(cvars_map(i)), :);
end
%% example on how to use new spatial color map

% we plot 2 seconds around the spike

% they are spatially oriented, so the further down you go in the plot the
% further away it is from the PTX site

% color coordinated so that the further away the more BLUE it becomes.

spike = 1332759;
show_eeg_spatial_color(cat1(spike-(2*2000):spike+(2*2000),:),2000,cat1_ptx_rank,colormap)
