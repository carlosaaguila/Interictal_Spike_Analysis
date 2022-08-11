
%add paths for the unfiltered data and the event detector
SZdetector = '/gdrive/public/USERS/binkh/Depth and Surface Testing/Paper 1 Seizure Model/';
data = '/gdrive/public/USERS/binkh/Depth and Surface Testing/Spike Analysis/All Seizure Data/';
addpath('/gdrive/public/USERS/binkh/Depth and Surface Testing/Paper 1 Seizure Model/Data/')
addpath(data)
addpath(genpath(data))
addpath(SZdetector)
addpath(genpath(SZdetector))

load('AllExpTimingSec.mat'); load('dtds.mat'); load('allExpThElecData.mat')
cat1 = load('Exp1Filt.mat'); cat2 = load('Exp2Filt.mat'); cat3 = load('Exp3Filt.mat'); cat4 = load('Exp4Filt.mat'); cat5 = load('Exp5Filt.mat'); 
cat1 = cat1.allSzDataFiltExp1; cat2 = cat2.allSzDataFiltExp2; cat3 = cat3.allSzDataFiltExp3; cat4 = cat4.allSzDataFiltExp4; cat5 = cat5.allSzDataFiltExp5;
cats = {cat1,cat2,cat3,cat4,cat5};

%initilize the variables
sdThresh =130;
featStartSecExp = [350 310 660 160 470];
featEndSecExp = [2550.85 7245 3227 7252 6929];
featStart= {};
featEnd= {};
for exper=1:5
featStart{exper} = find(allExpTimingSec{exper}>featStartSecExp(exper),1);
featEnd{exper} = find(allExpTimingSec{exper}>featEndSecExp(exper),1);
end

%run the detector for each channel in each cat
nchns = zeros(1,5);
nchns(1) = size(cat1,1);
nchns(2) = size(cat2,1);
nchns(3) = size(cat3,1);
nchns(4) = size(cat4,1);
nchns(5) = size(cat5,1);

evStart={};
evStop = {};
spikeStart ={};
spikeStop = {};
numSpikes={};
sdThresh = 130;
for exper = 1:5
    for detChanData = 1:nchns(exper)
        [evStart{exper}{detChanData}, evStop{exper}{detChanData}, spikeStart{exper}{detChanData}, spikeStop{exper}{detChanData}, numSpikes{exper}{detChanData}] = eventDetector(cats{exper}(detChanData,:), dtds(exper), featStart{exper}(1), featEnd{exper}(1), sdThresh);
    end
end

catspikes.evStart = evStart;
catspikes.evStop = evStop;
catspikes.spikeStart = spikeStart;
catspikes.spikeStop = spikeStop;
catspikes.numSpikes = numSpikes;
pathway = '/gdrive/public/USERS/aguilac/Projects/cat_data/catspikes_filtered.mat';
save(pathway, '-struct', 'catspikes_filtered')

disp('saved catspikes')