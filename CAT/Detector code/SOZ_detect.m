% We will find the SOZ
% this is quantifiable by the leading electrode in a seizure event.

addpath("/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Cat/Bink_files/")
load('catspikes.mat')

%%
seizure_leader = {};
for i=1:length(numSpikes)
    for j=1:length(numSpikes{i})
        for k=1:length(numSpikes{i}{j})
            if numSpikes{i}{j}(k) >= 20
                seizure_leader{i}{j}{k} = [j,k,spikeStart{i}{j}{k}(1),evStart{i}{j}(k)];
            end
        end
    end
end