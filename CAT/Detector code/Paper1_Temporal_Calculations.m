%% Load Data
cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Paper 1 Seizure Model\Data')
load('AllExpTimingSec'); load('dtds.mat'); load('AllExpThElecData'); load('allExpStimPts')

%% Get stim points 
% stimThreshExp = [200 500 500 400 800]; stimElecExp = [3 3 3 1 6];
% allExpStimPts = cell(5,1);
% for exper = 1:5
%     cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Spike Analysis\All Seizure Data\Full Experiment Data Unfiltered')
%     fileName = ['Exp' num2str(exper) 'Unfilt'];
%     load(fileName)
%     disp(['Experiment ' num2str(exper) ' data has loaded'])
%     stimThresh = stimThreshExp(exper); stimElec = stimElecExp(exper);
%     stimElecDiff = diff(allEventData(stimElec,:));
%     % Get the points above threshold
%     [~, stimPts] = find(stimElecDiff > stimThresh);
%     allExpStimPts{exper} = stimPts;
% end

%% Get running STD feature
thElecExp = [4 3 3 3 6];
% Set threshold channel
featStartSecExp = [350 310 660 160 470];
featEndSecExp = [2550.85 7245 3227 7252 6929];
% Set window length and overlap
winLenMS = 400;
winLapMS = 300;
% Get standard deviation running feature for each experiment
sdFeatLong = cell(1,5); sdFeatWins = cell(1,5); sdFeatWinPts = cell(1,5); sdFeatLongTime = cell(1,5); sdFeatLongNoMem = cell(1,5);
memThreshExp = [130 275 300 285 200];
%memThreshExp = [130 275 175 285 200];
memWinLen = 15;
memFactor = 0.5;
%exper = 3;
for exper = 1:5
    % Get start and end points for given experiment
    featStart = find(allExpTimingSec{exper}>featStartSecExp(exper),1);
    featEnd = find(allExpTimingSec{exper}>featEndSecExp(exper),1);
    % Calculate window length and overlap points for each experiment
    winLen = floor(winLenMS/dtds(exper));
    winLap = floor(winLapMS/dtds(exper));
    % Get number of windows
    numWin = floor((length(featStart:featEnd)-winLap)/(winLen-winLap));    
    % Get the standard deviation feature of every window
    detFeat = zeros(1,numWin); detFeatWinPt = zeros(1,numWin);
    detFeatNoMem = zeros(1,numWin); 
    for k = 1:numWin
        detFeatWinPt(k) = featStart+(k-1)*(winLen-winLap);             
        if k>memWinLen && sum(detFeat(k-memWinLen:k-1)>memThreshExp(exper)) == memWinLen
            detFeat(k) = std(allExpThElecData{exper}(detFeatWinPt(k):detFeatWinPt(k)+winLen))+mean(detFeat(k-memWinLen:k-1))*memFactor;
        else
            detFeat(k) = std(allExpThElecData{exper}(detFeatWinPt(k):detFeatWinPt(k)+winLen));
        end
        detFeatNoMem(k) = std(allExpThElecData{exper}(detFeatWinPt(k):detFeatWinPt(k)+winLen));
    end
    featClipEnd = detFeatWinPt(k)+winLen-1;
    % Extend each sd value over the length of the window - overlap
    sdFeatWins{exper} = detFeat;
    sdFeatWinPts{exper} = detFeatWinPt;
    sdFeatLong{exper} = kron(detFeat,ones(1,winLen-winLap));
    sdFeatLongNoMem{exper} = kron(detFeatNoMem,ones(1,winLen-winLap));
    % Add the end of the last window since no overlap
    sdFeatLong{exper} = [sdFeatLong{exper} repmat(sdFeatLong{exper}(end),1,winLap)];
    sdFeatLongNoMem{exper} = [sdFeatLongNoMem{exper} repmat(sdFeatLongNoMem{exper}(end),1,winLap)];
    % Get the time for the sd feature
    sdFeatLongTime{exper} = allExpTimingSec{exper}(featStart:featClipEnd);
end

%% Plot the standard deviation feature
yMin = [-3000 -4100 -5100 -6000 -5000]; yMax = [1300 3500 3500 2700 3000];
exper = 3;
plot(allExpTimingSec{exper}/60,allExpThElecData{exper}/1000,'Color',[0.7 0.7 0.7])
hold on
plot(sdFeatLongTime{exper}/60,sdFeatLong{exper}/1000,'b')
line([0 allExpTimingSec{exper}(end)]/60,memThreshExp(exper)*[1 1]/1000,'Color','k')

% plot(sdFeatLongTime{3}/60,sdFeat3_300/1000,'b')
% %line([0 allExpTimingSec{exper}(end)]/60, [300 300]/1000,'Color','b')
% plot(sdFeatLongTime{3}/60,sdFeat3_250/1000,'r--')
% %line([0 allExpTimingSec{exper}(end)]/60, [250 250]/1000,'Color','r')
% plot(sdFeatLongTime{3}/60,sdFeat3_175/1000,'g-.')
% %line([0 allExpTimingSec{exper}(end)]/60, [175 175]/1000,'Color','g')
%% Take out windows during bad sections
% Sections to ignore due to large stim trains or noise/artifact
badSectionsSecExp{1} = [];
badSectionsSecExp{2} = [1750 1793; 2746 3110; 3825 4515; 5360 5480; 6658 6730];
badSectionsSecExp{3} = [];
badSectionsSecExp{4} = [2002 2014; 4922 4938; 5890 5950; 6921 6971];
badSectionsSecExp{5} = [1932 1944; 5118 5430; 6145 6274];

% Loop through to take out any windows that are in the bad sections
sdFeatWinPtsGood = cell(1,5); sdFeatWinsGood = cell(1,5);
for exper = 1:5
    badSections = badSectionsSecExp{exper};
    if isempty(badSectionsSecExp{exper})
        sdFeatWinPtsGood{exper} = sdFeatWinPts{exper};
        sdFeatWinsGood{exper} = sdFeatWins{exper};
    else
        % Set up binary variables to get points in the bad sections
        sdPtsBadBin = zeros(size(badSectionsSecExp{exper},1),length(sdFeatWinPts{exper}));
        for i = 1:size(badSectionsSecExp{exper},1)
            badSections(i,1) = floor(find(allExpTimingSec{exper}>badSectionsSecExp{exper}(i,1),1));
            badSections(i,2) = ceil(find(allExpTimingSec{exper}>badSectionsSecExp{exper}(i,2),1));
            % Find starts/stops in each group of bad points
            sdPtsBadBin(i,:) = (sdFeatWinPts{exper} > badSections(i,1)) & (sdFeatWinPts{exper} < badSections(i,2));            
        end
        % Sum the binary vectors to get start/stop points in any bad section
        allsdPtsBadBin = sum(sdPtsBadBin)>0;
        sdFeatWinPtsGood{exper} = sdFeatWinPts{exper}(~allsdPtsBadBin);
        sdFeatWinsGood{exper} = sdFeatWins{exper}(~allsdPtsBadBin);
    end
end

%% Get Event start and stop times from STD Feature threshold crossings
evStart1 = cell(1,5); evStop1 = cell(1,5);
% Find threshold crossings
for exper = 1:5
    sdThreshBin = (sdFeatWinsGood{exper}>memThreshExp(exper));
    evStart1{exper} = sdFeatWinPtsGood{exper}(strfind(sdThreshBin,[0 1])+1); % Add 1 since strfind returns the beginning index
    evStop1{exper} = sdFeatWinPtsGood{exper}(strfind(sdThreshBin,[1 0])+2); % Add 2 for strfind and give an extra 50ms after the thresh cross
    if length(evStart1{exper}) > length(evStop1{exper})
        evStop1{exper} = [evStop1{exper} sdFeatWinPtsGood{exper}(end)];
    end
end

%% Find subthreshold windows and get average std
subThreshWins = cell(5,1); subThreshWinPts = cell(5,1);
avgSubThreshSTD = zeros(1,5); %meanSubThreshSTD = zeros(1,5);
for exper = 1:5
    subThreshWins{exper} = sdFeatWinsGood{exper}(sdFeatWinsGood{exper}<memThreshExp(exper));
    subThreshWinPts{exper} = sdFeatWinPtsGood{exper}(sdFeatWinsGood{exper}<memThreshExp(exper));
    avgSubThreshSTD(exper) = sqrt(sum(subThreshWins{exper}.^2)/length(subThreshWins{exper}));
    %meanSubThreshSTD(exper) = mean(subThreshWins{exper});
end

%% Get each spike beginning and end in every event - Remove events with no spikes
numSpikes = cell(1,5); spikeStart = cell(1,5); spikeStop = cell(1,5);
badEvents = cell(1,5);
evStart2 = evStart1; evStop2 = evStop1; %evHasStim2 = evHasStim1;
for exper = 1:5
    spikeStart{exper} = cell(1,length(evStart1{exper}));
    spikeStop{exper} = cell(1,length(evStart1{exper}));
    numSpikes{exper} = zeros(1,length(evStart1{exper}));
    for event = 1:length(evStart1{exper})
        % Find where events cross the -6STD threshold
        spikeThreshBin = allExpThElecData{exper}(evStart1{exper}(event):evStop1{exper}(event))<-6*avgSubThreshSTD(exper);
        negCross = strfind(spikeThreshBin,[0 1])+1;
        posCross = strfind(spikeThreshBin,[1 0])+1;
        % Remove if there are no crossings, or just a negative or positive
        if isempty(negCross) || isempty(posCross);
            badEvents{exper} = [badEvents{exper} event];
        else
            % If negCross ~= posCross remove the appropriate one
            if length(negCross) > length(posCross); negCross(end) = []; end
            if length(posCross) > length(negCross); posCross(1) = []; end
            % Throw out any detections 10 samples long or shorter
            longSpikeBin = posCross-negCross > 10;
            negCross = negCross(longSpikeBin);
            posCross = posCross(longSpikeBin);
            % Throw out any returns to baseline 10 samples long or shorter
            if negCross > 1
                longISIBin = negCross(2:end)-posCross(1:end-1) > 10;
                negCross = negCross([true longISIBin]);
                posCross = posCross([longISIBin true]);
            end
            % Check again if there are no spikes after removing any
            % Remove first spike if it is not below -10STD
            ampChange = 1;
            while ampChange == 1; % Keep checking after removal
                if isempty(negCross) || isempty(posCross);
                    ampChange = 0;
                elseif min(allExpThElecData{exper}(evStart1{exper}(event)+negCross(1)-1:evStart1{exper}(event)+posCross(1)-1))< -10*avgSubThreshSTD(exper)
                    ampChange = 0;
                else
                    negCross(1) = []; posCross(1) = [];
                end
            end
            % Check again if there are no spikes after removing any
            if isempty(negCross) || isempty(posCross);
                badEvents{exper} = [badEvents{exper} event];
            else % Save spike start/stop times and # spikes
                spikeStart{exper}{event} = evStart1{exper}(event)+negCross-1;
                spikeStop{exper}{event} = evStart1{exper}(event)+posCross-1;
                numSpikes{exper}(event) = length(negCross);
            end
        end
    end
    % Remove any events without any spikes
    spikeStart{exper}(badEvents{exper}) = []; spikeStop{exper}(badEvents{exper}) = [];
    evStart2{exper}(badEvents{exper}) = []; evStop2{exper}(badEvents{exper}) = [];
    numSpikes{exper}(badEvents{exper}) = []; %evHasStim2{exper}(badEvents{exper}) = [];
end

%% Take out events triggered by stim
evStimTrigBin = cell(5,1);
for exper = 1:5
    evStimTrigBin{exper} = ones(1,length(evStart2{exper}));
    for event = 1:length(evStart2{exper})
        allClipPts = spikeStart{exper}{event}(1)-400:spikeStart{exper}{event}(1)+160;
        if isempty(intersect(allClipPts,allExpStimPts{exper}))
            evStimTrigBin{exper}(event) = 0;
        end
    end
    evStimTrigBin{exper} = logical(evStimTrigBin{exper});
    evStart2{exper}(evStimTrigBin{exper}) = []; evStop2{exper}(evStimTrigBin{exper}) = [];
    spikeStart{exper}(evStimTrigBin{exper}) = []; spikeStop{exper}(evStimTrigBin{exper}) = [];
    numSpikes{exper}(evStimTrigBin{exper}) = [];
% subplot(5,1,exper)
% plot(evStart2{exper}(~evStimTrigBin{exper}),numSpikes{exper}(~evStimTrigBin{exper}),'bo'); hold on
% plot(evStart2{exper}(evStimTrigBin{exper}),numSpikes{exper}(evStimTrigBin{exper}),'ro');
end
    
evStart = evStart2; evStop = evStop2;

%% Get event duration and interval and spike duration and interval
% Initialize variables
evDur = cell(5,1); evDurSec = cell(5,1);
iei = cell(5,1); ieiSec = cell(5,1);
spikeDur = cell(5,1); spikeDurSec = cell(5,1);
firstSpikeDur = cell(5,1); firstSpikeDurSec = cell(5,1);
isi = cell(5,1); isiSec = cell(5,1);

for exper = 1:5
    evDur{exper} = zeros(length(evStart2{exper}),1); evDurSec{exper} = zeros(length(evStart2{exper}),1);    
    iei{exper} = zeros(length(evStart2{exper})-1,1); ieiSec{exper} = zeros(length(evStart2{exper})-1,1);   
    spikeDur{exper} = cell(length(evStart2{exper}),1); spikeDurSec{exper} = cell(length(evStart2{exper}),1);
    firstSpikeDur{exper} = zeros(length(evStart2{exper}),1); firstSpikeDurSec{exper} = zeros(length(evStart2{exper}),1);
    isi{exper} = cell(length(evStart2{exper}),1);isiSec{exper} = cell(length(evStart2{exper}),1);    
    % Loop through each event to get event durations and intervals
    for event = 1:length(evStart2{exper})
        % Event duration from the first spike start to the last spike stop
        evDur{exper}(event) = spikeStop{exper}{event}(end)-spikeStart{exper}{event}(1);
        evDurSec{exper}(event) = allExpTimingSec{exper}(spikeStop{exper}{event}(end))-allExpTimingSec{exper}(spikeStart{exper}{event}(1));
        if event < length(evStart2{exper})
            % inter event interval from end of event last spike to start of
            % first spike of next event
            iei{exper}(event) = spikeStart{exper}{event+1}(1)-spikeStop{exper}{event}(end);
            ieiSec{exper}(event) = allExpTimingSec{exper}(spikeStart{exper}{event+1}(1))-allExpTimingSec{exper}(spikeStop{exper}{event}(end));
        end
                
        spikeDur{exper}{event} = zeros(numSpikes{exper}(event),1);
        spikeDurSec{exper}{event} = zeros(numSpikes{exper}(event),1);
        isi{exper}{event} = zeros(numSpikes{exper}(event)-1,1);
        isiSec{exper}{event} = zeros(numSpikes{exper}(event)-1,1);
        % Loop through each spike to get spike durations and intervals
        for spk = 1:numSpikes{exper}(event) 
            % Spike duration from spike start to spike end
            spikeDur{exper}{event}(spk) = spikeStop{exper}{event}(spk)-spikeStart{exper}{event}(spk);
            spikeDurSec{exper}{event}(spk) = allExpTimingSec{exper}(spikeStop{exper}{event}(spk))-allExpTimingSec{exper}(spikeStart{exper}{event}(spk));
            if numSpikes{exper}(event)>1 && spk<numSpikes{exper}(event)
                % inter spike interval from end of spike to start of next
                isi{exper}{event}(spk) = spikeStart{exper}{event}(spk+1)-spikeStop{exper}{event}(spk);
                isiSec{exper}{event}(spk) = allExpTimingSec{exper}(spikeStart{exper}{event}(spk+1))-allExpTimingSec{exper}(spikeStop{exper}{event}(spk));
            end            
        end 
        firstSpikeDur{exper}(event) = spikeDur{exper}{event}(1);
        firstSpikeDurSec{exper}(event) = spikeDurSec{exper}{event}(1);
    end
end

%% Save variables
cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Paper 1 Seizure Model\Data')
% save('temporalGeneral.mat','evStart','evStop','spikeStart','spikeStop','numSpikes','-v7.3');
% save('temporalAdvanced.mat','iei','ieiSec','isi','isiSec','evDur','evDurSec','spikeDur','spikeDurSec','-v7.3');
save('detectorVars.mat','sdFeatLong','sdFeatLongNoMem','sdFeatLongTime','sdFeatWinsGood','sdFeatWinPtsGood','memThreshExp','avgSubThreshSTD','-v7.3');


%% Plot the histogram of each detector
for exper = 1:5
    %subplot(1,5,exper)
    %histogram(sdFeatWinsGood{exper}); hold on
    %line(memThreshExp(exper)*[1 1], [0 10000])
    subTWinds
end
%%
subThreshWins = cell(5,1); supraThreshWins = cell(5,1);
avgSubThreshSTD = zeros(1,5); avgSupraThreshSTD = zeros(1,5); 
for exper = 1:5
    subThreshWins{exper} = sdFeatWinsGood{exper}(sdFeatWinsGood{exper}<memThreshExp(exper));
    supraThreshWins{exper} = sdFeatWinsGood{exper}(sdFeatWinsGood{exper}>=memThreshExp(exper));
    avgSubThreshSTD(exper) = sqrt(sum(subThreshWins{exper}.^2)/length(subThreshWins{exper}));
    avgSupraThreshSTD(exper) = sqrt(sum(supraThreshWins{exper}.^2)/length(supraThreshWins{exper}));
end
avgSubThreshSTDAll = sqrt(sum(cell2mat(subThreshWins').^2)/length(cell2mat(subThreshWins')));
avgSupraThreshSTDAll = sqrt(sum(cell2mat(supraThreshWins').^2)/length(cell2mat(supraThreshWins')));
