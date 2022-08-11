function [evStart, evStop, spikeStart, spikeStop, numSpikes] = eventDetector(detChanData, dt, featStart, featEnd, sdThresh)
%
% This function takes in LFP data from the detector channel and runs the automated
% epileptiform event and spike detectors
%
% Inputs:
% detChanData - [1 x n] vector of LFP data from the detector channel of the ECoG array for a given experiment
% dt - the sampling interval for data in detChanData
% featStart - the sample of detChanData on which the detector will start
% featEnd - the sample of detChanData on which the detector will end
% sdThresh - the standard deviation threshold used to deliniate epileptiform activity from baseline
%
% Outputs:
% evStart - an array of the start points of each detected epileptiform event
% evStop - an array of the end points of each detected epileptiform event
% spikeStart - a cell array the length of evStart, in which each cell contains the start point for every detected
%   spike within the corresponding event
% spikeStop - a cell array the length of evStart, in which each cell contains the end point for every detected
%   spike within the corresponding event
% numSpikes - a vector the length of evStart giving the number of detected spikes in each event
%


% Set window length and overlap (in milliseconds)
winLenMS = 400;
winLapMS = 300;
% Calculate window length and overlap points for each experiment
winLen = floor(winLenMS/dt); % winLen = samples, 400ms / millisecond/sample
winLap = floor(winLapMS/dt);
% Set the parameters for the correction factor for long events
memWinLen = 15; % Number of windows to look back
memFactor = 0.5; % Fraction of the avg sd in those windows to add to next
% Get number of windows
numWin = floor((length(featStart:featEnd)-winLap)/(winLen-winLap));
% Get the standard deviation feature of every window
sdFeatWins = zeros(1,numWin); sdFeatWinPts = zeros(1,numWin);
for k = 1:numWin
    sdFeatWinPts(k) = featStart+(k-1)*(winLen-winLap);
    % Check to see if the long event correction factor needs to be added
    if k>memWinLen && sum(sdFeatWins(k-memWinLen:k-1)>sdThresh) == memWinLen
        sdFeatWins(k) = std(detChanData(sdFeatWinPts(k):sdFeatWinPts(k)+winLen))+mean(sdFeatWins(k-memWinLen:k-1))*memFactor;
    else
        sdFeatWins(k) = std(detChanData(sdFeatWinPts(k):sdFeatWinPts(k)+winLen));
    end
end

% Find standard deviation threshold crossings
sdThreshBin = (sdFeatWins>sdThresh);
evStart1 = sdFeatWinPts(strfind(sdThreshBin,[0 1])+1); % Add 1 since strfind returns the beginning index
evStop1 = sdFeatWinPts(strfind(sdThreshBin,[1 0])+2); % Add 2 for strfind and give an extra 50ms after the thresh cross
% Make sure it does not end in the middle of an event
if length(evStart1) > length(evStop1)
    evStop1 = [evStop1 sdFeatWinPts(end)];
end

% Find subthreshold windows and get pooled standard deviation
subThreshWins = sdFeatWins(sdFeatWins<sdThresh);
avgSubThreshSTD = sqrt(sum(subThreshWins.^2)/length(subThreshWins));

% Set dummy event start/dtop variables
evStart2 = evStart1; evStop2 = evStop1; 

% Initialize spike start/stop cell arrays
spikeStart = cell(1,length(evStart1));
spikeStop = cell(1,length(evStart1));
% Initialize array for number of spikes in each event
numSpikes = zeros(1,length(evStart1));
% Create empy array to track events with no spikes in them
badEvents = [];
% Loop through detected events
for event = 1:length(evStart1)
    % Find where events cross the -6 times pooled standard deviation threshold
    spikeThreshBin = detChanData(evStart1(event):evStop1(event))<-6*avgSubThreshSTD;
    negCross = strfind(spikeThreshBin,[0 1])+1;
    posCross = strfind(spikeThreshBin,[1 0])+1;
    % Mark as bad event if there are no crossings, or just a single negative or positive
    if isempty(negCross) || isempty(posCross);
        badEvents = [badEvents event];
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
        ampChange = 1;
        while ampChange == 1; % Keep checking after removal
            if isempty(negCross) || isempty(posCross);
                ampChange = 0;
            % Remove first spike if it is not below -10STD (first spikes are typically much larger, so probably not the
            % beginning of a new event)
            elseif min(detChanData(evStart1(event)+negCross(1)-1:evStart1(event)+posCross(1)-1))< -10*avgSubThreshSTD
                ampChange = 0;
            else
                negCross(1) = []; posCross(1) = [];
            end
        end
        % Check again if there are no spikes after removing any
        if isempty(negCross) || isempty(posCross);
            badEvents = [badEvents event];
        else % Save spike start/stop times and # spikes
            spikeStart{event} = evStart1(event)+negCross-1;
            spikeStop{event} = evStart1(event)+posCross-1;
            numSpikes(event) = length(negCross);
        end
    end
end
% Remove any events without any spikes
spikeStart(badEvents) = []; spikeStop(badEvents) = [];
evStart2(badEvents) = []; evStop2(badEvents) = [];
numSpikes(badEvents) = [];

% Assign to outputs
evStart = evStart2; evStop = evStop2;

end