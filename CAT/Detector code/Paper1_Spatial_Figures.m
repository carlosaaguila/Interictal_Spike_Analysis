%% Load data
cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Paper 1 Seizure Model\Data')
%cd('/mnt/local/gdrive/public/USERS/binkh/Depth and Surface Testing/Paper 1 Seizure Model/Data')
load('AllExpTimingSec'); load('dtds.mat'); load('allExpThElecData'); %load('allExpStimPts'); load('allExpAnnotations')
load('temporalGeneral.mat'); load('GreenPurpleCMap129'); load('baseFSClips');

thElecExp = [4 3 3 3 6];

%% Get baseline and fs clips for each event on every electrode
% baseClips = cell(5,1); fsClips = cell(5,1);
% for exper = 1:5
%     cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Spike Analysis\All Seizure Data\Full Experiment Data Unfiltered')
%     fileName = ['Exp' num2str(exper) 'Unfilt'];
%     load(fileName)
%     disp(['Experiment ' num2str(exper) ' data has loaded'])
%     baseClips{exper} = zeros(32,length(evStart{exper}),201);
%     fsClips{exper} = zeros(32,length(evStart{exper}),201);    
%     for event = 1:length(evStart{exper})
%         fsClipPts = spikeStart{exper}{event}(1)-40:spikeStart{exper}{event}(1)+160;
%         baseClipPts = spikeStart{exper}{event}(1)-400:spikeStart{exper}{event}(1)-200;
%         if (event > 1) && (spikeStart{exper}{event}(1)-400 < evStop{exper}(event-1))
%             disp(['Base clip starts before previous event # ' num2str(event-1) ' ends'])
%         end
%         for elec = 1:32
%             baseClips{exper}(elec,event,:) = allEventData(elec,baseClipPts); 
%             fsClips{exper}(elec,event,:) = allEventData(elec,fsClipPts); 
%         end        
%     end
% end
% % for -400->-200, overlapping events are: Exp3 - Ev 719, 1147, 1149; Exp4 - Ev 1233, 1610
% % (these start appx where or right after previous one stops - same event really)
% cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Paper 1 Seizure Model\Data')

%% Normalized fsClips be removing baseline mean
fsClipsNorm = cell(5,1);
baseClipsNoMean = cell(5,1); fsClipsNoMean = cell(5,1);
for exper = 1:5                    
    fsClips{exper} = fsClips{exper}/1000; baseClips{exper} = baseClips{exper}/1000;
    fsClipsNorm{exper} = (fsClips{exper}-repmat(mean(baseClips{exper},3),1,1,201));    
    fsClipsNoMean{exper} = (fsClips{exper}-repmat(mean(fsClips{exper},3),1,1,201));
    baseClipsNoMean{exper} = (baseClips{exper}-repmat(mean(baseClips{exper},3),1,1,201));
end


%% Extrapolate minimum and saturation under threshold of saturated events in experiment 2

% For events with min < -2, see where two fitted lines cross to estimate min and negSum
minTrueTrain = cell(5,1); minEstTrain = cell(5,1); minEstTest = cell(5,1);
minTrainError = cell(5,1); 
trapEstTrain = cell(5,1); trapTrueTrain = cell(5,1); trapEstTest = cell(5,1);
trapTrainError = cell(5,1);
for exper = 1:5
    nanMat = nan(32,size(fsClips{exper},2));
    trapEstTrain{exper} = nanMat; trapTrueTrain{exper} = nanMat; trapEstTest{exper} = nanMat;
    minTrueTrain{exper} = nanMat; minEstTrain{exper} = nanMat; minEstTest{exper} = nanMat;
    for elec = 1:32
        for event = 1:size(fsClips{exper},2)
            tmpClip = squeeze(fsClipsNorm{exper}(elec,event,:));
            tmpMinIdx = find(tmpClip == min(tmpClip));
            % Run min/negSum estimate on unsaturated clips that have similar shape to the saturated ones in exp 2
            if (min(tmpClip)<-2) && (length(find(tmpClip == min(tmpClip))) < 2) && (tmpMinIdx < 140) && (tmpMinIdx > 25)
                % Take 6 points above -2 ascending and descending to fit lines and estimate spike min/negSum
                if (find(tmpClip < -2,1,'first')-1) >5
                    x1(2) = find(tmpClip < -2,1,'first')-1;
                else
                    x1(2) = tmpMinIdx-5;
                end
                if isempty(find(tmpClip(tmpMinIdx:end) > -2,1,'first')) || (find(tmpClip(tmpMinIdx:end) > -2,1,'first')+tmpMinIdx-1+1 > 196)
                    x2(1) = tmpMinIdx+5;
                else
                    x2(1) = find(tmpClip(tmpMinIdx:end) > -2,1,'first')+tmpMinIdx-1+1;
                end
                x1(1) = x1(2)-5; if x1(1) < 1; x1(1) = 1; end
                x2(2) = x2(1)+5; if x2(2) > 201; x2(2) = 201; end
                p1 = polyfit(x1,tmpClip(x1)',1);
                p2 = polyfit(x2,tmpClip(x2)',1);
                xint = (p2(2)-p1(2))/(p1(1)-p2(1));
                
                minEstTrain{exper}(elec,event) = polyval(p1,xint);
                minTrueTrain{exper}(elec,event) = min(tmpClip);
                % Get trap estimate and truth for events with a min estimated between the -2 thresh crossings
                if xint > x1(2) && xint < x2(1)
                    tmpClip2 = tmpClip;
                    tmpClip2(x1(2):round(xint)) = polyval(p1,x1(2):round(xint));
                    tmpClip2(round(xint)+1:x2(1)) = polyval(p2,round(xint)+1:x2(1));
                    trapEstTrain{exper}(elec,event) = trapz((1:201)*dtds(exper),tmpClip2);
                    trapTrueTrain{exper}(elec,event) = trapz((1:201)*dtds(exper),tmpClip);
                end                                                                       
            end
            % Estimate mins for saturated clips
            if exper == 2 && any(elec == [3 4 7 8]) && (length(find(tmpClip == min(tmpClip))) > 9)
                x1(2) = find(tmpClip == min(tmpClip),1,'first')-1;
                x2(1) = find(tmpClip == min(tmpClip),1,'last')+1;
                x1(1) = x1(2)-5; if x1(1) < 1; x1(1) = 1; end
                x2(2) = x2(1)+5; if x2(2) > 201; x2(2) = 201; end
                p1 = polyfit(x1,tmpClip(x1)',1);
                p2 = polyfit(x2,tmpClip(x2)',1);
                xint = (p2(2)-p1(2))/(p1(1)-p2(1));
                minEstTest{exper}(elec,event) = polyval(p1,xint);
                
                if xint > x1(2) && xint < x2(1)
                    tmpClip2 = tmpClip;
                    tmpClip2(x1(2):round(xint)) = polyval(p1,x1(2):round(xint));
                    tmpClip2(round(xint)+1:x2(1)) = polyval(p2,round(xint)+1:x2(1));
                    trapEstTest{exper}(elec,event) = trapz((1:201)*dtds(exper),tmpClip2);
                end
%                     xplot1 = x1(1):ceil(xint); xplot2 = floor(xint):x2(2);
%                     plot(tmpClip,'b'); hold on
%                     plot(xplot1,polyval(p1,xplot1),'k'); plot(xplot2,polyval(p2,xplot2),'k');
%                     line(x1, tmpClip(x1),'Color','r'); line(x2, tmpClip(x2),'Color','r')
%                     line([1 201], (minEstTest{exper}(elec,event)+0.3236)*[1 1],'Color','g')
%                     title(['Exp' num2str(exper) ' - Elec' num2str(elec) ' - Ev' num2str(event)])
%                     ylim([-6 2]); xlim([1 201])
%                     pause; clf;              
            end            
        end
    end
    % Get error of estimates on unsaturated clips
    minTrainError{exper} = minEstTrain{exper}-minTrueTrain{exper};
    trapTrainError{exper} = trapEstTrain{exper}-trapTrueTrain{exper};
end
% Get min/negSum error offset and remove to get median error of 0
minEstOffset = nanmedian(reshape(cell2mat(minTrainError'),1,[]));
minEstTest{2} = minEstTest{2}-minEstOffset;
trapEstOffset = nanmedian(reshape(cell2mat(trapTrainError'),1,[]));
trapEstTest{2} = trapEstTest{2} - trapEstOffset;
%% Look at training error of extrapolation and regression
for exper = 1:5
subplot(5,1,exper)
%scatter(minTrueTrain{exper}(:),negSumTrueTrain{exper}(:),'bo')
%scatter(thWidthTrain{exper}(:),negSumTrueTrain{exper}(:),'bo')
%scatter3(minTrueTrain{exper}(:),thWidthTrain{exper}(:),negSumTrueTrain{exper}(:))
histogram(minTrainError{exper}-minEstOffset,[-5:0.1:5])
% plot(reshape(negSumTrueTrain{exper},1,[]),'bo'); hold on
% plot(reshape(negSumEstTrain{exper},1,[]),'ro');
 title(num2str(exper))
%pause; clf
end
 histogram(reshape(cell2mat(minTrainError'),1,[]),[-5:0.1:5]); hold on
 histogram(reshape(cell2mat(minTrainError'),1,[])-minEstOffset,[-5:0.1:5])

  %histogram(reshape(cell2mat(negSumTrainError'),1,[]))
  %histogram(reshape(cell2mat(trapTrainError'),1,[]))

  plot(reshape(negSumEstTest{2},1,[]),'bo')


%% Get mean squares of baseline and fs clips and get mean fsClips voltage z-scored to its baseline
baseVar = cell(5,1); fsVar = cell(5,1); 
basePower = cell(5,1); fsPower = cell(5,1);
baseMMS = cell(5,1); fsMMS = cell(5,1); 
baseInt = cell(5,1); fsInt = cell(5,1); 
for exper = 1:5    
    zMat = zeros(32,size(fsClips{exper},2));
    baseVar{exper} = zMat; fsVar{exper} = zMat;   
    basePower{exper} = zMat; fsPower{exper} = zMat;
    baseMMS{exper} = zMat; fsMMS{exper} = zMat;
    baseInt{exper} = zMat; fsInt{exper} = zMat;
    for elec = 1:32
        for event = 1:size(fsClips{exper},2)       
            baseVar{exper}(elec,event) = var(squeeze(baseClips{exper}(elec,event,:)),1);
            fsVar{exper}(elec,event) = var(squeeze(fsClips{exper}(elec,event,:)),1);
            
            tmpBaseClip = squeeze(baseClipsNoMean{exper}(elec,event,:));      
            basePower{exper}(elec,event) = mean(tmpBaseClip.^2);
            baseMMS{exper}(elec,event) = min(tmpBaseClip) + max(tmpBaseClip);                
            baseInt{exper}(elec,event) = trapz((1:201)*dtds(exper),tmpBaseClip);
            
            tmpfsClip = squeeze(fsClipsNorm{exper}(elec,event,:));
            fsPower{exper}(elec,event) = mean(tmpfsClip.^2);
            if isnan(minEstTest{exper}(elec,event))
                fsMMS{exper}(elec,event) = min(tmpfsClip) + max(tmpfsClip);                
            else
                 fsMMS{exper}(elec,event) = minEstTest{exper}(elec,event) + max(tmpfsClip);                 
            end            
            if isnan(trapEstTest{exper}(elec,event))
                fsInt{exper}(elec,event) = trapz((1:201)*dtds(exper),tmpfsClip);
            else
                fsInt{exper}(elec,event) = trapEstTest{exper}(elec,event);
            end            
        end
    end
end

%% Get bin points for each experiment
binLen = [22 96 90 108 110];
binLap = [11 48 45 54 55];
binStarts = {1,[1 785],1,1,[1 607 1374]};
binEnds = {283,[784 1296],1159,1380,[606 1373 1571]};
binPts = cell(5,1); binSplits = cell(5,1);
for exper = 1:5
    binPts{exper} = [];     
    for j = 1:length(binStarts{exper})
        tmpNumBin = floor((length(binStarts{exper}(j):binEnds{exper}(j))-binLap(exper))/(binLen(exper)-binLap(exper)));
        tmpBinPts = cell(1,tmpNumBin);
        for i = 1:tmpNumBin
            if i < tmpNumBin
                tmpBinPts{i} = (i-1)*binLap(exper)+binStarts{exper}(j):(i-1)*binLap(exper)+binStarts{exper}(j)+binLen(exper)-1;
            else
                tmpBinPts{i} = (i-1)*binLap(exper)+binStarts{exper}(j):binEnds{exper}(j);
            end           
        end
        binPts{exper} = [binPts{exper} tmpBinPts];
        if (length(binStarts{exper})>1) && (j<length(binStarts{exper}))
            binSplits{exper} = [binSplits{exper} length(binPts{exper})];
        end
    end
end
binMids = zeros(5,24);
for exper = 1:5; for i = 1:24; binMids(exper,i) = median(binPts{exper}{i}); end; end

%% Test for significance in overlapping bins and get average mean V if significant
expAlpha = zeros(5,1); numBin = zeros(5,1);
binMWU = cell(5,1); 
binPower = cell(5,1);
binMMS = cell(5,1); binInt = cell(5,1);
binCorrMat = cell(5,1); binCorr = cell(5,1); 
for exper = 1:5
    numBin(exper) = length(binPts{exper});
    expAlpha(exper) = 0.05/length(binPts{exper});
    zMat = zeros(32,numBin(exper)); nMat = nan(32,numBin(exper));
    binMWU{exper} = zMat;
    binPower{exper} = nMat; binMMS{exper} = nMat; binInt{exper} = nMat;
    binCorrMat{exper} = cell(32,numBin(exper)); binCorr{exper} = cell(32,numBin(exper));
    for i = 1:numBin(exper)
        tmpMask = logical(triu(ones(length(binPts{exper}{i})),1));
        for elec = 1:32
            binMWU{exper}(elec,i) = ranksum(baseVar{exper}(elec,binPts{exper}{i}),fsVar{exper}(elec,binPts{exper}{i}));
            if binMWU{exper}(elec,i) < expAlpha(exper)
                binPower{exper}(elec,i) = nanmedian(fsPower{exper}(elec,binPts{exper}{i}));                 
                binMMS{exper}(elec,i) = nanmedian(fsMMS{exper}(elec,binPts{exper}{i}));                 
                binInt{exper}(elec,i) = nanmedian(fsInt{exper}(elec,binPts{exper}{i}));                
            end
            binCorrMat{exper}{elec,i} = corr(squeeze(fsClipsNorm{exper}(elec,binPts{exper}{i},:))');            
            binCorr{exper}{elec,i} = binCorrMat{exper}{elec,i}(tmpMask);
        end
    end
end

% Get focus and surround MMS thresholds
allbmmm = reshape(cell2mat(binMMS),1,[]);
focThresh = -1*prctile(abs(allbmmm(allbmmm<0)),33);
surrThresh = prctile(allbmmm(allbmmm>0),33);

%% Plot pairwise correlation statistics for each bin over all experiments (Figure 6)
allThreshCorr = []; allThreshCorrGroup = [];
allThreshCorrMedian = zeros(1,24);
allThreshCorr25 = zeros(1,24);
allThreshCorr75 = zeros(1,24);
for i = 1:24
    tmpThreshCorr = [];
    for exper = 1:5
        tmpThreshCorr = [tmpThreshCorr binCorr{exper}{thElecExp(exper),i}'];
    end
    %subplot(4,6,i); histogram(tmpThreshCorr)
    allThreshCorr = [allThreshCorr tmpThreshCorr];
    allThreshCorrGroup = [allThreshCorrGroup i*ones(1,length(tmpThreshCorr))];
    allThreshCorrMedian(i) = nanmedian(tmpThreshCorr);
    allThreshCorr25(i) = prctile(tmpThreshCorr,25);
    allThreshCorr75(i) = prctile(tmpThreshCorr,75); 
end

errorbar(1:24,allThreshCorrMedian,allThreshCorr25-allThreshCorrMedian,allThreshCorr75-allThreshCorrMedian,'o','color','k','MarkerFaceColor','k','LineWidth',1.5,'MarkerSize',8)
ylim([0.85 1]); set(gca,'box','off')
set(gca,'Linewidth',2,'xtick',[10 20],'fontsize',12)


%% Plot grids with norm MMS color and mean traces for example bins (Figure 8)
cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Paper 1 Seizure Model\Figures\Spatial Bin Plots\')
colormap([0.5 0.5 0.5; gpMap129; 0.5 0.5 0.5]); %colormap(gpMap129);
[X,Y] = meshgrid(1:8,1:4);
%L = 10.^(linspace(-2,0,63));
L = 10.^(-1.1:0.03:0.73);
logmap = [-fliplr(L) 0 L];
ctr = 1;
for exper = 1:5%[1 4]
    for i = 1:24%[1 12 24]
        %subplot(5,3,ctr)
        %ctr = ctr+1;
        
        [~,~,cLogIdx] = histcounts(binMMS{exper}(:,i),logmap);
        tmpC = reshape(cLogIdx,4,8);
        imagesc(tmpC,[1 length(logmap)])%imagesc(tmpC,[0 128])
        hold on
        
%         for elec = 1:32
%             if isnan(binMMS{exper}(elec,i))
%             else
%                 tmpx = linspace(X(elec)-0.4,X(elec)+0.4,201);
%                 tmpMean = squeeze(mean(fsClipsNorm{exper}(elec,binPts{exper}{i},:)/10,2));
%                 tmpStd = squeeze(std(fsClipsNorm{exper}(elec,binPts{exper}{i},:)/10,[],2));
%                 fill([tmpx fliplr(tmpx)], -1*[tmpMean'+tmpStd' fliplr(tmpMean'-tmpStd')]+Y(elec),[0.7 0.7 0.7],'FaceAlpha',0.5); hold on
%                 plot(tmpx,-1*tmpMean+Y(elec),'k','LineWidth',1.5);
%             end
%         end
        
        title(['MMS | Experiment ' num2str(exper) ' | Bin ' num2str(i)])
        set(gca,'Ytick',[],'xtick',[],'box','off')
        cb = colorbar; set(cb,'YTick',[2 25 35 59 67 78 101 124]);        
        
        set(gcf,'color','white','visible','off','Position',[50 50 1600 900],'PaperPosition', [.25 .25 16 10])
        frameName = ['Exp' num2str(exper) 'MMSnormBin#' num2str(i)];
        print(gcf,'-dpng',frameName)
        
    end
end


%% Plot focus and surround bin area and intensity sum for each experiment (Figure 8)

binNumFocElecs = zeros(5,24); binNumSurrElecs = zeros(5,24);
binTotalFocInt = zeros(5,24); binTotalSurrInt = zeros(5,24);
for exper = 1:5
    for i = 1:24
        tmpFocElecs = binMMS{exper}(:,i)<=focThresh;
        binNumFocElecs(exper,i) = sum(tmpFocElecs);
        binTotalFocInt(exper,i) = sum(binInt{exper}(tmpFocElecs,i));
        tmpSurrElecs = binMMS{exper}(:,i)>=surrThresh;
        binNumSurrElecs(exper,i) = sum(tmpSurrElecs);
        binTotalSurrInt(exper,i) = sum(binInt{exper}(tmpSurrElecs,i));
    end
    
    subplot(1,5,exper)
    yyaxis right
    bar(-1*binNumFocElecs(exper,:),1,'FaceColor',[0 0.7 0],'linestyle','none','FaceAlpha',0.25)
    hold on
    bar(binNumSurrElecs(exper,:),1,'FaceColor',[0.7 0 0.7],'linestyle','none','FaceAlpha',0.25)
    xlim([1 24]); ylim([-30 30]);
    yyaxis left
    plot(1:24,binTotalFocInt(exper,:),'-','Color',[0 0.7 0]','LineWidth',2); hold on
    plot(1:24,binTotalSurrInt(exper,:),'-','Color',[0.7 0 0.7],'LineWidth',2);
    xlim([1 24]); ylim([-3500 3500]);
    for n = 1:length(binSplits{exper}); line(binSplits{exper}(n)*[1 1], [-7000 7000],'Color',[0.7 0.7 0.7]); hold on; end    
    set(gca, 'box', 'off','FontSize',10)
end

%% Bar plot of surround and focus area and intensity at beginning and end of each exp (Figure 8)

% Area
focStart = zeros(5,1); focEnd = zeros(5,1); focGrowth = zeros(5,1);
surrStart = zeros(5,1); surrEnd = zeros(5,1); surrGrowth = zeros(5,1);
for exper = 1:5       
    focStart(exper) = median(binNumFocElecs(exper,1:3));
    focEnd(exper) = median(binNumFocElecs(exper,end-2:end));
    focGrowth(exper) = 100*(focEnd(exper)-focStart(exper))/focStart(exper);

    surrStart(exper) = median(binNumSurrElecs(exper,1:3));
    surrEnd(exper) = median(binNumSurrElecs(exper,end-2:end));
    surrGrowth(exper) = 100*(surrEnd(exper)-surrStart(exper))/surrStart(exper);
end

subplot(121)
bar([-1*focStart -1*focEnd],'FaceColor',[0 0.7 0]); hold on
bar([surrStart surrEnd],'FaceColor',[0.7 0 0.7])
ylim([-17 26]);
set(gca, 'box', 'off','LineWidth',1,'FontSize',10)%,'ytick',[0 5 10 15])
title('Foc/Surr Area Growth')


% Intensity
focIntStart = zeros(5,1); focIntEnd = zeros(5,1); focIntGrowth = zeros(5,1);
surrIntStart = zeros(5,1); surrIntEnd = zeros(5,1); surrIntGrowth = zeros(5,1);
for exper = 1:5
    focIntStart(exper) = median(binTotalFocInt(exper,1:3));
    focIntEnd(exper) = median(binTotalFocInt(exper,end-2:end));
    focIntGrowth(exper) = 100*(focIntEnd(exper)-focIntStart(exper))/focIntStart(exper);
    
    surrIntStart(exper) = median(binTotalSurrInt(exper,1:3));
    surrIntEnd(exper) = median(binTotalSurrInt(exper,end-2:end));
    surrIntGrowth(exper) = 100*(surrIntEnd(exper)-surrIntStart(exper))/surrIntStart(exper);
end
subplot(122)
bar([focIntStart focIntEnd],'FaceColor',[0 0.7 0]); hold on
bar([surrIntStart surrIntEnd],'FaceColor',[0.7 0 0.7]); ylim([-2500 1200])
set(gca, 'box', 'off','LineWidth',1,'FontSize',10)%,'ytick',[0 5 10 15])
title('Foc/Surr Intensity Growth')


%% Look at total focus and surround intensity and area for every event and correlation
% Get area and intensity of focus and surround for every event
numFocElecEv = cell(5,1); numSurrElecEv = cell(5,1);
focIntEv = cell(5,1); surrIntEv = cell(5,1);
focIntEvZ = cell(5,1); surrIntEvZ = cell(5,1);
binIntFocZ = zeros(5,24); binIntSurrZ = zeros(5,24);
for exper = 1:5
    numFocElecEv{exper} = sum(fsMMS{exper}<=focThresh);
    numSurrElecEv{exper} = sum(fsMMS{exper}>=surrThresh);
    focIntEv{exper} = sum(fsInt{exper}.*(fsMMS{exper}<=focThresh));
    focIntEvZ{exper} = (focIntEv{exper}-nanmean(focIntEv{exper}))/nanstd(focIntEv{exper});
    surrIntEv{exper} = sum(fsInt{exper}.*(fsMMS{exper}>=surrThresh));
    surrIntEvZ{exper} = (surrIntEv{exper}-nanmean(surrIntEv{exper}))/nanstd(surrIntEv{exper});
end
    
% Plot total focus and surround intensity for each event and bin medians
for exper = 1:5
    subplot(2,5,exper)
    scatter(1:length(focIntEv{exper}),focIntEv{exper},'.','MarkerEdgeColor',[0 0.7 0],'MarkerEdgeAlpha',0.5); hold on
    scatter(1:length(surrIntEv{exper}),surrIntEv{exper},'.','MarkerEdgeColor',[0.7 0 0.7],'MarkerEdgeAlpha',0.5); hold on
    plot(binMids(exper,:),binTotalFocInt(exper,:),'Color',[0 0 0],'LineWidth',2);
    plot(binMids(exper,:),binTotalSurrInt(exper,:),'Color',[0 0 0],'LineWidth',2);
    xlim([0 length(focIntEv{exper})])
    ylim([-4000 2000])
end

% Look at correlation between focus and surround intensity using every event
for exper = 1:5
    cmap = parula(length(focIntEv{exper}));
    subplot(2,5,exper+5)
    scatter(focIntEv{exper}/abs(min(focIntEv{exper})),surrIntEv{exper}/max(surrIntEv{exper}),[],cmap,'.')
    [rval, pval] = corr(focIntEv{exper}',surrIntEv{exper}');
    title(['r=' num2str(rval) ' p=' num2str(pval)])
end

% Plot focus and surround area for each event and bin medians
figure
for exper = 1:5
    subplot(2,5,exper)
    scatter(1:length(focIntEv{exper}),-1*numFocElecEv{exper},'.','MarkerEdgeColor',[0 0.7 0],'MarkerEdgeAlpha',0.5); hold on
    scatter(1:length(surrIntEv{exper}),numSurrElecEv{exper},'.','MarkerEdgeColor',[0.7 0 0.7],'MarkerEdgeAlpha',0.5); hold on
    plot(binMids(exper,:),-1*binFocArea{exper},'Color',[0 0 0],'LineWidth',2);
    plot(binMids(exper,:),binSurrArea{exper},'Color',[0 0 0],'LineWidth',2);
    xlim([0 length(focIntEv{exper})])
    ylim([-32 32])
end

% Look at correlation between focus and surround area using every event
for exper = 1:5
    cmap = parula(length(focIntEv{exper}));
    subplot(2,5,exper+5)
    scatter(numFocElecEv{exper},numSurrElecEv{exper},[],cmap,'.')
    [rval, pval] = corr(numFocElecEv{exper}',numSurrElecEv{exper}');
    title(['r=' num2str(rval) ' p=' num2str(pval)])
end

% Look at correlation between focus/surround intensity and area (number of electrodes)
figure
for exper = 1:5
    cmap = parula(length(focIntEv{exper}));
    subplot(2,5,exper)
    scatter(numFocElecEv{exper},focIntEv{exper},[],cmap,'.')
    [rval, pval] = corr(numFocElecEv{exper}',focIntEv{exper}');
    title(['r=' num2str(rval) ' p=' num2str(pval)])
    subplot(2,5,exper+5)
    scatter(numSurrElecEv{exper},surrIntEv{exper},[],cmap,'.')
    [rval, pval] = corr(numSurrElecEv{exper}',surrIntEv{exper}');
    title(['r=' num2str(rval) ' p=' num2str(pval)])
end

% Look at the correlation between focus and surround for intensity (Z) and area of all events across all experiments
figure
subplot(211)
cmap5 = [9 46 86; 5 127 140; 90 211 149; 103 160 12; 197 148 5]/255;
for exper = 1:5    
scatter(focIntEvZ{exper},surrIntEvZ{exper},200,cmap5(exper,:),'.'); hold on
end
[rval, pval] = corr(cell2mat(focIntEvZ')',cell2mat(surrIntEvZ')');
title(['Intensity Z to Exp, Pearson r = ' num2str(rval) ', p = ' num2str(pval)])

subplot(212)
for exper = 1:5    
scatter(numFocElecEv{exper},numSurrElecEv{exper},200,cmap5(exper,:),'.'); hold on
end
[rval, pval] = corr(cell2mat(numFocElecEv')',cell2mat(numSurrElecEv')');
title(['Area, Pearson r = ' num2str(rval) ', p = ' num2str(pval)])

%% Plot Delay Maps

%% By threshold crossing
expThresh = zeros(5,1);
delayTimeAllNeg = cell(5,1);
delayTimeAllNegLag = cell(5,1); 
delayMedianNegLag = nan(5,32); 
delayMedianNegLagInFoc = nan(5,32); %delayMedianNegHalfNan = nan(5,32); 
for exper = 1:5    
    expThresh(exper) = 6*std(reshape(baseClipsNoMean{exper},1,[]));    
    delayTimeAllNeg{exper} = nan(32,size(fsClips{exper},2));
    delayTimeAllNegLag{exper} = nan(32,size(fsClips{exper},2));
    for event = 1:size(fsClips{exper},2)
        tmpLagNeg = nan(32,1);
        for elec = 1:32            
            if sum(binMMS{exper}(elec,:)<=focThresh)>0
            tmpClip = squeeze(fsClipsNorm{exper}(elec,event,:))'-repmat(fsClipsNorm{exper}(elec,event,1),1,201);
            tmpCrossBinNeg = tmpClip<-1*expThresh(exper);
            tmpCrossLocNeg = strfind(tmpCrossBinNeg,[0 1])+1;
            if ~isempty(tmpCrossLocNeg); tmpLagNeg(elec) = tmpCrossLocNeg(1); end
            end
        end
        delayTimeAllNeg{exper}(:,event) = tmpLagNeg*dtds(exper);        
        delayTimeAllNegLag{exper}(:,event) = (tmpLagNeg-min(tmpLagNeg))*dtds(exper);
    end
    delayMedianNegLag(exper,:) = nanmedian(delayTimeAllNegLag{exper},2);    
    %halfNanNeg = sum(isnan(delayTimeAllNeg{exper}),2)<size(fsClips{exper},2)/2;    
    %delayMedianNegHalfNan(exper,halfNanNeg) = delayMedianNeg(exper,halfNanNeg);
    %delayMedianNegLagInFoc(exper,(sum(binMMS{exper}<focThresh,2)>0)) = delayMedianNeg(exper,(sum(binMMS{exper}<focThresh,2)>0));
end

%% Plot delay map
colormap([0.5 0.5 0.5; jet(30)]);
for exper = 1:5       
    subplot(3,2,exper); 
    imagesc(reshape(delayMedianNegLag(exper,:),4,8)); title(num2str(exper)); caxis([-1 30])
    cb = colorbar; set(cb,'YTick',[0 10 20 30]);
    %imagesc(reshape(delayMedianNeg(exper,:),4,8)); colorbar; title(num2str(exper)); caxis([-1 30])
    set(gca,'Ytick',[],'xtick',[],'box','off'); axis off;
end

%% Calculate speeds
allFocSpeed = cell(5,1);
medianSpeed = zeros(1,5); speedPrctiles = zeros(5,4);
allSpeeds = []; speedGroup = [];
for exper = 1:5
    tmpFocElecs = find(~isnan(delayMedianNegLag(exper,:)));
    elecCombs = combnk(tmpFocElecs,2);
    allFocSpeed{exper} = nan(size(fsClips{exper},2),length(elecCombs));
    for event = 1:size(fsClips{exper},2)
        for i = 1:length(elecCombs)
            e1 = elecCombs(i,1); e2 = elecCombs(i,2);
            tmpDist = sqrt((ceil(e1./4)-ceil(e2./4))^2 + ((mod(e1,4)+1)-(mod(e2,4)+1))^2);
            tmpTime = abs(delayTimeAllNegLag{exper}(e1,event)-delayTimeAllNegLag{exper}(e2,event));
            if tmpTime > 0
                allFocSpeed{exper}(event,i) = tmpDist/tmpTime;
            end
        end
    end
    medianSpeed(exper) = nanmedian(allFocSpeed{exper}(:));
    speedPrctiles(exper,:) = prctile(allFocSpeed{exper}(:),[2.5 25 75 97.5]);
    allSpeeds = [allSpeeds; allFocSpeed{exper}(:)];
    speedGroup = [speedGroup; exper*ones(numel(allFocSpeed{exper}),1)];
end
boxplot(allSpeeds,speedGroup);

%% Get lag times by neg crossing in each bin
expThresh = zeros(5,1);
delayTimeNegBin = cell(5,24); delayTimeNegBinMedian = zeros(5,32,24);
tmpMap = jet(30); %tmpMap = parula(24);
for exper = 1:5
    figure(exper)
    expThresh(exper) = 6*std(reshape(baseClipsNoMean{exper},1,[]));
    for b = 1:24        
        %subplot(4,6,b); colormap([0.5 0.5 0.5; flipud(tmpMap(3:27,:))]);
        delayTimeNegBin{exper,b} = nan(32,length(binPts{exper}{b}));
        for evNum = 1:length(binPts{exper}{b})
            event = binPts{exper}{b}(evNum);
            tmpLagNeg = nan(32,1); 
            for elec = 1:32
                %tmpClip = squeeze(fsClipsNorm{exper}(elec,event,:))';
                tmpClip = squeeze(fsClipsNorm{exper}(elec,event,:))'-repmat(fsClipsNorm{exper}(elec,event,1),1,201);
                tmpCrossBinNeg = tmpClip<-1*expThresh(exper);
                tmpCrossLocNeg = strfind(tmpCrossBinNeg,[0 1])+1;
                if ~isempty(tmpCrossLocNeg); tmpLagNeg(elec) = tmpCrossLocNeg(1); end
            end
            delayTimeNegBin{exper,b}(:,evNum) = (tmpLagNeg-min(tmpLagNeg))*dtds(exper);            
        end
        delayTimeNegBinMedian(exper,:,b) = nanmedian(delayTimeNegBin{exper,b},2);
        %noiseVec = normrnd(0,0.1,1,32);
        %scatter((1:32)+noiseVec, squeeze(delayTimeNegBinMedian(exper,:,b)),'o','filled','MarkerEdgeColor',tmpMap(b,:),'MarkerFaceColor',tmpMap(b,:))
        %imagesc(reshape(delayTimeNegBinMedian(exper,:,b),4,8)); colorbar; title([num2str(exper) ' - ' num2str(b)]); caxis([-1 24])
    end
end

%% Other plots


%% Plot consecutive first spikes offest from black to gray (Figure 6)
for exper = 1:5
    figure
    tmpx = 1:201;
    cmap = bone(size(fsClipsNorm{exper},2));
    %cmap = bone(size(fsClipsNorm{exper},2)+50);
    cmap1 = linspace(0,0.9,size(fsClipsNorm{exper},2));
    %for elec = 1:32
    elec = thElecExp(exper);%4;
    for event = 1:size(fsClipsNorm{exper},2)
        %plot(tmpx+event-1,squeeze(fsClipsNorm{exper}(elec,event,:))','Color',cmap(event,:))
        plot(tmpx+0.5*(event-1),squeeze(fsClipsNorm{exper}(elec,event,:))','Color',cmap1(event)*[1 1 1])
        hold on
    end
    title(['Exp ' num2str(exper) ' - Elec' num2str(elec)])
    %   pause; clf
end

%% Scatter Event MMS and plot bin median MMS over top (Figure 7)
binMids = zeros(5,24);
for exper = 1:5; for i = 1:24; binMids(exper,i) = median(binPts{exper}{i}); end; end

exper = 1;
elecs = [4 18 7];%1:32;%
subIdx = reshape(1:32,8,[])'; subIdx = subIdx(:);
for i = 1:length(elecs)
    subplot(1,length(elecs),i) %subplot(4,8,subIdx(i))
    scatter(1:length(fsMMS{exper}),fsMMS{exper}(elecs(i),:),'.', 'MarkerEdgeColor', [0.4 0.4 0.4]); hold on
    plot(binMids(exper,:), binMMS{exper}(elecs(i),:),'k-d','MarkerFaceColor','k','LineWidth',1.5) %
    line([0 length(fsMMS{exper})+1], focThresh*[1 1],'LineStyle','--','Color','k')
    line([0 length(fsMMS{exper})+1], surrThresh*[1 1],'LineStyle','--','Color','k')
    xlim([0 length(fsMMS{exper})+1]); ylim([min(min(fsMMS{exper}(elecs,:))) max(max(fsMMS{exper}(elecs,:)))])
    xlabel('Event Number'); ylabel('MMS'); 
    title(['Event and Bin MMS - Exp ' num2str(exper) ' Elec ' num2str(elecs(i))])
    %title(['Exp ' num2str(exper) ' Elec ' num2str(elecs(i))])
end


%% Fig 7 example spikes
exper = 1;
evs = [76 168 269];
elecs = [4 7 18];
for i = 1:3 
    subplot(3,1,i)
    plot((-40:160).*dtds(exper),squeeze(fsClipsNorm{exper}(elecs(i),evs,:)),'k','LineWidth',2); hold on
        line([-40 160].*dtds(exper), [0 0], 'Color', 0.5*[1 1 1])
        xlim([-40 160].*dtds(exper));
        ylim([-5 2]);
end
