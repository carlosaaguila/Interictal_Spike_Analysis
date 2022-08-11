%% Load Data
cd('\\borel.seas.upenn.edu\g\public\USERS\binkh\Depth and Surface Testing\Paper 1 Seizure Model\Data')
load('AllExpTimingSec'); load('dtds.mat'); load('AllExpThElecData'); %load('allExpStimPts'); load('allExpAnnotations')
load('temporalGeneral.mat'); load('temporalAdvanced.mat'); load('detectorVars.mat');

%% Event Detector Examples
exExps = [1 1];
exStartsS = [1218 1496]; exEndsS = [1235 1515]; %events 204-209 for polyspikes, event 235 for sz 
for i = 1:length(exExps)
    subplot(2,1,2)
    exStart = find(allExpTimingSec{exExps(i)}>exStartsS(i),1);
    exEnd = find(allExpTimingSec{exExps(i)}>exEndsS(i),1);
    plot(allExpTimingSec{exExps(i)}(exStart:exEnd),allExpThElecData{exExps(i)}(exStart:exEnd)/1000,'Color',[0.5 0.5 0.5])
    hold on
    ylm = [min(allExpThElecData{exExps(i)}(exStart:exEnd)/1000)-0.2 max(allExpThElecData{exExps(i)}(exStart:exEnd)/1000)+0.2]; 
    ylim(ylm)
    tmpEvStarts = find(evStart{exExps(i)}>exStart & evStart{exExps(i)}<exEnd);
    for k = tmpEvStarts
        line(allExpTimingSec{exExps(i)}(evStart{exExps(i)}(k))*[1 1], ylm,'Color','g');
        line(allExpTimingSec{exExps(i)}(evStop{exExps(i)}(k))*[1 1], ylm,'Color','r');
    end
    xlim([exStartsS(i) exEndsS(i)])  
    set(gca, 'box', 'off','FontSize',10)
    
    subplot(2,1,1)
    exStartSD = find(sdFeatLongTime{exExps(i)}>exStartsS(i),1);
    exEndSD = find(sdFeatLongTime{exExps(i)}>exEndsS(i),1);       
    plot(sdFeatLongTime{exExps(i)}(exStartSD:exEndSD),sdFeatLong{exExps(i)}(exStartSD:exEndSD)/1000,'m')    
    hold on
    plot(sdFeatLongTime{exExps(i)}(exStartSD:exEndSD),sdFeatLongNoMem{exExps(i)}(exStartSD:exEndSD)/1000,'b') 
    line([exStartsS(i) exEndsS(i)],memThreshExp(exExps(i))*[1 1]/1000,'Color','k','LineStyle','--')
    ylm2 = [-0.5 max(sdFeatLong{exExps(i)}(exStartSD:exEndSD))/1000+0.2];
    ylim(ylm2)
    for k = tmpEvStarts
        line(allExpTimingSec{exExps(i)}(evStart{exExps(i)}(k))*[1 1], ylm2,'Color','g');
        line(allExpTimingSec{exExps(i)}(evStop{exExps(i)}(k))*[1 1], ylm2,'Color','r');
    end
    xlim([exStartsS(i) exEndsS(i)])    
    set(gca,'xticklabel',[],'xtick',[])
    set(gca, 'box', 'off')
    
%     set(gcf,'color','white','Position',[50 50 1200 900],'PaperPosition', [.25 .25 7 4])
%     frameName = ['szDetV3part' num2str(i)];
%     print(gcf,'-dpng',frameName)
    
    pause
    clf
end

%% Spike Detector Examples
exper = 1; event = 206; %219;
line([allExpTimingSec{exper}(evStart{exper}(event)) allExpTimingSec{exper}(evStop{exper}(event))],[0 0],'Color','k')
hold on
plot(allExpTimingSec{exper}(evStart{exper}(event):evStop{exper}(event)),allExpThElecData{exper}(evStart{exper}(event):evStop{exper}(event))/1000,'Color',[0.5 0.5 0.5])
for k = 1:numSpikes{exper}(event)
    plot(allExpTimingSec{exper}(spikeStart{exper}{event}(k)),-6*avgSubThreshSTD(exper)*[1 1]/1000 ,'g*')
    plot(allExpTimingSec{exper}(spikeStop{exper}{event}(k)),-6*avgSubThreshSTD(exper)*[1 1]/1000 ,'r*')
end
line([allExpTimingSec{exper}(evStart{exper}(event)) allExpTimingSec{exper}(evStop{exper}(event))],-6*avgSubThreshSTD(exper)*[1 1]/1000,'Color','k','Linestyle','--')
xlim([allExpTimingSec{exper}(evStart{exper}(event)) allExpTimingSec{exper}(evStop{exper}(event))])
xlabel('Time (s)'); ylabel('Voltage (mV)')
title([ 'Experiment ' num2str(exper) ' | Event ' num2str(event) ' | # of Spikes = ' num2str(numSpikes{exper}(event))])
set(gca, 'box', 'off','LineWidth',1,'FontSize',12)


%% Each Event Example
%exExps = [1 5 2]; exStartsS = [553 4995.5 3381]; exEndsS = [557 4997.5 3403];
exExps = [1 1 1]; exStartsS = [553 4995.5 3381]; exEndsS = [557 4997.5 3403];
for i = 1:length(exExps)
    subplot(3,1,i)
    exStart = find(allExpTimingSec{exExps(i)}>exStartsS(i),1);
    exEnd = find(allExpTimingSec{exExps(i)}>exEndsS(i),1);
    plot(allExpTimingSec{exExps(i)}(exStart:exEnd),allExpThElecData{exExps(i)}(exStart:exEnd)/1000,'Color',[0.5 0.5 0.5])    
    xlim([exStartsS(i) exEndsS(i)])  
    set(gca, 'box', 'off','LineWidth',1,'FontSize',10)            
end

%% Event examples from each experiment for each type
%exStartSsps = [619.2 842.2; 409.2 2624.4; 851 1434.6; 272.3 1875.5; 1241.4 4048.2;];    
% With bursts from seizures in the same setup as polyspikes for comparison
exStartSsps = [619.2 1843.3; 409.2 3332.8; 851 1255.6; 272.3 4450.6; 1241.4 4626.7;];    
exStartSz = [1832 3315 1248 4441 4606];

ctr = 1;
for exper = 1:5
    figure(1)
    for typ = 1:2
        subplot(5,2,ctr); hold on
        exStart = find(allExpTimingSec{exper}>exStartSsps(exper,typ),1);
        exEnd = find(allExpTimingSec{exper}>(exStartSsps(exper,typ)+1),1);
        line([exStartSsps(exper,typ) exStartSsps(exper,typ)+1], [0 0], 'Color', [0.7 0.7 0.7])
        plot(allExpTimingSec{exper}(exStart:exEnd),allExpThElecData{exper}(exStart:exEnd)/1000,'Color',[0.3 0.3 0.3])
        if exper == 1; line([exStartSsps(exper,typ) exStartSsps(exper,typ)+0.2], [-4 -4]); line([1 1]*(exStartSsps(exper,typ)+0.1), [-4 -2]); end
        xlim([exStartSsps(exper,typ) exStartSsps(exper,typ)+1])
        ylim([-5 3])
        set(gca, 'box', 'off','LineWidth',1,'FontSize',10)
        set(gcf,'Position',[10 215 735 800])
        ctr = ctr+1;
    end
    figure(2)
    subplot(5,1,exper); hold on
    exStartZ = find(allExpTimingSec{exper}>exStartSz(exper),1);
    exEndZ = find(allExpTimingSec{exper}>(exStartSz(exper)+25),1);
    line([exStartSz(exper) exStartSz(exper)+25], [0 0], 'Color', [0.7 0.7 0.7])
    plot(allExpTimingSec{exper}(exStartZ:exEndZ),allExpThElecData{exper}(exStartZ:exEndZ)/1000,'Color',[0.3 0.3 0.3])
    if exper == 1; line([exStartSz(exper) exStartSz(exper)+2], [-4 -4]); line([1 1]*(exStartSz(exper)+1), [-4 -2]); end
    xlim([exStartSz(exper) exStartSz(exper)+25])
    ylim([-5 3])
    set(gca, 'box', 'off','LineWidth',1,'FontSize',10)
    set(gcf,'Position',[750 215 735 800])
    %frameName = ['AllExpEventExampleType' num2str(typ) 'Exp' num2str(exper) '.svg'];
    %title(['Type' num2str(typ) 'Exp ' num2str(exper)])
    %saveas(gcf,frameName)
    %clf
end
%frameName = ['TestType' num2str(typ) '.svg'];
%tmpFig = figure(typ);
%tmpFig.Renderer = 'Painters';
%saveas(tmpFig,frameName)


%% Get indices for single spikes, polyspikes and seizures
singleIdx = cell(5,1); polyIdx = cell(5,1); szIdx = cell(5,1);
singleStartSec = cell(5,1); polyStartSec = cell(5,1); szStartSec = cell(5,1);
numSingle = zeros(5,1); numPoly = zeros(5,1); numSz = zeros(5,1);
numSpikesAllPoly = []; numSpikesAllSz = [];
for exper = 1:5
        singleIdx{exper} = numSpikes{exper}==1;
        polyIdx{exper} = numSpikes{exper}<20 & numSpikes{exper}>1;
        szIdx{exper} = numSpikes{exper}>=20;
        
        numSingle(exper) = sum(singleIdx{exper});
        numPoly(exper) = sum(polyIdx{exper});
        numSz(exper) = sum(szIdx{exper});
        
        numSpikesAllPoly = [numSpikesAllPoly numSpikes{exper}(polyIdx{exper})];
        numSpikesAllSz = [numSpikesAllSz numSpikes{exper}(szIdx{exper})];
        
        singleStartSec{exper} = allExpTimingSec{exper}(evStart{exper}(singleIdx{exper}));
        polyStartSec{exper} = allExpTimingSec{exper}(evStart{exper}(polyIdx{exper}));
        szStartSec{exper} = allExpTimingSec{exper}(evStart{exper}(szIdx{exper}));
end

disp(['Of ' num2str(length(cell2mat(evStart))) ' detected events, there were: '...
    char(10) num2str(sum(numSingle)) ' Single Spikes (' num2str(100*sum(numSingle)/length(cell2mat(evStart))) '%)'...
    char(10) num2str(sum(numPoly)) ' Polyspikes (' num2str(100*sum(numPoly)/length(cell2mat(evStart))) '%) Median of ' num2str(median(numSpikesAllPoly)) ' spikes/event'...
    char(10) num2str(sum(numSz)) ' Seizures (' num2str(100*sum(numSz)/length(cell2mat(evStart))) '%) Median of ' num2str(median(numSpikesAllSz)) ' spikes/event'])

singleExpStartsMin = [singleStartSec{1}(1) singleStartSec{2}(1) singleStartSec{3}(1) singleStartSec{4}(1) singleStartSec{5}(1)]/60;
polyExpStartsMin = [polyStartSec{1}(1) polyStartSec{2}(1) polyStartSec{3}(1) polyStartSec{4}(1) polyStartSec{5}(1)]/60;
szExpStartsMin = [szStartSec{1}(1) szStartSec{2}(1) szStartSec{3}(1) szStartSec{4}(1) szStartSec{5}(1)]/60;

tenPolyFirst = zeros(5,10); tenPolyBeforeSz = zeros(5,10);
for exper = 1:5
    tenPolyFirst(exper,:) = numSpikes{exper}(find(polyIdx{exper}>0,10,'first'));
    tmpPoly = polyIdx{exper}(1:find(szIdx{exper}>0,1,'first'));
    tenPolyBeforeSz(exper,:) = numSpikes{exper}(find(tmpPoly>0,10,'last'));
end

numSingleBeforeSz = zeros(1,5); numSingleAfterSz = zeros(1,5);
for exper = 1:5
    numSingleBeforeSz(exper) = sum(singleIdx{exper}(1:find(szIdx{exper}>0,1,'first')));
    numSingleAfterSz(exper) = sum(singleIdx{exper}(find(szIdx{exper}>0,1,'first'):end));
end
    
%% Scatter # spikes vs experiment time of each event
xlabs = {10:10:40; 30:30:120; 15:15:45; 30:30:120; 30:30:120}; 
for exper = 1:5
    subplot(5,1,exper)
    hold on
    line([0 allExpTimingSec{exper}(evStart{exper}(end))/60],[20 20],'LineStyle','--','Color',[0.7 0.7 0.7])    
    plot(singleStartSec{exper}/60,numSpikes{exper}(singleIdx{exper}),'b.')%,'MarkerSize',3)
    plot(polyStartSec{exper}/60,numSpikes{exper}(polyIdx{exper}),'g.')%,'MarkerSize',3)
    plot(szStartSec{exper}/60,numSpikes{exper}(szIdx{exper}),'r.')%,'MarkerSize',3)
    set(gca,'YScale','log')
    set(gca,'YTick',[1:9 10:10:90 100:100:500])
    labels = cell(1,length([1:9 10:10:90 100:100:500]));
    labels{1} = '1'; labels{10} = '10'; labels{19} = '100';
    set(gca,'YTickLabel',labels)    
    set(gca,'XTick',xlabs{exper})
    ylim([1 10000])
    xlim([0 allExpTimingSec{exper}(evStart{exper}(end))/60])
    %title(['Experiment ' num2str(exper)],'FontWeight','normal','FontSize',12)
    %if exper == 5; xlabel('Experiment Time (min)','FontSize',12); end
    %if exper == 3; ylabel('Number of Spikes per Event','FontSize',12); end
    %set(gca, 'box', 'off','LineWidth',1.5,'FontSize',12)
    set(gca, 'box', 'off','FontSize',7,'LineWidth',0.5)
end
% annsExp = [2 3 5]; annsTime = {[11 20], [5], [9 16]};
% for i = 1:length(annsExp)
%     exper = annsExp(i);
%     subplot(5,1,exper)
%     for j = 1:length(annsTime{i})
%         line((allExpAnnotations{exper}{1}(annsTime{i}(j))/60)*[1 1], [1 100],'Color','k')
%     end
% end
    %set(gcf,'color','white','Position',[50 50 1200 900],'PaperPosition', [.25 .25 8 9])
%     set(gcf,'color','white','Position',[50 10 765 1000],'PaperPosition', [.25 .25 7 5])
%     frameName = 'numSpikeScatterNoStimTrigV2';
%     print(gcf,'-dpng',frameName)

%% Histogram of # spikes overall (Determine polyspike vs. seizure by # spike threshold)
% histogram(cell2mat(numSpikesPool),[1:10 11:3:max(cell2mat(numSpikes))+3],'FaceColor',[0.5 0.5 0.5])
% hold on
histogram(cell2mat(numSpikes),[1:14 15:5:max(cell2mat(numSpikes))+5],'FaceColor',[0.5 0.5 0.5])
%histogram(cell2mat(numSpikes),[1:1:max(cell2mat(numSpikes))])
%[histVals,histEdges] = histcounts(cell2mat(numSpikes),[1:1:max(cell2mat(numSpikes))]);
%set(gca,'XTick',[0 50 100 150])    
set(gca, 'box', 'off','LineWidth',1,'FontSize',12)
%xlim([0 160])
ylim([0.9 10000])
set(gca,'yscale','log')

%% Boxplots of Seizure Vs Polyspike frequency and post-event delay
polyFreq = []; szFreq = [];
for exper = 1:5    
    polyFreq = [polyFreq numSpikes{exper}(polyIdx{exper})./evDurSec{exper}(polyIdx{exper})'];
    szFreq = [szFreq numSpikes{exper}(szIdx{exper})./evDurSec{exper}(szIdx{exper})'];
end

subplot(121)
boxplot([polyFreq szFreq],[zeros(1,length(polyFreq)) ones(1,length(szFreq))],'Labels',{'Polyspikes','Seizures'})
ylabel('Spike Frequency (spikes/s)')
set(gca,'YTick',[0 10 20 30 40])    
set(gca, 'box', 'off','LineWidth',1.5,'FontSize',12)

disp(['Median Ps Frequency = ' num2str(median(polyFreq)) 'Spikes/s'])
disp(['Median Sz Frequency = ' num2str(median(szFreq)) 'Spikes/s'])
[p,~] = ranksum(polyFreq,szFreq);
disp(['Ranksum p-value = ' num2str(p)])

ieiSecSP = []; ieiSecSz = [];
for exper = 1:5
    ieiSecSP = [ieiSecSP ieiSec{exper}(numSpikes{exper}(1:end-1)<20&(ieiSec{exper}<26)')'];
    ieiSecSz = [ieiSecSz ieiSec{exper}(numSpikes{exper}(1:end-1)>=20&(ieiSec{exper}<26)')'];
end

subplot(122)
boxplot([ieiSecSP ieiSecSz],[zeros(1,length(ieiSecSP)) ones(1,length(ieiSecSz))],'Labels',{'Single/Polyspikes','Seizures'})
set(findobj(gca,'Type','text'),'FontSize',12)
ylim([0 26])
set(gca,'YTick',[0 10 20])    
ylabel('Post-Event Delay (s)')
set(gca, 'box', 'off','LineWidth',1.5,'FontSize',12)

disp(['Median S/P Delay = ' num2str(median(ieiSecSP)) 's'])
disp(['Median Sz Delay = ' num2str(median(ieiSecSz)) 's'])
[p,~] = ranksum(ieiSecSP,ieiSecSz);
disp(['Ranksum p-value = ' num2str(p)])
    

%% Buildup between spikes example and boxplot
exper = 2;exStartsS = 4935; exEndsS = 5015;
    subplot(1,3,[1 2])
    exStart = find(allExpTimingSec{exper}>exStartsS,1);
    exEnd = find(allExpTimingSec{exper}>exEndsS,1);
    plot(allExpTimingSec{exper}(exStart:exEnd),allExpThElecData{exper}(exStart:exEnd)/1000,'Color',[0.5 0.5 0.5])  
    hold on;
    for n = 832:841; line([1 1]*allExpTimingSec{exper}(evStart{2}(n)), [-5 3], 'Color', 'g'); text(allExpTimingSec{exper}(evStart{2}(n)),3,num2str(numSpikes{2}(n))); end
    xlim([exStartsS exEndsS])  
    set(gca, 'box', 'off','FontSize',10)            

preSzAll = []; postSzAll = []; oneEvBwSz = [];
preCtr = 1; postCtr = 1; oneCtr = 1;
for exper = 1:5
    for event = 1:length(evStart{exper})
        if event<length(evStart{exper}) && numSpikes{exper}(event)<20 && numSpikes{exper}(event+1)>=20 && numSpikes{exper}(event-1)>=20
            oneEvBwSz = [oneEvBwSz numSpikes{exper}(event)];
        elseif event<length(evStart{exper}) && numSpikes{exper}(event)<20 && numSpikes{exper}(event+1)>=20
            preSzAll = [preSzAll numSpikes{exper}(event)];
        elseif event>1 && numSpikes{exper}(event-1)>=20 && numSpikes{exper}(event)<20            
            postSzAll = [postSzAll numSpikes{exper}(event)];
        end
    end
end

subplot(1,3,3)
boxplot([preSzAll postSzAll],[zeros(1,length(preSzAll)) ones(1,length(postSzAll))],'Labels',{'Pre-Seizure Events','Post-Seizure Events'});
set(gca,'YTick',[0 5 10 15])    
ylabel('Spikes/Event'); ylim([0 16])
set(gca, 'box', 'off','FontSize',10)

set(gcf,'color','white','Position',[50 50 1600 800],'PaperPosition', [.25 .25 16 10])
%frameName = ['numspike events before and after seizures Boxplot'];
%print(gcf,'-dpng',frameName)

disp(['Median Pre-Seizure = ' num2str(median(preSzAll)) ' spikes'])
disp(['Median Post-Seizure = ' num2str(median(postSzAll)) ' spikes'])
[p,~] = ranksum(preSzAll,postSzAll);
disp(['Ranksum p-value = ' num2str(p)])

%% Look at spike frequency within events
exper = 5;
tmpSz = find(szIdx{exper});
for i = 1:length(tmpSz)
    event = tmpSz(i);
%subplot(211); 
hold on
if numSpikes{exper}(event)>=20
plot(allExpThElecData{exper}(evStart{exper}(event):evStop{exper}(event))-i*500)
end
% title(['Exp' num2str(exper) ' Event ' num2str(event)])
% subplot(212); hold on
% plot(1./diff(allExpTimingSec{exper}(spikeStart{exper}{event})))
%pause; clf
end

% instSpikeFreqPS = cell(5,1); 
% for exper = 1:5
%     tmpPoly = find(polyIdx{exper});
%     instSpikeFreqPS{exper} = nan(length(tmpPoly),max(numSpikes{exper}(polyIdx{exper}))-1);
%     for evNum = 1:length(tmpPoly)
%         event = tmpPoly(evNum);
%         instSpikeFreqPS{exper}(evNum,1:(numSpikes{exper}(event)-1)) = 1./diff(allExpTimingSec{exper}(spikeStart{exper}{event}));
%     end
%     %subplot(2,5,exper); plot(instSpikeFreqPS{exper}'); xlim([0 10]); ylim([0 40]);
%     subplot(2,5,exper); plot(nanmean(instSpikeFreqPS{exper})); xlim([0 10]); ylim([0 15]);
% end
% instSpikeFreqSz = cell(5,1); szFreqBy10 = cell(5,1);
% for exper = 1:5
%     tmpSz = find(szIdx{exper});
%     instSpikeFreqSz{exper} = nan(length(tmpSz),max(numSpikes{exper}(szIdx{exper}))-1);
%     szFreqBy10{exper} = nan(length(tmpSz),max(numSpikes{exper}(szIdx{exper}))-10);
%     for evNum = 1:length(tmpSz)
%         event = tmpSz(evNum);
%         instSpikeFreqSz{exper}(evNum,1:(numSpikes{exper}(event)-1)) = 1./diff(allExpTimingSec{exper}(spikeStart{exper}{event}));
%         for i = 1:numSpikes{exper}(event)-10
%             szFreqBy10{exper}(evNum,i) = mean(instSpikeFreqSz{exper}(evNum,i:i+9));
%         end
%     end
%     %subplot(2,5,exper+5); plot(instSpikeFreqSz{exper}'); xlim([0 125]);% ylim([0 40]);
%     subplot(2,5,exper+5); plot(nanmean(instSpikeFreqSz{exper})); xlim([0 125]); ylim([0 15]);
% end

    
    