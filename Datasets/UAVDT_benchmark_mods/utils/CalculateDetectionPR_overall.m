function CalculateDetectionPR_overall(detection)

gtDataDir = './GT/';
resDateDir = './RES_DET';
seqDirs = {'M0203','M0205','M0208','M0209','M0403','M0601','M0602','M0606','M0701','M0801','M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401'};
seqLens = [1007,646,265,1576,514,372,480,1374,1308,298,1101,1859,269,659,604,864,1182,719,445,1050];

attribute.name = [];
attribute.rec = [];
attribute.prec = [];
attribute.AP = [];

detectorSet = {detection};
seqSet = {1:20};

for idDet = 1:length(detectorSet)

    detectorName = detectorSet{idDet};
    %% detection threshold configuration

    idSeq = 0;
    allgtMatch = [];
    alldetMatch = [];
    allFrames = 0;
    
    while(idSeq<21)
        idSeq = idSeq + 1;

        curSeq = seqSet{idDet};
        if(ismember(idSeq, curSeq))
            numFrames = seqLens(idSeq);
            sequenceName = seqDirs{idSeq};
            
            gt = load(sprintf('%s%s%s',gtDataDir,sequenceName,'_gt_whole.txt'));

            detections = load(sprintf('%s/%s/%s%s',resDateDir,detection,sequenceName,'.txt'));
            file_gt_ign = sprintf('%s%s%s',gtDataDir,sequenceName,'_gt_ignore.txt');
            pass = {'M0203','M0205','M0208','M0403','M0601','M0602','M0606','M0701','M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401'};
            % ignore the detections in the ignored regions
            if ismember(sequenceName, pass)
                gt_ign = dlmread(file_gt_ign);
                for gt_i = 1:length(gt_ign(:,1))
                    detections(detections(:,1) == gt_ign(gt_i,1) & detections(:,3)>gt_ign(gt_i,3) & detections(:,4)>gt_ign(gt_i,4) & detections(:,3)+detections(:,5)<gt_ign(gt_i,3)+gt_ign(gt_i,5) & detections(:,4)+detections(:,6)<gt_ign(gt_i,4)+gt_ign(gt_i,6),:) = [];
                end
            end
            
            gtMatch = [];
            detMatch = [];
            for fr = 1:numFrames
                curLineGt = gt(:,1) == fr;
                curLineDet = detections(:,1) == fr;
                gt0 = gt(curLineGt,3:6);
                
                gt0 = cat(2,gt0,zeros(size(gt0,1),1));
                dt0 = detections(curLineDet,3:7);
                
                [gt1, dt1] = evalRes(gt0, dt0);
                gtMatch = cat(1, gtMatch, gt1(:,5));
                detMatch = cat(1, detMatch, dt1(:,5:6));
            end
            allgtMatch = cat(1, allgtMatch, gtMatch);
            alldetMatch = cat(1, alldetMatch, detMatch);
            allFrames = allFrames + numFrames;
        end
    end
    
    [~,idrank] = sort(-alldetMatch(:,1));
    fp=cumsum(alldetMatch(idrank,2)==0);
    tp=cumsum(alldetMatch(idrank,2)==1);
    rec=tp/max(1,numel(allgtMatch));
    prec=tp./max(1,(fp+tp));
    AP = roundn(VOCap(rec,prec)*100,-2);
    
    fprintf('    %s AP_overall: %.2f ...\n', detectorName, AP);
    %figure(),plot(rec,prec,'LineWidth',4);
    %axis([0 1 0 1]);

    %xlabel('recall'),ylabel('precision'),title([detectorName ' AP = ' num2str(AP)]);
    
    attribute.name{idDet} = [detectorName]; %
    attribute.rec{idDet} = rec;
    attribute.prec{idDet} = prec;
    attribute.AP{idDet} = AP;
end

save(sprintf('./det_EVA/%s_overall.mat',detection),'attribute');