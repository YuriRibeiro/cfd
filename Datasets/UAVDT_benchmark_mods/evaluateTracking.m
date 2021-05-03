function [allMets, metsBenchmark, metsMultiCam] = evaluateTracking(seqmap, resDir, gtDataDir, benchmark, identifier)

% Input:
% - seqmap
% Sequence map.
%
% - resDir
% The folder containing the tracking results. Each one should be saved in a
% separate .txt file with the name of the respective sequence (see ./res/data)
%
% - gtDataDir
% The folder containing the ground truth files.
%
%
% Output:
% - allMets
% Scores for each sequence
% 
% - metsBenchmark
% Aggregate score over all sequences
%
% - metsMultiCam
% Scores for multi-camera evaluation
% 
% addpath(genpath('.'));
% warning off;

% Benchmark specific properties
world = 0;
threshold = 0.5;

% Read sequence list
sequenceListFile = seqmap;
allSequences = parseSequences2(sequenceListFile);
gtMat = [];
resMat = [];

% Evaluate sequences individually
allMets = [];
metsBenchmark = [];
metsMultiCam = [];

for ind = 1:length(allSequences)
    
        sequenceName = char(allSequences(ind));
        fprintf('\t... %s\n',sequenceName);

        gtFilename = sprintf('%s%s%s',gtDataDir,sequenceName,'_gt.txt');
        gtdata = dlmread(gtFilename);
        gtdata(gtdata(:,7)==0,:) = [];     % ignore 0-marked GT
        gtdata(gtdata(:,1)<1,:) = [];      % ignore negative frames

        [~, ~, ic] = unique(gtdata(:,2)); % normalize IDs
        gtdata(:,2) = ic;
        gtMat{ind} = gtdata;

        % Parse result
        resFilename = [resDir, sequenceName,  '.txt'];
        if strcmp(benchmark, 'UAVDT')
            resFilename = preprocessResult(resFilename, sequenceName, gtDataDir);
        end
        
        % Skip evaluation if output is missing
        if ~exist(resFilename,'file')
            error('Invalid submission. Result for sequence %s not available!\n', sequenceName);
        end
        
        % Read result file
        if exist(resFilename,'file')
            s = dir(resFilename);
            if s.bytes ~= 0
                resdata = dlmread(resFilename);
            else
                resdata = zeros(0,9);
            end
        else
            error('Invalid submission. Result file for sequence %s is missing or invalid\n', resFilename);
        end
        resdata(resdata(:,1)<1,:) = [];      % ignore negative frames
        resdata(resdata(:,1) > max(gtMat{ind}(:,1)),:) = []; % clip result to gtMaxFrame
        
        file_gt_ign = sprintf('%s%s%s',gtDataDir,sequenceName,'_gt_ignore.txt');
        pass = {'M0203','M0205','M0208','M0403','M0601','M0602','M0606','M0701','M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401'};
        if ismember(sequenceName, pass)
            gt_ign = dlmread(file_gt_ign);
            for gt_i = 1:length(gt_ign(:,1))
                resdata(resdata(:,1) == gt_ign(gt_i,1) & resdata(:,3)>gt_ign(gt_i,3) & resdata(:,4)>gt_ign(gt_i,4) & resdata(:,3)+resdata(:,5)<gt_ign(gt_i,3)+gt_ign(gt_i,5) & resdata(:,4)+resdata(:,6)<gt_ign(gt_i,4)+gt_ign(gt_i,6),:) = [];
            end
        end
      
        resMat{ind} = resdata;
    
    % Sanity check
    frameIdPairs = resMat{ind}(:,1:2);
    [u,I,~] = unique(frameIdPairs, 'rows', 'first');
    hasDuplicates = size(u,1) < size(frameIdPairs,1);
    if hasDuplicates
        ixDupRows = setdiff(1:size(frameIdPairs,1), I);
        dupFrameIdExample = frameIdPairs(ixDupRows(1),:);
        rows = find(ismember(frameIdPairs, dupFrameIdExample, 'rows'));
        
        errorMessage = sprintf('Invalid submission: Found duplicate ID/Frame pairs in sequence %s.\nInstance:\n', sequenceName);
        errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(1),:)), newline];
        errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(2),:)), newline];
        assert(~hasDuplicates, errorMessage);
    end
    
    % Evaluate sequence
    [metsCLEAR, mInf, additionalInfo] = CLEAR_MOT_HUN(gtMat{ind}, resMat{ind}, threshold, world);
    metsID = IDmeasures(gtMat{ind}, resMat{ind}, threshold, world);
    mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];
    allMets(ind).name = sequenceName;
    allMets(ind).m    = mets;
    allMets(ind).IDmeasures = metsID;
    allMets(ind).additionalInfo = additionalInfo;

    printMetrics(mets); fprintf('\n');
    
    if strcmp(identifier, 'seqDirs_overall')
        evalFile = fullfile(resDir, sprintf('eval_%s.txt',sequenceName));
        dlmwrite(evalFile, mets);
    end
    
end

% Overall scores
metsBenchmark = evaluateBenchmark(allMets, world);

fprintf('\n');
fprintf(' ********************* Your %s Results *********************\n', benchmark);
printMetrics(metsBenchmark);

if strcmp(identifier, 'seqDirs_overall')
    evalFile = fullfile(resDir,'eval.txt');
else
    evalFile = fullfile(resDir, sprintf('eval_%s.txt',identifier));
end

dlmwrite(evalFile, metsBenchmark);