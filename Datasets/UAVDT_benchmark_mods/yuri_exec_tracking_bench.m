%% LIMPAR VARS LOCAIS E IMPORTAR LISTA DE DETECTORES
clear;
clc;
addpath(genpath('.'));
[subFoldersNames_toTrack] = yuri_gen_list_of_detections_to_track_bench();
[~, numOfFolders] = size(subFoldersNames_toTrack);

seqDirs_overall = {'M0203','M0205','M0208','M0209','M0403','M0601','M0602','M0606','M0701','M0801',...
                   'M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401'}; % overall testing sequences

%% Track Bench
fprintf('\n[INFO] Início da Análise Tracking. %d resultados de trackers serão analisados.\n', numOfFolders);

for k=1:numOfFolders
    detectorName = subFoldersNames_toTrack{1,k};
    trackerName = subFoldersNames_toTrack{2,k};
    fprintf('Analisando Tracking. Det: %s, Tracker: %s ... (Tracker %d de %d)\n', detectorName, trackerName, k, numOfFolders);
    seqDirs = seqDirs_overall;
    resultsDir = ['./RES_MOT/' detectorName '/' trackerName '/'];
    benchmarkDir = './GT/';
    [allMets, metsBenchmark] = evaluateTracking(seqDirs, resultsDir, benchmarkDir, 'UAVDT', 'overall');
end
fprintf('[INFO] FIM da Análise de Tracking.\n');

%% Best Trackers Sequence MOTA

seqDirs_day = {'M0208','M0209','M0403','M0601','M0602','M0606','M0801','M0802','M1301','M1302','M1303','M1401'}; % daylight
seqDirs_night = {'M0203','M0205','M1001','M1007','M1101'}; % night
seqDirs_fog = {'M0701','M1004','M1009'}; % fog

seqDirs_low = {'M0203','M0205','M0209','M0801','M0802','M1101','M1303','M1401'}; % low altitude
seqDirs_medium = {'M0208','M0403','M0602','M0606','M1001','M1004','M1007','M1009','M1301','M1302'}; % medium altitude
seqDirs_high = {'M0601','M0701'}; % high altitude

seqDirs_front = {'M0208','M0209','M0403','M0601','M0602','M0606','M1001','M1004','M1007','M1101','M1301','M1401'}; % front view
seqDirs_side = {'M0205','M0209','M0403','M0601','M0606','M0802','M1101','M1302','M1303'}; % side view
seqDirs_bird = {'M0203','M0701','M0801','M1009'}; % bird view

seqDirs_long = {'M0209','M1001'}; % long sequence

% Pick Best MOTA Selected Models
yuri_best_mota_overall_models; 

seqs = {{'seqDirs_overall','seqDirs_day', 'seqDirs_night', 'seqDirs_fog', 'seqDirs_low', ...
        'seqDirs_medium', 'seqDirs_high', 'seqDirs_front', 'seqDirs_side',...
        'seqDirs_bird', 'seqDirs_long'},...
        {seqDirs_overall, seqDirs_day, seqDirs_night, seqDirs_fog, seqDirs_low, ...
        seqDirs_medium, seqDirs_high, seqDirs_front, seqDirs_side,...
        seqDirs_bird, seqDirs_long}};
    
[~, numOfDets] = size(yuri_best_mota_overall_selected_det);
[~, numOfSeqs] = size(seqs{1,2});

for k=1:numOfDets
    detectorName = yuri_best_mota_overall_selected_det{1,k};
    trackerName = yuri_best_mota_overall_selected_tracker{1,k};
    
    for j=1:numOfSeqs
        seqName = seqs{1,1}{j};
        seqDirs = seqs{1,2}{j};
        fprintf('Analisando Tracking. Det: %s, Tracker: %s, Seq: %s ... \n', detectorName, trackerName,seqName);
        resultsDir = ['./RES_MOT/' detectorName '/' trackerName '/'];
        benchmarkDir = './GT/';
        [allMets, metsBenchmark] = evaluateTracking(seqDirs, resultsDir, benchmarkDir, 'UAVDT', seqName);
    end

end





