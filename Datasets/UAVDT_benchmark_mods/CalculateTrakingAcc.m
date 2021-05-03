clear, clc;
addpath(genpath('.'));

%% evaluate the tracker in different attribute
seqDirs_overall = {'M0203','M0205','M0208','M0209','M0403','M0601','M0602','M0606','M0701','M0801',...
                   'M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401'}; % overall testing sequences

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

%% Four Detectors: FRCNN, RFCN, RON, SSD
%% Eight Trackers: CEM, CMOT, DSORT, GOG, IOU, MDP, SMOT, SORT
detectorName = 'YOLOv5_UAVDT_5-26_Feb_2021_04h_26m-1280x1280';
trackerName = 'SORT';
seqDirs = seqDirs_overall;
resultsDir = ['./RES_MOT/' detectorName '/' trackerName '/'];

benchmarkDir = './GT/';

[allMets, metsBenchmark] = evaluateTracking(seqDirs, resultsDir, benchmarkDir, 'UAVDT');