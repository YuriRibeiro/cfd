% calculate the AP scores of detectors in terms of different attributes
addpath(genpath('.'));
detection = 'det_FRCNN'; % detector names, i.e., det_FRCNN, det_RFCN, det_RON, det_SSD
CalculateDetectionPR_overall(detection); % calculate the AP scores of the test-set.
CalculateDetectionPR_seq(detection); % calculate the AP scores in sequence-level attribute

% calculate the AP scores in object-level attribute
for obj_attr = 1:3
    % 1 for Vehicle Category;
    % 2 for Vehicle Occlusion;
    % 3 for Out-of-view;
    CalculateDetectionPR_obj(detection, obj_attr);
end