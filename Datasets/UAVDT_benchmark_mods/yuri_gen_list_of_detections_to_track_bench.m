function [subFoldersNames_toTrack] = yuri_gen_list_of_detections_to_track_bench()
    %%   ESPECIFICAR DETECÇÕES PARA BENCHMARK.
    
    % Set tracker names and os.sep
    sep = filesep;
    
    % Get a list of all folders in this folder.
    subFolders_paths = struct2cell(dir('./RES_MOT'));
    [~,numSubFolders] = size(subFolders_paths);
    
    % Cell containing all subfolder names from subFolders_paths such that:
    subFoldersNames_toTrack = {};
    idx = 1;
    for j=3:numSubFolders
        subSubFolder = [subFolders_paths{2,j} sep subFolders_paths{1,j}];
        subSubFolders = struct2cell(dir(subSubFolder));
        [~,numSubSubFolders] = size(subSubFolders);
        detector = subFolders_paths{1,j};
        
        for k=3:numSubSubFolders    
            tracker = subSubFolders{1,k};
            resfolder = [subSubFolders{2,k} sep subSubFolders{1,k}];
            track_results_file = [resfolder sep 'eval.txt'];
            if isfolder(resfolder) && ~isfile(track_results_file)
                subFoldersNames_toTrack{1,idx} = detector;
                subFoldersNames_toTrack{2,idx} = tracker;
                idx = idx + 1;
                fprintf('Adicionando %s/%s para análise de tracking.\n',detector, tracker);
            end
        end
    end
    if isempty(subFoldersNames_toTrack)
        fprintf('Nenhum experimento foi adicionado para o tracking benchmark.\n');
    end
