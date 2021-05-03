function [subFoldersNames_APoverall, subFoldersNames_seq, subFoldersNames_obj] = yuri_gen_list_of_detections_to_bench()
    %%   ESPECIFICAR DETECTORES PARA O BENCHMARK AP Overall, AP seq e AP obj.
    
    % Get a list of all folders in this folder.
    subFolders_paths = struct2cell(dir('./RES_DET'));
    subFolders_benchmarked = struct2cell(dir('./det_EVA'));
    subFolders_benchmarked = {subFolders_benchmarked{1,:}};
    [~,numSubFolders] = size(subFolders_paths);

    % Cell containing all subfolder names from subFolders_paths such that:
        %Don't add '.' and '..' to subFolders paths.
        %Don't add paths that has already been benchmarked, i.e. exists in
        %det_EVA folder.
    subFoldersNames_APoverall = {};
    idx = 1;
    for k=3:numSubFolders
        det_eva_foname=[subFolders_paths{1,k} '_overall.mat'];
        if ~any(strcmp(subFolders_benchmarked, det_eva_foname))
            subFoldersNames_APoverall{1,idx} = subFolders_paths{1,k};
            idx = idx + 1;
            fprintf('Adicionando %s para análise AP_overall.\n',subFolders_paths{1,k});
        end
    end
    if isempty(subFoldersNames_APoverall)
        fprintf('Nenhum experimento foi adicionado para o AP_overall benchmark.\n');
    end
    %%
    %   ESPECIFICAR DETECTORES PARA O BENCHMARK A NIVEL DE OBJETOS (CAR, TRUCK,
    %   BUS) E TIPOS DE SEQUÊNCIA (DAY, NIGHT, FOG, HIGH VIEW, LOW VIEW, ...)


    specificSubFoldersNames_baseline = {'det_SSD', 'det_RON', 'det_RFCN', ...
                                        'det_FRCNN'};
    yv5_bap_selected_models = {};
    yv3_bap_selected_models = {}; 
    % load best ap selected models
    yuri_yv5_best_ap_overall_models_selection;
    yuri_yv3_best_ap_overall_models_selection;
    specificSubFoldersNames_yuri = [specificSubFoldersNames_baseline ...
                                    yv3_bap_selected_models ...
                                    yv5_bap_selected_models];
    [~, numSpecificSubfolders] = size(specificSubFoldersNames_yuri);

    subFoldersNames_seq = {};
    subFoldersNames_obj = {};
    idx_seq = 1;
    idx_obj = 1;
    for k=1:numSpecificSubfolders
        seq_foname=[specificSubFoldersNames_yuri{1,k} '_seq.mat'];
        obj1_foname = [specificSubFoldersNames_yuri{1,k} '_obj_1.mat'];
        obj2_foname = [specificSubFoldersNames_yuri{1,k} '_obj_2.mat'];
        obj3_foname = [specificSubFoldersNames_yuri{1,k} '_obj_3.mat'];

        if ~any(strcmp(subFolders_benchmarked, seq_foname))
            subFoldersNames_seq{1,idx_seq} = specificSubFoldersNames_yuri{1,k};
            idx_seq = idx_seq + 1;
            fprintf('Adicionando %s para análise Seq.\n',specificSubFoldersNames_yuri{1,k});
        end

        if ~any(strcmp(subFolders_benchmarked, obj1_foname)) || ...
           ~any(strcmp(subFolders_benchmarked, obj2_foname)) || ...
           ~any(strcmp(subFolders_benchmarked, obj3_foname))
                subFoldersNames_obj{1,idx_obj} = specificSubFoldersNames_yuri{1,k};
                idx_obj = idx_obj +1;
                fprintf('Adicionando %s para análise Obj.\n',specificSubFoldersNames_yuri{1,k});
        end
    end

    if isempty(subFoldersNames_seq)
        fprintf('Nenhum experimento foi adicionado para o AP_seq benchmark.\n');
    end
    if isempty(subFoldersNames_obj)
        fprintf('Nenhum experimento foi adicionado para o AP_sobj benchmark.\n');
    end