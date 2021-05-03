%% LIMPAR VARS LOCAIS E IMPORTAR LISTA DE DETECTORES
clear;
clc;
addpath(genpath('.'));
[subFoldersNames_APoverall, subFoldersNames_seq, subFoldersNames_obj] = ...
                                    yuri_gen_list_of_detections_to_bench();
%% AP OVERALL          
fprintf('[INFO] Início da Análise (overall).\n');
for k=1:numel(subFoldersNames_APoverall)
    det_name = subFoldersNames_APoverall{k};
    fprintf('Analisando (overall): %s...', det_name);
    CalculateDetectionPR_overall(det_name);
end
fprintf('[INFO] FIM da Análise (overall).\n');
%% SEQUÊNCIAS (DAY, NIGHT, FOG, HIGH VIEW, LOW VIEW, ETC.)
fprintf('[INFO] Início da Análise (seq).\n');
for k=1:numel(subFoldersNames_seq)
    det_name = subFoldersNames_seq{k};
    fprintf('Analisando (seq): %s...\n', det_name);
    CalculateDetectionPR_seq(det_name);
end
fprintf('[INFO] FIM da Análise (seq).\n');
%% SEQUÊNCIAS DE OBJETOS: (CAR, BUS, TRUCK)
% calculate the AP scores in object-level attribute
fprintf('[INFO] Início da Análise (obj).\n');
for k=1:numel(subFoldersNames_obj)
    det_name = subFoldersNames_obj{k};
    fprintf('Analisando (obj): %s...\n', det_name);
    for obj_attr = 1:3
        % 1 for Vehicle Category;
        % 2 for Vehicle Occlusion;
        % 3 for Out-of-view;
        CalculateDetectionPR_obj(det_name, obj_attr);
    end
end
fprintf('[INFO] FIM da Análise (obj).\n');
%% Limpar variáveis 'boilerplate'
clearvars det_name k obj_attr