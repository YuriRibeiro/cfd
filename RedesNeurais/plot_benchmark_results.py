#%%
import yegconfigs
import pathlib
import pandas as pd
import os
from scipy.io import loadmat

root = yegconfigs.root

class _Benchmark_Utils():
    """
    Esta classe implementa métodos para mover os arquivos YOLOv*_UAVDT_*/YOLO_UAVDT_Det/{exp_name}/000x000
    para a pasta correta do benchmark do uavdt dataset.
    '{root}/Datasets/UAVDT/UAV-benchmark-MOTD_v1.0/RES_DET/'
    """  
    def __init__(self): pass

    @staticmethod
    def baseline_models():
        return ['det_RON', 'det_SSD', 'det_FRCNN', 'det_RFCN']

    @staticmethod
    def _get_det_paths_with_resolutions(det_paths_dict):
        """
        return a pd.DataFrame: rows: exp_name_abbreviated (yv5_0, yv5_1, ...);
                               collumns: resolutions (256x256, 384x385, ...);
                               contents: paths to the detections folder.
        """
        resolutions = [f'{q}x{q}' for q in range(256, 1536+128, 128)]
        dic = {}
        for name, path in det_paths_dict.items():
            det_txt_folder_paths = [x for x in path.glob('[0-9]*x[0-9]*') if x.is_dir()]
            if det_txt_folder_paths == []: print(f'Name: {name}. Path: {path}. Detecções não encontradas.')
            list_ = [None]*len(resolutions)
            for folderpath in det_txt_folder_paths:
                res = folderpath.stem
                if not res in resolutions: print("Problemas na resolução.{res} não encontrada.")
                list_[resolutions.index(res)] = folderpath
            dic[name] = list_ 
        df = pd.DataFrame(dic, index=resolutions).T
        return df
    
    @staticmethod
    def _symlinks_det_folder_to_uavdt(det_paths_dict, only_dataframe = False):
        dest_uavdt = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'RES_DET'
        det_paths = _Benchmark_Utils._get_det_paths_with_resolutions(det_paths_dict)
        index = det_paths.index
        table_old_new_path = {k:[] for k in det_paths.columns}
        fake_det = '1,-1,1,1,1,1,0.1,1,-1'
        for res, paths_at_res in det_paths.items():
            for path in paths_at_res:
                suffix = path.parent.stem
                assert res == path.stem, 'Error: det_folder_resolution != pd.DataFrame resolution.'
                new_folder_name = f'{suffix}-{res}'
                dest_folder_name = dest_uavdt / new_folder_name
                if dest_folder_name.exists() and not only_dataframe:
                    print(f'Aviso: pasta {str(dest_folder_name)} já existe no destino... Pulando...')
                    continue
                dest_folder_name.mkdir(parents=True, exist_ok=True)
                table_old_new_path[res].append({'old':path,'new':dest_folder_name})
                if not only_dataframe:
                    det_files = path / f'det_{path.parent.stem}'
                    det_files = det_files.glob('*.txt')
                    for file in det_files:
                        dest = dest_folder_name / f'{str(file.stem)}.txt'
                        if os.path.getsize(file) == 0:
                            # If it is an empty file, generate a line with
                            # a small 'detection' just to don't cause bug in the benchmark suite.
                            with open(dest, 'w') as f:
                                f.write(fake_det)
                        else:
                            os.symlink(file, dest)
        return pd.DataFrame(table_old_new_path, index=det_paths.index)
    
    @staticmethod
    def _gather_AP_overall_data(net_dataframe):
        deteva_uavdt_fopath = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'det_EVA'
        deteva_apoverall_files = deteva_uavdt_fopath.glob('*_overall.mat')
        dic = {}
        baseline_models = _Benchmark_Utils.baseline_models()
        for file_path in deteva_apoverall_files:
            content = loadmat(file_path, squeeze_me=True)['attribute']
            name = content['name'].item()
            if name in baseline_models: continue
            #prec = content['prec'].item()
            #rec = content['rec'].item()
            ap = content['AP'].item()
            # pick model name
            this_net_name = name.split('-')[0]
            for exp_name_abbreviated in net_dataframe.index:
                series_name = net_dataframe['name'].loc[exp_name_abbreviated]
                if this_net_name == series_name: break
            exp_number = int(exp_name_abbreviated[4:])
            resolution = int(name.split('-')[-1].split('x')[-1])
            net_model = net_dataframe.loc[exp_name_abbreviated]['model']
            dic[name] = [exp_name_abbreviated, exp_number, net_model, resolution, ap]
        return pd.DataFrame.from_dict(dic, columns = ['exp_name', 'exp_number', 'model', 'resolution', 'AP_overall'], orient='index')

    @staticmethod
    def _plot_resolution_vs_ap_overall(df):
        models = df.model.unique()
        resolutions = df.resolution.unique()
        ap = {}
        for model in models:
            res_data = {}
            for resolution in resolutions:
                data = df[(df.model == model) & (df.resolution == resolution)]['AP_overall']
                res_data[resolution] = data
            ap[model] = res_data
        pass


class Plot:
    def __init__(self): pass


     
class YV5_DET(yegconfigs.YV5_CONFIGS, _Benchmark_Utils):
    def __init__(self):
        pass

    @staticmethod
    def transfer_to_uavdt_bench_folder(only_dataframe=False):
        return YV5_DET._symlinks_det_folder_to_uavdt(YV5_DET.get_det_paths(), only_dataframe=only_dataframe)
    
    @staticmethod
    def get_AP_overall_results():
        return YV5_DET._gather_AP_overall_data(YV5_DET.get_df())
    
    @staticmethod
    def plot_resolution_vs_ap_overall():
        YV5_DET()._plot_resolution_vs_ap_overall(YV5_DET.get_AP_overall_results())



class YV3_DET(yegconfigs.YV3_CONFIGS, _Benchmark_Utils):
    def __init__(self):
        pass

    @staticmethod
    def transfer_to_uavdt_bench_folder(only_dataframe=False):
        return YV3_DET._symlinks_det_folder_to_uavdt(YV3_DET.get_det_paths(), only_dataframe=only_dataframe)
    
    @staticmethod
    def get_AP_overall_results():
        return YV3_DET._gather_AP_overall_data(YV3_DET.get_df())
        



# Symlink det files to uavdt benchmark RES_DET folder
#YV5_DET().transfer_to_uavdt_bench_folder(only_dataframe=True)

# Gather all AP for overall detections
a = YV5_DET().get_AP_overall_results()
#a.loc[ (a.model=='yv5_S') & (a.resolution == 256)].plot.scatter( )
#a.plot.scatter(x='resolution', y='AP_overall')

#YV5_DET().plot_resolution_vs_ap_overall()

# %%
