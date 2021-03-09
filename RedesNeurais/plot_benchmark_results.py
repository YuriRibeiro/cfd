#%%
import yegconfigs
import pathlib
import pandas as pd
import os
from scipy.io import loadmat
import re
import matplotlib.pyplot as plt

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
    def _gather_speed_vs_resolution_data(det_paths_dict, cfg_df):
        df = _Benchmark_Utils._get_det_paths_with_resolutions(det_paths_dict)
        df = pd.concat((df, cfg_df.loc[df.index].model), axis=1) #concat model name

        # Gather speed from speed files.
        dics = {} #key = resolution, cotents[exp_abbreviated order]
        p = re.compile('/\d+\\.\d+ ms')
        for exp_abbr in df.index:
            temp = {}
            temp[exp_abbr] = {}
            for res in df.columns:
                if res == 'model': continue
                path = df.loc[exp_abbr, res]
                with open(path / 'detection_speed.txt', 'r') as f:
                    l = f.readline()
                speed = float(p.search(l).group()[1:-3])
                temp[exp_abbr].update({'model': df.loc[exp_abbr]['model'], res : speed})
            dics.update(temp)
        
        #model-resolution-speed-dataframe
        mrs_df = pd.DataFrame.from_dict(dics, orient='index')
        mrs_df.sort_values(by=['model'], inplace=True)

        df_mean = []
        df_std = []
        for model in df.model.unique():
            df_mean.append(mrs_df.loc[mrs_df.model == model].mean())
            df_std.append(mrs_df.loc[mrs_df.model == model].std())
        df_mean = pd.concat(df_mean, axis=1)
        df_mean.columns = df.model.unique()
        df_std = pd.concat(df_std, axis=1)
        df_std.columns = df.model.unique()
        """df_mean:
                    yv5_S      yv5_M      yv5_L       yv5_X
        256x256     1.966667   3.166667   7.166667   12.233333
        384x384     3.000000   6.666667   6.866667   11.566667
        (....)
        """
        return mrs_df, df_mean, df_std
    
    @staticmethod
    def _plot_resolution_vs_speed(det_paths_dict, cfg_df, color_cycler):
        mrs_df, df_mean, df_std = _Benchmark_Utils._gather_speed_vs_resolution_data(det_paths_dict, cfg_df)
        models = cfg_df.model.unique()

        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(10,5))
            x = [int(r.split('x')[-1]) for r in df_mean.index]
            for model in models:
                y = df_mean[model]
                err = df_std[model]
                c = color_cycler[model]
                ax.plot(x,y,'o-', color=c, label=model)
                ax.errorbar(x,y,err, color=c, capsize=5)
            
            ax.set_xticks(x)
            ax.set_xlabel('Resolução (pixel)')
            ax.set_ylabel('Tempo (ms/ img)')
            ax.set_title('Resolução da Imagem VS Tempo Médio para Detecção')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(loc='upper left')
            
    @staticmethod
    def _plot_ap_vs_speed(det_paths_dict, cfg_df):
        mrs_df, df_mean, df_std = _Benchmark_Utils._gather_speed_vs_resolution_data(det_paths_dict, cfg_df)
        pass
        
    @staticmethod
    def _plot_resolution_vs_ap_overall(df): #boxplot
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
    
    @staticmethod
    def plot_resolution_vs_speed():
        #YV5_DET()._gather_speed_vs_resolution_data(YV5_DET.get_det_paths(), YV5_DET.get_df())
        #_plot_ap_vs_speed
        YV5_DET()._plot_resolution_vs_speed(YV5_DET.get_det_paths(), YV5_DET.get_df(),
                                            YV5_DET.color_cycler())


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
#a = YV5_DET().get_AP_overall_results()
#a.loc[ (a.model=='yv5_S') & (a.resolution == 256)].plot.scatter( )
#a.plot.scatter(x='resolution', y='AP_overall')


#YV5_DET().plot_resolution_vs_ap_overall()


#YV5_DET().plot_resolution_vs_speed()
# %%
