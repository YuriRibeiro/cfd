#%%
import yegconfigs
import pathlib
import pandas as pd
import os
from scipy.io import loadmat
import re
import matplotlib.pyplot as plt
import numpy as np

root = yegconfigs.root


class _Benchmark_Utils():  
    def __init__(self):
        self.baseline_models = ['det_RON', 'det_SSD', 'det_FRCNN', 'det_RFCN']
        self.dest_uavdt = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'RES_DET'
        self.deteva_uavdt_fopath = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'det_EVA'

        self.ap_overall_data = self._gather_AP_overall_data()
        self.det_paths_with_resolutions = self._get_det_paths_with_resolutions()
        self.speed_vs_resolution_df = self._gather_speed_vs_resolution_data()
        
    def _get_det_paths_with_resolutions(self):
        """
        return a pd.DataFrame: rows: exp_name_abbreviated (yv5_0, yv5_1, ...);
                               collumns: resolutions (256x256, 384x385, ...);
                               contents: paths to the detections folder.
        """
        resolutions = [f'{q}x{q}' for q in range(256, 1536+128, 128)]
        dic = {}
        det_paths_dict = self.det_paths_dict
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
    
    def _symlinks_det_folder_to_uavdt(self, only_dataframe = False):
        dest_uavdt = self.dest_uavdt
        det_paths = self.det_paths_with_resolutions
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
    
    def _gather_AP_overall_data(self):
        deteva_uavdt_fopath = self.deteva_uavdt_fopath
        deteva_apoverall_files = deteva_uavdt_fopath.glob('*_overall.mat')
        dic = {}
        baseline_models = self.baseline_models
        for file_path in deteva_apoverall_files:
            content = loadmat(file_path, squeeze_me=True)['attribute']
            name = content['name'].item()
            if name in baseline_models: continue
            #prec = content['prec'].item()
            #rec = content['rec'].item()
            ap = content['AP'].item()
            # pick model name
            this_net_name = name.split('-')[0]
            for exp_name_abbreviated in self.net_dataframe.index:
                series_name = self.net_dataframe['name'].loc[exp_name_abbreviated]
                if this_net_name == series_name: break
            exp_number = int(exp_name_abbreviated[4:])
            resolution = int(name.split('-')[-1].split('x')[-1])
            net_model = self.net_dataframe.loc[exp_name_abbreviated]['model']
            dic[name] = [exp_name_abbreviated, exp_number, net_model, resolution, ap]
        return pd.DataFrame.from_dict(dic, columns = ['exp_name', 'exp_number', 'model', 'resolution', 'AP_overall'], orient='index')
    
    def _gather_speed_vs_resolution_data(self):
        df = self.det_paths_with_resolutions
        cfg_df = self.net_dataframe
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
    
    def _plot_resolution_vs_speed(self):
        mrs_df, df_mean, df_std = self.speed_vs_resolution_df
        color_cycler = self.color_cycler
        cfg_df = self.net_dataframe
        det_paths_dict = self.det_paths_dict
        models = cfg_df.model.unique()

        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(10,5))
            x = [int(r.split('x')[-1]) for r in df_mean.index]
            for model in models:
                y = df_mean[model]
                err = df_std[model]
                c = color_cycler()[model]
                ax.plot(x,y,'o-', color=c, label=model, linewidth=0.1, markersize=5)
                ax.errorbar(x,y,err, color=c, capsize=5)
            
            # ax.hlines([33.33], 256, 1536, linestyle='dashed', color='black', linewidth=1)
            # ax.set_yticks(np.concatenate([ax.get_yticks(), [33.3]]))
            ax.set_ylim([0, ax.get_yticks().max()])
            ax.set_xticks(x)
            ax.set_xlabel('Resolução (pixel)')
            ax.set_ylabel('Tempo (ms/ img)')
            ax.set_title('Resolução da Imagem VS Tempo Médio para Detecção')
            # ax.annotate(f'(30 FPS)',
            #             xy=(384,33),
            #             xytext=(0, 1),  # 3 points vertical offset
            #             textcoords="offset points",
            #             ha='center', va='bottom',
            #             fontsize = 16)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(loc='upper left')
            
    def _plot_ap_vs_speed(self):
        #mrs_df, df_mean, df_std = self.speed_vs_resolution_df
        pass
        
    def _plot_resolution_vs_ap_overall(self):
        df = self.ap_overall_data
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

    def _table_resolution_vs_ap_overall(self):
        df = self.ap_overall_data
        ordered_list_of_models = self.get_models()
        models = np.array(ordered_list_of_models)
        df_order = []
        qtt_of_models = len(models)
        dft = df.copy()
        for idx in dft.index:
            exp_number = dft.at[idx, 'exp_number']
            train_step = int(exp_number) // qtt_of_models
            dft.at[idx, 'model'] += f' ({ train_step })'
        # Pivot:        
        dft = dft.pivot(index='model', columns='resolution', values='AP_overall')
        # Reorder
        new_index = []
        for idx in dft.index:
            model, trainstep = idx.split(' ')
            mult = np.where(model == models)[0][0]*len(models)
            soma = int(trainstep[1:-1])
            new_index.append((idx, mult+soma)) #idx, new_pos
        new_index.sort(key=lambda x: x[1])
        new_index = [x[0] for x in new_index]
        dft = dft.reindex(new_index)
        # Add max_resolution to identify row-wise maximum
        max_cols = []
        for idx in range(len(dft.index)):
            max_ap = dft.max(axis=1)[idx]
            col = dft.columns[dft.iloc[idx] == max_ap][0]
            max_cols.append(f'{max_ap} ({col})')
        max_cols = pd.DataFrame(max_cols, index=dft.index, columns=['Max_AP_(Resolution)'])
        dft = pd.concat([dft, max_cols], axis=1)
        return dft
    
    def _select_best_ap_overall_models(self, export_to_matlab):
        bapoall = self.table_apoall_vs_resolution
        apoall = self.ap_overall_data
        nmodels = len(self.get_models())
        nexps = len(self.get_df())
        ntrainstep = nexps//nmodels
        best_ap_exp_names = []
        for i in range(0, nexps, ntrainstep):
            batchi = bapoall.iloc[i:i+ntrainstep]
            ap_max = 0
            idx_max = 0
            exp_num_max = 0
            res_max =0
            for idx, val in batchi['Max_AP_(Resolution)'].items():
                ap, res = val.split(' ')
                ap, res = float(ap), int(res[1:-1])
                if ap >= ap_max:
                    idx_max = idx
                    ap_max = ap
                    res_max = res
            model, trainstep = idx_max.split(' ')
            trainstep=int(trainstep[1:-1])
            exp_num_max = trainstep * nmodels + self.get_models().index(model)
            best_ap_exp_names.append(apoall[(apoall.exp_number == exp_num_max) & (apoall.resolution == res_max)].index[0])
        
        net_name = model[:3]
        if export_to_matlab:
            dest = self.dest_uavdt.parent / f'yuri_{net_name}_best_ap_overall_models_selection.m'

            best_ap_exp_names_str = ", ".join(map(repr,best_ap_exp_names))
            with open(dest, 'w') as f:
                f.write(f'{net_name}_bap_selected_models = {{{best_ap_exp_names_str}}};')

        return best_ap_exp_names



class Plot_YV5_DET_Bench(_Benchmark_Utils, yegconfigs.YV5_CONFIGS):
    def __init__(self):
        self.det_paths_dict = self.get_det_paths() #yegconfigs.YV5_CONFIGS
        self.net_dataframe = self.get_df()
        super(Plot_YV5_DET_Bench, self).__init__()
        self.table_apoall_vs_resolution = self._table_resolution_vs_ap_overall()
        

    def transfer_to_uavdt_bench_folder(self, only_dataframe=False):
        return self._symlinks_det_folder_to_uavdt(only_dataframe=only_dataframe)
    
    def get_AP_overall_results(self):
        return self._gather_AP_overall_data()
    
    def plot_resolution_vs_ap_overall(self):
        return self._plot_resolution_vs_ap_overall()
    
    def table_resolution_vs_ap_overall(self):
        return self.table_apoall_vs_resolution
    
    def plot_resolution_vs_speed(self):
        return self._plot_resolution_vs_speed()
    
    def select_best_ap_overall_models(self, export_to_matlab=True):
        return self._select_best_ap_overall_models(export_to_matlab)

# Plot class
Plot = yegconfigs.Plot

# Init
a = Plot_YV5_DET_Bench()
a.select_best_ap_overall_models()

## Symlink det files to uavdt benchmark RES_DET folder and a df with orig/dest files
#df_1 = a.transfer_to_uavdt_bench_folder(only_dataframe=True)

#Gather all AP results for overall detections
##df_2 = a.get_AP_overall_results()

## Plot mean resolutions vs speed
#a.plot_resolution_vs_speed()

## Plot resolution vs ap_overall
#a.plot_resolution_vs_ap_overall()

## Make Latex Table resolution vs ap_overall
#a.table_resolution_vs_ap_overall()

pass
# %%
