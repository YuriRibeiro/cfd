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
    def __init__(self, auto_calc=True):
        self.baseline_models = ['det_RON', 'det_SSD', 'det_FRCNN', 'det_RFCN']
        self.baseline_speeds_fps = [11.11, 41.55, 2.75, 4.65] #gpu speeds..
        self.cut_seq_obj_classes = ['truck', 'bus', 'lagre-occ', 'medium-occ', 'small-occ', 'medium-out', 'small-out', 'long']
        self.dest_uavdt = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'RES_DET'
        self.deteva_uavdt_fopath = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'det_EVA'
        self.det_paths_with_resolutions = self._get_det_paths_with_resolutions()

        if auto_calc:
            self.ap_overall_data = self._gather_AP_overall_data()
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
                    #print(f'Aviso: pasta {str(dest_folder_name)} já existe no destino... Pulando...')
                    table_old_new_path[res].extend([path, 'N/A'])
                    continue
                table_old_new_path[res].extend([path, dest_folder_name])
                dest_folder_name.mkdir(parents=True, exist_ok=True)
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
        multi_idx = []
        for exp in det_paths.index:
            multi_idx.extend([(exp, 'old'),(exp, 'new',)]) 

        multi_idx = pd.MultiIndex.from_tuples(multi_idx, names=["Exp", "Path"])
        # Check all old paths: a.query("Path == 'old'")
        # Check all new paths: a.query("Path == 'new'")
        return pd.DataFrame(table_old_new_path, index=multi_idx)
    
    def _gather_AP_overall_data(self):
        deteva_uavdt_fopath = self.deteva_uavdt_fopath
        net = self.get_df().name[0][:6]
        deteva_apoverall_files = deteva_uavdt_fopath.glob(f'{net}_*_overall.mat')
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
    
    def _plot_resolution_vs_speed(self, save=False):
        mrs_df, df_mean, df_std = self.speed_vs_resolution_df
        color_cycler = self.color_cycler
        cfg_df = self.net_dataframe
        det_paths_dict = self.det_paths_dict
        models = cfg_df.model.unique()

        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
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
            Plot.set_xlabel(ax, 'Resolução (pixel)', fontsize=14)
            Plot.set_ylabel(ax, 'Tempo (ms/ img)', fontsize=14)
            Plot.set_title(ax, 'Tempo Médio para Detecção VS Resolução da Imagem')
            # ax.annotate(f'(30 FPS)',
            #             xy=(384,33),
            #             xytext=(0, 1),  # 3 points vertical offset
            #             textcoords="offset points",
            #             ha='center', va='bottom',
            #             fontsize = 16)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(loc='upper left', fontsize=14)
        
        if save:
            net_name = model[:3]
            path = Plot.make_output_file_path('res_vs_mean_speed_vs_ap', net=net_name, phase='test')
            fig.savefig(path)
                   
    def _future_plot_resolution_vs_ap_overall(self):
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

    def _table_resolution_vs_ap_overall(self, save=False):
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

        if save:
            net = model[:3]
            latex = dft.to_latex(index=True, float_format="%.2f")
            path = Plot.make_output_file_path('table_all_res_vs_ap_overall', net=net, ext='txt', phase='test')
            with open(path, 'w') as f:
                f.write(latex)
        return dft
    
    def _select_best_ap_overall_models(self, export_to_matlab=False):
        bapoall = self.table_apoall_vs_resolution
        apoall = self.ap_overall_data
        nmodels = len(self.get_models())
        nexps = len(bapoall)
        ntrainstep = nexps//nmodels
        best_ap_exp_names = []
        for i in range(0, nexps, nmodels):
            batchi = bapoall.iloc[i:i+nmodels]
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

    def _plot_ap_vs_speed_best_models(self, save=False):
        #mrs_df, df_mean, df_std = self.speed_vs_resolution_df
        bap_exp_names = self.select_best_ap_overall_models(export_to_matlab = False)
        df = self.ap_overall_data.loc[bap_exp_names][['model', 'resolution', 'AP_overall']]
        _,mean_speed_vs_res,_ = self._gather_speed_vs_resolution_data()
        models = [] #'yv5_S', ...
        model_resolution = [] #('model (resolution)')
        x = [] # x in (ms/img)
        y = [] # y in (AP
        for model, res, ap in df.values:
            res = f'{res}x{res}'
            models.append(model)
            model_resolution.append(f'{model} ({res})')
            x.append(mean_speed_vs_res.loc[res, model])
            y.append(ap)
        # PLOT
        net_name = self.get_df().name[0][:6]
        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize = (10,4), constrained_layout=True)
            for mi, mri, xi, yi in zip(models, model_resolution, x, y):
                c = self.color_cycler()[mi]
                ax.plot(xi, yi, '*', color=c, markersize = 18, label = mri)
            Plot.set_title(ax,f'Modelos Selecionados ({net_name}): AP VS Velocidade de Detecção')
            Plot.set_xlabel(ax, 'Velocidade (ms/ img)', fontsize=14)
            Plot.set_ylabel(ax, 'AP', fontsize = 14)
            ax.legend(loc='upper left',fontsize=12)
            for spine in ax.spines.values():
                spine.set_visible(False)

        if save:
            net_name = model[:3]
            path = Plot.make_output_file_path('best_models_speed_vs_ap', net=net_name, phase='test')
            fig.savefig(path)

    def _plot_pr_curve_best_models(self, save=False):
        deteva_uavdt_fopath = self.deteva_uavdt_fopath
        bestmodels = self._select_best_ap_overall_models(export_to_matlab=False)
        baseline_models = self.baseline_models
        baseline_speeds_fps = self.baseline_speeds_fps
        deteva_paths = [deteva_uavdt_fopath / f'{p}_overall.mat' for p in bestmodels + baseline_models]
        dic = {}
        _,mean_speed_vs_res,_ = self._gather_speed_vs_resolution_data()
        for file_path in deteva_paths:
            content = loadmat(file_path, squeeze_me=True)['attribute']
            name = content['name'].item()
            prec = content['prec'].item()
            rec = content['rec'].item()
            ap = content['AP'].item()
            # pick model name if it is not a baseline model
            if not name in baseline_models:
                this_net_name = name.split('-')[0]
                for exp_name_abbreviated in self.net_dataframe.index:
                    series_name = self.net_dataframe['name'].loc[exp_name_abbreviated]
                    if this_net_name == series_name: break
                res = name.split('-')[-1]
                model = self.net_dataframe.loc[exp_name_abbreviated]['model']
            else:
                model = name
                res = 'N/A'
                exp_name_abbreviated = name
            if name in baseline_models:
                speed = baseline_speeds_fps[baseline_models.index(name)]
            else:
                speed = 1000/(mean_speed_vs_res.loc[res, model])#fps
            dic[name] = [exp_name_abbreviated, model, res, ap, speed, prec, rec]
        # Plot
        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
            for k,(exp_name_wn, model, res, ap, fps, p, r) in dic.items(): #wn = with number
                x = r
                y = p
                if not model in baseline_models:
                    c = self.color_cycler()[model]
                else:
                    c = yegconfigs.BaselineModels.color_cycler()[model]
                ax.plot(x, y, color=c, label=f'{model} ({ap:.2f}%, {fps:.2f})')
                
            net_name = bestmodels[0][:6]
            Plot.set_title(ax,f'Precision VS Recall -- Modelos {net_name} e de Referência')
            Plot.set_xlabel(ax, 'Recall', fontsize=14)
            Plot.set_ylabel(ax, 'Precision', fontsize = 14)
            ax.legend(loc='lower left', fontsize=12, ncol=2)
            for spine in ax.spines.values():
                spine.set_visible(False)

        if save:
            net_name = self.get_models()[0][:3]
            path = Plot.make_output_file_path('best_models_precision_vs_recall', net=net_name, phase='test')
            fig.savefig(path)
        return 

    def _plot_barchart_seq_obj(self, save=False):
        deteva_uavdt_fopath = self.deteva_uavdt_fopath
        bestmodels = self._select_best_ap_overall_models(export_to_matlab=False)
        df = self.get_df()
        bestmodels_abn = [df[df.name == pt.split('-')[0]].index[0] for pt in bestmodels]
        baseline_models = self.baseline_models
        seq_paths = [deteva_uavdt_fopath / f'{p}_seq.mat' for p in bestmodels + baseline_models]
        obj1_paths = [deteva_uavdt_fopath / f'{p}_obj_1.mat' for p in bestmodels + baseline_models]
        obj2_paths = [deteva_uavdt_fopath / f'{p}_obj_2.mat' for p in bestmodels + baseline_models]
        obj3_paths = [deteva_uavdt_fopath / f'{p}_obj_3.mat' for p in bestmodels + baseline_models]
        cut = self.cut_seq_obj_classes
        dic = {}
        total_paths = [seq_paths, obj1_paths, obj2_paths, obj3_paths]
        for path in total_paths:
            for file_path, model_abb in zip(path, bestmodels_abn + baseline_models):
                content = loadmat(file_path, squeeze_me=True)['attribute']
                names = content['name'].item()
                aps = content['AP'].item()
                if not model_abb in baseline_models: model_abb = df.model.loc[model_abb]
                if not model_abb in dic: dic[model_abb] = {}
                dic[model_abb].update({k:v for k,v in zip(names, aps) if k not in cut})
        
        # BAR CHART PLOT
        legends = dic.keys()
        labels = dic[model_abb].keys()
        
        #heights
        values = [list(dic[di].values()) for di in legends]
        M = np.vstack(values)
        bar_heights = [M[idx, :] for idx in range(M.shape[0])]
                
        # Colors
        c = {**self.color_cycler(), **yegconfigs.BaselineModels.color_cycler()} 

        # Save path
        net_name = df.model.iloc[0][:3]
        path = Plot.make_output_file_path('bar_chart_seq_obj_bench', net=net_name,  phase='test')
        # Make
        self._barplot(M, bar_heights, c, labels, legends=legends, save_path=path,  save=save)
        
    def _barplot(self, M, bar_heights, c_cicler, labels, save_path, legends='', barWidth=0.3, step=3, save=False):
        # Calc bar positions:
        step = step
        if M.shape[1]*barWidth <= step: print("Warning! Barras Sobrepostas. (barWidth*(#barras_no_grupo) >= step).")
        r1 = np.arange(0, step*M.shape[1], step)
        bar_positions = [[x + idx*barWidth for x in r1] for idx in range(0, M.shape[0])]
        
        # Bar Plot
        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True)
            for barp, barh, exp_name in zip(bar_positions, bar_heights, legends):
                barh[np.where(barh < 0.5)] = 0.5 # Expand low values to be visible in the plot.
                ax.bar(barp, barh, width=barWidth, color=c_cicler[exp_name], edgecolor='white', label=exp_name)
                        
            # Add xticks on the middle of the group bars
            x_ticks_labels = [f'{name}' for name in labels]
            x_ticks_pos = [step*r + 0.5*len(legends)*barWidth  for r in range(len(barp))]
            
            ax.grid(False)
            ax.set_xticks(x_ticks_pos)
            Plot.set_ylabel(ax, 'AP', fontsize=14)
            Plot.set_title(ax, 'AP por Atributo das Sequências de Vídeo')
            Plot.set_xticklabels(ax, x_ticks_labels, rotation=60)
            Plot.set_yticklabels(ax,)
            fig.legend(loc='upper right', ncol=2, fontsize=14)
            #Plot.set_legends(fig, [ax], bbox_to_anchor=(1,1))

            for spine in ax.spines.values():
                spine.set_visible(False)

        # SaveFig
        if save: fig.savefig(save_path)

#AutoCalc Should be set off, if you want to link det files to uavdt
class Plot_YV5_DET_Bench(_Benchmark_Utils, yegconfigs.YV5_CONFIGS):
    def __init__(self, auto_calc=True):
        self.det_paths_dict = self.get_det_paths() #yegconfigs.YV5_CONFIGS
        self.net_dataframe = self.get_df()
        super(Plot_YV5_DET_Bench, self).__init__(auto_calc=auto_calc)
        if auto_calc:
            self.table_apoall_vs_resolution = self.table_resolution_vs_ap_overall()

    def transfer_to_uavdt_bench_folder(self, only_dataframe=False):
        return self._symlinks_det_folder_to_uavdt(only_dataframe=only_dataframe)
    
    def get_AP_overall_results(self):
        return self._gather_AP_overall_data()
    
    def future_plot_resolution_vs_ap_overall(self):
        return self.future_plot_resolution_vs_ap_overall()
    
    def table_resolution_vs_ap_overall(self, save=False):
        return self._table_resolution_vs_ap_overall(save)
    
    def plot_resolution_vs_speed(self, save=False):
        return self._plot_resolution_vs_speed(save)
    
    def select_best_ap_overall_models(self, export_to_matlab=True):
        return self._select_best_ap_overall_models(export_to_matlab)
    
    def plot_ap_vs_speed_best_models(self, save=False):
        return self._plot_ap_vs_speed_best_models(save)
    
    def plot_pr_curve_best_models(self, save=False):
        return self._plot_pr_curve_best_models(save)
    
    def plot_barchart_seq_obj(self, save):
        return self._plot_barchart_seq_obj(save)


class Plot_YV3_DET_Bench(_Benchmark_Utils, yegconfigs.YV3_CONFIGS):
    def __init__(self, auto_calc=True):
        self.det_paths_dict = self.get_det_paths() #yegconfigs.YV3_CONFIGS
        self.net_dataframe = self.get_df()
        super(Plot_YV3_DET_Bench, self).__init__(auto_calc=auto_calc)
        if auto_calc:
            self.table_apoall_vs_resolution = self.table_resolution_vs_ap_overall()

    def transfer_to_uavdt_bench_folder(self, only_dataframe=False):
        return self._symlinks_det_folder_to_uavdt(only_dataframe=only_dataframe)
    
    def get_AP_overall_results(self):
        return self._gather_AP_overall_data()
    
    def future_plot_resolution_vs_ap_overall(self):
        return self.future_plot_resolution_vs_ap_overall()
    
    def table_resolution_vs_ap_overall(self, save=False):
        return self._table_resolution_vs_ap_overall(save)
    
    def plot_resolution_vs_speed(self, save=False):
        return self._plot_resolution_vs_speed(save)
    
    def select_best_ap_overall_models(self, export_to_matlab=True):
        return self._select_best_ap_overall_models(export_to_matlab)
    
    def plot_ap_vs_speed_best_models(self, save=False):
        return self._plot_ap_vs_speed_best_models(save)
    
    def plot_pr_curve_best_models(self, save=False):
        return self._plot_pr_curve_best_models(save)
    
    def plot_barchart_seq_obj(self, save):
        return self._plot_barchart_seq_obj(save)

# Plot Class from yegconfigs
Plot = yegconfigs.Plot


if __name__ == '__main__':

    # Init
    SAVE = False
    auto_calc=True #False only for the first run

    a = Plot_YV5_DET_Bench(auto_calc=auto_calc)
    #a = Plot_YV3_DET_Bench(auto_calc=auto_calc)
    
    ## Symlink det files to uavdt benchmark RES_DET folder and a df with orig/dest files
    df_1 = a.transfer_to_uavdt_bench_folder(only_dataframe=False)
    print('Paths transferidos para o det folder: \n', df_1.query("Path == 'new'"))
        # Check all old paths: a.query("Path == 'old'")
        # Check all new paths: a.query("Path == 'new'")

    ##Gather all AP results for overall detections
    df_2 = a.get_AP_overall_results()

    ## (Future) Plot resolution vs ap_overall
    #a.future_plot_resolution_vs_ap_overall()

    ## Selectbest ap exp for each model
    a.select_best_ap_overall_models(export_to_matlab=True)

    ## Make Latex Table resolution vs ap_overall
    a.table_resolution_vs_ap_overall(save = SAVE)

    ## Plot mean resolutions vs speed
    a.plot_resolution_vs_speed(save = SAVE)

    # Plot AP vs Speed for the best models()
    a.plot_ap_vs_speed_best_models(save= SAVE)

    ## Plot PR Curve for the best and baseline models
    a.plot_pr_curve_best_models(save= SAVE)

    ## Plot seq and obj det scores:
    a.plot_barchart_seq_obj(save= SAVE)

    pass
# %%
