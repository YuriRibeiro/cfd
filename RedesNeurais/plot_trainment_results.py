#%%
from tensorboard.backend.event_processing import event_accumulator
import os, pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import yegconfigs


# DO NOT PRINT STATUS?
SILENT = False

## Override:
if yegconfigs.SILENT == True: SILENT = True

class YV5_Config(yegconfigs.YV5_CONFIGS):
    def __init__(self): pass

    @staticmethod
    def get_nrows_plot_catloss():
        return 4
    
    @staticmethod
    def get_net_name():
        return 'yv5'

    @staticmethod
    def load_tb_paths():
        return YV5_Config.get_tb_paths()


class YV3_Config(yegconfigs.YV3_CONFIGS):    
    def __init__(self): pass

    @staticmethod
    def get_nrows_plot_catloss():
        return 3

    @staticmethod
    def get_net_name():
        return 'yv3'
        
    @staticmethod
    def load_tb_paths():
        return YV3_Config.get_tb_paths()


class Plot_Train_Data():
    def __init__(self):
        size_guidance = self.load_tb_size_guidance()
        self.tb_paths = self.load_tb_paths()
        
        #Experiment names
        self.exp = self.tb_paths.keys()
        # Individual dataframes for each one experiment
        self.individual_exp_df = {}
        # Create individual exp dataframes
        for k,v in self.tb_paths.items():
            self.ea = event_accumulator.EventAccumulator(str(v),
                                                         size_guidance=size_guidance)
            self.ea.Reload()
            self.scalar_tags = self.ea.Tags()['scalars']
            self.individual_exp_df[k] = self._to_dataframe()
        # Concatenated dataframes for exp sequences
        self._cat_df()
        # Calc best fitness points
        self.best_fitness_points = self._get_best_fitness_points()


    def _to_dataframe(self):
        data = {}
        headers = []
        steps = []
        first_run = True
        for tag in self.scalar_tags:
            headers.append(tag)
            check_step = []
            for idx, tgv in enumerate(self.ea.Scalars(tag)):
                walltime, step, value = tgv.wall_time, tgv.step, tgv.value
                if idx in data:
                    data[idx].append(value)
                else:
                    data[idx] = [value]
                if first_run:
                    steps.append(step)
                check_step.append(step)
            first_run = False
            if check_step != steps:
                raise Exception("Steps diferentes nos dados fornecidos.")
        
        s = pd.DataFrame(steps, columns=['steps'])
        df = pd.DataFrame(list(data.values()), columns = headers)
        df = pd.concat([df,s], axis=1)

        # calcular o fitness score : fi = 0.9*map05:0.95 + 0.1*map0.5
        df['fitness'] = df['metrics/mAP_0.5']*0.1 + df['metrics/mAP_0.5:0.95']*0.9
        pass

        return df
    
    def _cat_df(self):
        cats = self.load_cats()
        self.cats = {}
        for nets in cats:
            new = []
            cat = None
            #if nets[0] == None: # YOLOv5 301 and 302
            #    self.cats[nets[1]] = self.individual_exp_df[nets[1]].copy()
            #    continue
            cdf_0 = self.individual_exp_df[nets[0]]
            exp_col = pd.DataFrame({'exp' : [nets[0]]*len(cdf_0.index)})
            cdf_0 = pd.concat([cdf_0, exp_col], axis = 1)
            for i in range(1, len(nets)):
                net = nets[i]
                cdf_1 = self.individual_exp_df[net].copy()
                exp_col = pd.DataFrame({'exp' : [net]*len(cdf_1.index)})
                cdf_1 = pd.concat([cdf_1, exp_col], axis = 1) #place an exp identifier
                cdf_1.index += 1 # 0-indexed to 1-indexed
                cdf_1.index += cdf_0.index.max() # sum indexes
                cat = pd.concat([cdf_0, cdf_1], axis=0) #concat linewise
                cdf_0 = cat
            self.cats[nets[-1]] = cat.copy() #store a copy of the concat using the last name.
    

    def _get_best_fitness_points(self):
        """
        Return: Dict whose keys are the exp_names of cat_exps (self.get_cat_exps_labels())
                and the values are a list cotaining the best fitness point for each exp_name.
        """
        vlines = [0] + self.get_vline_pos() #index
        vlines = vlines + [vlines[-1] + 50]
        exps = self.get_cat_exps_labels()
        best_points = {}

        for exp in exps:
            best_points[exp] = []
            fitness = self.cats[exp]['fitness']
            for idx in range(len(vlines)-1):
                best_x = fitness.iloc[ vlines[idx] : vlines[idx+1]].argmax()
                best_points[exp].append(best_x+vlines[idx])
        return best_points

    def get_data_labels(self):
        return self.ea.Tags()['scalars']
    
    def get_individual_exps_labels(self):
        return self.individual_exp_df.keys()
    
    def get_cat_exps_labels(self):
        return self.cats.keys()

    def get_dataframe_individual_exps(self):
        """
        Dict containing a dataframe for each individual exp.
        """
        return self.individual_exp_df

    def get_dataframe_cat_exps(self):
        """
        Dict containing a dataframe for each concatenated experiment.
        """
        return self.cats
        
    def plot_cat_metrics(self, save = False):
        metrics = self.plot_fields()['metrics']
        experiments = self.get_cat_exps_labels()
        if not SILENT: print(f"[INFO] Plotando {self.get_net_name()} métricas AP, mAP, ...")

        for metric in metrics:
            with plt.style.context('bmh'):
                fig, ax = plt.subplots(3,2, figsize = Plot.properties('figsize_big'),
                                         constrained_layout=True)
                gs = ax[2, 0].get_gridspec()
                for axe in ax[2, :]: axe.remove()
                axbig = fig.add_subplot(gs[2, :])
                ax = ax.flatten()

            for idx, exp in enumerate(experiments):
                filtered_df = self.cats[exp][metrics]
                steps = self.cats[exp]['steps']
                c = self.color_cycler()[exp]
                x = filtered_df.index + Plot.offset_x()
                y = filtered_df[metric]

                # Plot best fitness points
                for best_x in self.best_fitness_points[exp]:
                    best_y = y.iloc[best_x]
                    Plot.plot_best_fitness_points(ax[idx], best_x, best_y)

                # Plot vertical lines
                Plot.plot_vlines(self.get_vline_pos(), ax[idx])
                if idx == 0:
                    Plot.plot_vlines(self.get_vline_pos(), axbig)

                # Plot x,y data from dataframe
                Plot.plot(x,y,fig,ax[idx],c, self.plot_exp_title(exp))
                Plot.plot(x,y,fig,axbig,c)

            #Remove axe[3] if the net is equal to yolov3
            if self.get_net_name() == 'yv3':
                fig.delaxes(ax[3])

            # Adjust ticks and labels in the small frame
            valid_axes = [*ax[:4], axbig]
            Plot.adjust_xticks(valid_axes, steps, self.get_vline_pos(), x)

            # Place Figure Title
            Plot.set_fig_suptitle(fig, self.plot_field_title(metric))
            
            # Place axbig Title
            Plot.set_xlabel(axbig, 'Épocas')
            
            # Place legends
            Plot.set_legends(fig, fig.axes, bbox_to_anchor=(1, 0.145))

            # Place indexes            
            for idx, axe in enumerate(valid_axes):
                Plot.set_title(axe, f'({idx+1})')

            output_file_path = Plot.make_output_file_path(metric, self.plot_field_title(metric), net=self.get_net_name())
            if save: plt.savefig(output_file_path)

    def _plot_cat_loss(self, save = False, plot_metric = 'train_loss'):
        metrics = self.plot_fields()[plot_metric]
        experiments = self.get_cat_exps_labels()
        
        const = 0
        final_title = ''
        for metric in metrics:
            final_title += f', {self.plot_field_title(metric)}'
        final_title = final_title[1:]
        tipo = 'Train:' if plot_metric == 'train_loss' else 'Test:'
        final_title = f'{tipo} {final_title}'

        with plt.style.context('bmh'):
                fig, ax = plt.subplots(self.get_nrows_plot_catloss(),3, figsize = Plot.properties('figsize_big'),
                                      constrained_layout=True, sharex=True)
                ax = ax.flatten()
                fig2, ax2 = plt.subplots(1,3, figsize=Plot.properties('figsize_medium'),
                                        constrained_layout=True, sharex=True)
                ax2 = ax2.flatten()
        
        for metric in metrics:
            idx = 0 + const
            for exp in experiments:
                filtered_df = self.cats[exp][metrics]
                steps = self.cats[exp]['steps']
                c = self.color_cycler()[exp]
                x = filtered_df.index + Plot.offset_x()
                y = filtered_df[metric]

                # Plot best fitness points
                for best_x in self.best_fitness_points[exp]:
                    best_y = y.iloc[best_x]
                    Plot.plot_best_fitness_points(ax[idx], best_x, best_y)
                
                # Plot vertical lines and add ticks to these lines
                Plot.plot_vlines(self.get_vline_pos(), ax[idx])
                if idx == const:
                    Plot.plot_vlines(self.get_vline_pos(), ax2[const])
                    Plot.set_title(ax2[idx], f'({idx+1})')
                                        
                # Plot x,y dataframe points
                Plot.plot(x, y, fig, ax[idx], c, self.plot_exp_title(exp))
                Plot.plot(x, y, fig, ax2[const], c, self.plot_exp_title(exp))

                idx += 3
                if idx >= 12: break
            const += 1   

        # Adjust xticks
        Plot.adjust_xticks([*ax, *ax2], steps, self.get_vline_pos(), x)

        # Figure Sup Title:
        Plot.set_fig_suptitle(fig, final_title)
        Plot.set_fig_suptitle(fig2, final_title)
        
        # Place Indexes for each axe:
        for idx in range(len(ax)):
            Plot.set_title(ax[idx], f'({idx+1})')
        
        # Lower Axe Label:
        Plot.set_xlabel(ax[-2], 'Épocas')
        Plot.set_xlabel(ax2[-2], 'Épocas')

        # Legends:
        Plot.set_legends(fig, fig.axes[::3], bbox_to_anchor=(1.02, 2.1))
        
        # Save Figure:
        
        if save: 
            output_file_path = Plot.make_output_file_path(plot_metric,"", net=self.get_net_name())
            fig.savefig(output_file_path)
            output_file_path = Plot.make_output_file_path(plot_metric+"_resumed","", net=self.get_net_name())
            fig2.savefig(output_file_path)
        
    def plot_cat_trainloss(self, save=False):
        if not SILENT: print(f"[INFO] Plotando {self.get_net_name()} train loss ...")
        self._plot_cat_loss(save=save, plot_metric = 'train_loss')
    
    def plot_cat_testloss(self, save=False):
        if not SILENT: print(f"[INFO] Plotando {self.get_net_name()} test loss ...")
        self._plot_cat_loss(save=save, plot_metric = 'test_loss')

    def plot_cat_learning_rates(self, save = False):
        metrics = self.plot_fields()['lr']
        figsize = Plot.properties('figsize_big')
        if not SILENT: print(f"[INFO] Plotando {self.get_net_name()} learning rates ...")
        
        final_title = ''
        for metric in metrics:
            final_title += f', {self.plot_field_title(metric)}'
        final_title = final_title[1:]

        with plt.style.context('bmh'):
                fig, ax = plt.subplots(3,1, figsize = figsize,
                                       constrained_layout=True, sharex=True)
                ax = ax.flatten()
                exp = list(self.get_cat_exps_labels())[-1]
                filtered_df = self.cats[exp][metrics]

                for idx, metric in enumerate(metrics):
                    steps = self.cats[exp]['steps']
                    c = 'm'
                    x = filtered_df.index + Plot.offset_x()
                    y = filtered_df[metric]
                    # Plot vertical lines
                    Plot.plot_vlines(self.get_vline_pos(), ax[idx])
                    # Plot x,y dataframe points
                    Plot.plot(x, y, fig, ax[idx], c)
                    # Set title
                    Plot.set_title(ax[idx], f'({idx+1})')
                
        # Fig superior title 
        Plot.set_fig_suptitle(fig, final_title)
        # Set xlabel
        Plot.set_xlabel(ax[idx], 'Épocas')
        # Change ticks from index to steps and format
        Plot.adjust_xticks(ax, steps, self.get_vline_pos(), x)
        output_file_path = Plot.make_output_file_path(metric, self.plot_field_title(metric), net=self.get_net_name())
        if save: plt.savefig(output_file_path)

class Plot_Yv3_Train_Data(Plot_Train_Data, YV3_Config):
    def __init__(self):
        super(Plot_Yv3_Train_Data, self).__init__()

class Plot_Yv5_Train_Data(Plot_Train_Data, YV5_Config):
    def __init__(self):
        super(Plot_Yv5_Train_Data, self).__init__()
        
class Plot:
    def __init__(self): pass

    @staticmethod
    def offset_x():
        offset = 1
        return offset

    @staticmethod
    def plot(x,y,fig=None, ax=None, color=None, leg=None):
        l, = ax.plot(x, y, linewidth=1.7, color=color, label=leg)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax, l
    
    @staticmethod
    def plot_vlines(vlines_pos :'list', ax):
        for pos in vlines_pos:
            ax.axvline(x=pos, color='black', linestyle='--',
                      linewidth = 0.8,alpha=0.8)                 
    @staticmethod
    def properties(name : 'str'):
        d = {'tick_params' : {'axis':'both', 'which':'major', 'labelsize':15},
            'figsize_big' : (16, 14),
            'figsize_medium' : (16, 7) }
        return d[name]
    
    @staticmethod
    def make_output_file_path(metric, metric_spec, net="Error", phase='train'):
        """
        metric: train_loss, test_loss, learning_rate, ...
        metric_spec: box loss, object. loss, ...
        """
        root = pathlib.Path(__file__).parent
        output_dir = root /'plots'/ net/ phase
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.datetime.now()
        time_stamp = start_time.strftime("%d_%B_%Y_%Hh_%Mm")
        metric = metric.replace("/", "_")
        return output_dir / f'{net}_{metric}_cat4567_{time_stamp}.pdf'
    
    @staticmethod
    def adjust_xticks(ax:'list', steps:'new index', vlines_pos, x:'old index'):
        # Change ticks from index to steps and format
        xtick_pattern_step = 25
        xticks_pattern = [i for i in range(0,x.max()+xtick_pattern_step, xtick_pattern_step)]
        for axe in [*ax]:
            axe.set_xticks(xticks_pattern)
            for vline in vlines_pos:
                # Add vlines
                if vline not in axe.get_xticks():
                    axe.set_xticks(list(axe.get_xticks()) + [vline])
            xticks = axe.get_xticks()
            if 0 not in xticks:
                axe.set_xticks(list(axe.get_xticks()) + [0])
                xticks = axe.get_xticks()
            xlabels = []
            for val in xticks:
                if val == 0:
                    xlabels.append('0')
                elif val in vlines_pos:
                     xlabels.append('50/ 0')
                elif val > 0 and val-1 in steps:
                    xlabels.append(str(steps[val-1]+1))
                else:
                    xlabels.append('')
            axe.set_xticklabels(xlabels)
            axe.tick_params(**Plot.properties('tick_params'))
    
    @staticmethod
    def set_legends(fig, axes: 'list', **kwargs):
        lines_labels = [ax.get_legend_handles_labels() for ax in axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        args = {'prop' : {'size':17}, 'ncol' : 2, 'bbox_to_anchor' : (1.038, 2.25),
               'bbox_transform' : plt.gcf().transFigure}
        if len(kwargs) > 0:
            for k,v in kwargs.items(): args[k] = v
        fig.legend(lines, labels, **args)
    
    @staticmethod
    def set_xlabel(ax, xlabel):
        ax.set_xlabel(xlabel, fontsize = 20, fontweight='bold', loc='center')
    @staticmethod
    def set_fig_suptitle(fig, title):
        fig.suptitle(title, fontsize=26)
    @staticmethod
    def set_title(ax, title):
        ax.set_title(title, fontsize=16, fontweight='bold', loc='left')
    @staticmethod
    def plot_best_fitness_points(ax,x,y):
        ax.plot(x,y, 'k*', markersize=10)

if __name__ == '__main__':

    save = False
    #a = Plot_Yv3_Train_Data()
    b = Plot_Yv5_Train_Data()
    #a.plot_cat_trainloss(save =         save)
    #a.plot_cat_metrics(save =           save)
    #a.plot_cat_testloss(save =          save)
    #a.plot_cat_learning_rates(save =    save)
    b.plot_cat_trainloss(save =         save)
    b.plot_cat_metrics(save =           save)
    b.plot_cat_testloss(save =          save)
    b.plot_cat_learning_rates(save =    save)
    pass
# %%