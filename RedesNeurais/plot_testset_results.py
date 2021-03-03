#%%
from tensorboard.backend.event_processing import event_accumulator
import os, pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

"""
['train/box_loss', 'train/obj_loss', 'train/cls_loss',
 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
 'val/box_loss', 'val/obj_loss', 'val/cls_loss',
 'x/lr0', 'x/lr1', 'x/lr2']
"""

# DEBUG:
DEBUG = False

class YV5_Config:
    def __init__(self): pass

    @staticmethod
    def load_yv5_tb_paths():
        root =  pathlib.Path(__file__).parent
        make_path = lambda exp, date, tb: root / exp / 'YOLOv5_UAVDT_train' / f'{exp}-{date}' / tb
        yv5_tb_paths = {
            'yv5_0' : make_path('YOLOv5_UAVDT_0', '21_Feb_2021_18h_17m', 'events.out.tfevents.1613943087.febe.6899.0'),
            'yv5_1' : make_path('YOLOv5_UAVDT_1', '21_Feb_2021_19h_26m', 'events.out.tfevents.1613947244.febe.4710.0'),
            'yv5_2' : make_path('YOLOv5_UAVDT_2', '21_Feb_2021_21h_42m', 'events.out.tfevents.1613956332.febe.17826.0'),
            'yv5_3' : make_path('YOLOv5_UAVDT_3', '22_Feb_2021_11h_36m', 'events.out.tfevents.1614005399.febe.29767.0'),
            'yv5_4' : make_path('YOLOv5_UAVDT_4', '25_Feb_2021_13h_13m', 'events.out.tfevents.1614270487.febe.3970.0'),
            'yv5_5' : make_path('YOLOv5_UAVDT_5', '26_Feb_2021_04h_26m', 'events.out.tfevents.1614325214.febe.32724.0'),
            'yv5_6' : make_path('YOLOv5_UAVDT_6', '26_Feb_2021_04h_25m', 'events.out.tfevents.1614325182.febe.32230.0'),
            'yv5_7' : make_path('YOLOv5_UAVDT_7', '26_Feb_2021_04h_25m', 'events.out.tfevents.1614325148.febe.31738.0')
            }
          #'yv5_301' : make_path('YOLOv5_UAVDT_301', '13_October_2020_14h_48m_24s', 'events.out.tfevents.1602612398.febe.21509.0'),
          #'yv5_302' : make_path('YOLOv5_UAVDT_302', '09_October_2020_15h_11m_52s', 'events.out.tfevents.1602268152.febe.16062.0')}
        
        return yv5_tb_paths
    
    @staticmethod
    def load_cats():
        return [
            ('yv5_0', 'yv5_4'),
            ('yv5_1', 'yv5_5'),
            ('yv5_2', 'yv5_6'),
            ('yv5_3', 'yv5_7')
            ]
                #(None, 'yv5_301'),
                #(None, 'yv5_301')]]


    @staticmethod
    def load_tb_size_guidance():
        size_guidance = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1
            }
        return size_guidance
    
    @staticmethod
    def plot_fields():
        metrics = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'fitness']
        train_loss = ['train/box_loss', 'train/obj_loss', 'train/cls_loss']
        test_loss = ['val/box_loss', 'val/obj_loss', 'val/cls_loss']
        lr = ['x/lr0', 'x/lr1', 'x/lr2']
        d = {'metrics':metrics, 'train_loss':train_loss, 'test_loss':test_loss, 'lr':lr}
        return d
    
    @staticmethod
    def plot_field_title(name):
        table = {
            'metrics/precision' : 'Precision',
            'metrics/recall' : 'Recall',
            'metrics/mAP_0.5' : 'mAP@0.5',
            'metrics/mAP_0.5:0.95' : 'mAP@0.5:0.95',
            'train/box_loss' : 'Box Loss',
            'train/obj_loss': 'Object. Loss',
            'train/cls_loss' : 'Classif. Loss',
            'val/box_loss' : 'Box Loss',
            'val/obj_loss' : 'Object. Loss',
            'val/cls_loss' : 'Classif. Loss',
            'x/lr0' : 'LR 0 (Box)',
            'x/lr1' : 'LR 1 (Object.)',
            'x/lr2' : 'LR 2 (Classif.)',
            'fitness' : 'Fitness'
            }
        return table[name]

    @staticmethod
    def plot_exp_title(name):
        table = {
            'yv5_4' : 'yv5_S',
            'yv5_5' : 'yv5_M',
            'yv5_6' : 'yv5_L',
            'yv5_7' : 'yv5_X'
            }
        return table[name]
    
    @staticmethod
    def color_cycler():
        colors = {
                'yv5_7' : '#EE6666',
                'yv5_6' : '#3388BB',
                'yv5_5' : '#9988DD',
                'yv5_4' : '#88BB44',
                'yv5_3' : '#EE6666',
                'yv5_2' : '#3388BB',
                'yv5_1' : '#9988DD',
                'yv5_0' : '#88BB44'
                }
        return colors
    
    @staticmethod
    def fitness(x):
        # From Submodules/yolov5/train.py :
        # It is Equivalent to: sum(0.1*map0.5 + 0.9*map0.5:0.95).
        
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (x[:, :4] * w).sum(1)

    @staticmethod
    def get_vline_pos():
        return [50]

    @staticmethod
    def learning_rates_schedules():
        return {
                'Scratch' : ['yv5_0', 'yv5_1', 'yv5_2', 'yv5_3'],
                'Finetune' : ['yv5_4', 'yv5_5', 'yv5_6', 'yv5_7']
                }

class YV5_Train_Data(YV5_Config):
    def __init__(self):
        size_guidance = YV5_Config.load_tb_size_guidance()
        self.yv5_tb_paths = YV5_Config.load_yv5_tb_paths()
        
        #Experiment names
        self.exp = self.yv5_tb_paths.keys()
        # Individual dataframes for each one experiment
        self.individual_exp_df = {}

        for k,v in self.yv5_tb_paths.items():
            self.ea = event_accumulator.EventAccumulator(str(v),
                                                         size_guidance=size_guidance)
            self.ea.Reload()
            self.scalar_tags = self.ea.Tags()['scalars']
            self.individual_exp_df[k] = self._to_dataframe()
        
        self._cat_df()
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
        cats = YV5_Config.load_cats()
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
                x = filtered_df.index
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

            output_file_path = Plot.make_output_file_path(metric, self.plot_field_title(metric))
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
                fig, ax = plt.subplots(4,3, figsize = Plot.properties('figsize_big'),
                                      constrained_layout=True, sharex=True)
                ax = ax.flatten()
                fig2, ax2 = plt.subplots(1,3, figsize=Plot.properties('figsize_medium'),
                                        constrained_layout=True)
                ax2 = ax2.flatten()
        
        for metric in metrics:
            idx = 0 + const
            for exp in experiments:
                filtered_df = self.cats[exp][metrics]
                steps = self.cats[exp]['steps']
                c = self.color_cycler()[exp]
                x = filtered_df.index
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
            output_file_path = Plot.make_output_file_path(plot_metric,"" )
            fig.savefig(output_file_path)
            output_file_path = Plot.make_output_file_path(plot_metric+"_resumed","" )
            fig2.savefig(output_file_path)
        
    def plot_cat_trainloss(self, save=False):
        self._plot_cat_loss(save=save, plot_metric = 'train_loss')
    
    def plot_cat_testloss(self, save=False):
        self._plot_cat_loss(save=save, plot_metric = 'test_loss')

    def plot_cat_learning_rates(self, save = False):
        metrics = self.plot_fields()['lr']
        figsize = Plot.properties('figsize_big')
        
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
                    x = filtered_df.index
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
        output_file_path = Plot.make_output_file_path(metric, self.plot_field_title(metric))
        if save: plt.savefig(output_file_path)

class Plot:
    def __init__(self): pass

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
    def make_output_file_path(metric, metric_spec, net='yv5', phase='train'):
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
        return output_dir / f'{metric}_cat4567_{time_stamp}.pdf'
    
    @staticmethod
    def adjust_xticks(ax:'list', steps:'new index', vlines_pos, x:'old index'):
        # Change ticks from index to steps and format
        xtick_pattern_step = 10
        xticks_pattern = [i for i in range(0,x.max()+xtick_pattern_step, xtick_pattern_step)]
        for axe in [*ax]:
            axe.set_xticks(xticks_pattern)
            for vline in vlines_pos:
                # Add vlines
                axe.set_xticks(list(axe.get_xticks()) + [vline])
            xticks = axe.get_xticks()
            if 0 not in xticks:
                axe.set_xticks(list(axe.get_xticks()) + [0])
                xticks = axe.get_xticks()
            xlabels = []
            for val in xticks:
                if val == 0 or val in vlines_pos:
                     xlabels.append('0')
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

        args = {'prop' : {'size':20}, 'ncol' : 2, 'bbox_to_anchor' : (1.035, 2.3),
               'bbox_transform' : plt.gcf().transFigure}
        if len(kwargs) > 0:
            for k,v in kwargs.items(): args[k] = v
        
        fig.legend(lines, labels, **args)
    
    @staticmethod
    def set_xlabel(ax, xlabel):
        ax.set_xlabel(xlabel, fontsize = 18, fontweight='bold', loc='center')
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
    #if DEBUG:
    root = pathlib.Path(__file__).parent
    os.chdir(root)
    a = YV5_Train_Data()
    a.plot_cat_trainloss(save =         False)
    a.plot_cat_metrics(save =           False)
    a.plot_cat_testloss(save =          False)
    a.plot_cat_learning_rates(save =    False)
    pass


# %%
