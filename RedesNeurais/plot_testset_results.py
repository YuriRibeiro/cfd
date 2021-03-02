#%%
from tensorboard.backend.event_processing import event_accumulator
import os, pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np

"""
['train/box_loss', 'train/obj_loss', 'train/cls_loss',
 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
 'val/box_loss', 'val/obj_loss', 'val/cls_loss',
 'x/lr0', 'x/lr1', 'x/lr2']
"""

# DEBUG:
DEBUG = False

# Maybe:
#datetime.datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%dT%H:%M:%SZ')

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
            'yv5_7' : make_path('YOLOv5_UAVDT_7', '26_Feb_2021_04h_25m', 'events.out.tfevents.1614325148.febe.31738.0')}
          #'yv5_301' : make_path('YOLOv5_UAVDT_301', '13_October_2020_14h_48m_24s', 'events.out.tfevents.1602612398.febe.21509.0'),
          #'yv5_302' : make_path('YOLOv5_UAVDT_302', '09_October_2020_15h_11m_52s', 'events.out.tfevents.1602268152.febe.16062.0')}
        
        return yv5_tb_paths
    
    @staticmethod
    def load_cats():
        return [('yv5_0', 'yv5_4'),
                ('yv5_1', 'yv5_5'),
                ('yv5_2', 'yv5_6'),
                ('yv5_3', 'yv5_7')]
                #(None, 'yv5_301'),
                #(None, 'yv5_301')]]


    @staticmethod
    def load_tb_size_guidance():
        size_guidance = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1}
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
            'train/obj_loss': 'Object Loss',
            'train/cls_loss' : 'Classif. Loss',
            'x/lr0' : 'Learning Rate 0',
            'x/lr1' : 'Learning Rate 1',
            'x/lr2' : 'Learning Rate 2',
            'fitness' : 'Fitness'
        }
        return table[name]
    
    @staticmethod
    def color_cycler():
        # One color for each network

        #custom_cycler = (cycler(color=['#EE6666', '#3388BB', '#9988DD',
        #                                '#88BB44', '#EECC55', '#FFBBBB']))
        #
        colors = {'yv5_7' : '#EE6666',
                  'yv5_6' : '#3388BB',
                  'yv5_5' : '#9988DD',
                  'yv5_4' : '#88BB44'}
        return colors
    
    @staticmethod
    def fitness(x):
        # From Submodules/yolov5/train.py :
        # It is Equivalent to: sum(0.1*map0.5 + 0.9*map0.5:0.95).
        
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (x[:, :4] * w).sum(1)


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
        
    def plot_cat_metrics(self):
        metrics = self.plot_fields()['metrics']
        experiments = self.get_cat_exps_labels()
        numexp = len(experiments)
        
        figsize = (14,12)

        for metric in metrics:
            with plt.style.context('bmh'):
                fig, ax = plt.subplots(3,2, figsize = figsize, constrained_layout=True)
                gs = ax[2, 0].get_gridspec()
                for axe in ax[2, :]: axe.remove()
                axbig = fig.add_subplot(gs[2, :])
                ax = ax.flatten()
                
                fig.suptitle(self.plot_field_title(metric), fontsize=26)
                axbig.set_xlabel('Épocas', fontsize=18)

            for idx, exp in enumerate(experiments):
                filtered_df = self.cats[exp][metrics]
                steps = self.cats[exp]['steps']
                c = self.color_cycler()[exp]
                x = filtered_df.index
                y = filtered_df[metric]

                fitness = self.cats[exp]['fitness']
                best_x = fitness.iloc[:50].argmax()
                best_y = y.iloc[best_x]
                ax[idx].plot(best_x,best_y, 'k*', markersize=10)
                best_x = fitness.iloc[50:100].argmax() + 50
                best_y = y.iloc[best_x]
                ax[idx].plot(best_x,best_y, 'k*', markersize=10)

                Plot.plot(x,y,fig,ax[idx],c, exp)
                Plot.plot(x,y,fig,axbig,c)
                ax[idx].axvline(x=50, color='black', linestyle='--', linewidth = 1)
                
                xticks = ax[idx].get_xticks()
                xlabels = []
                for val in xticks:
                    if val > 0:
                        val -= 1
                    if val in steps:
                        xlabels.append(str(steps[val]))
                    else:
                        xlabels.append("")
                ax[idx].set_xticklabels(xlabels)
            
            axbig.axvline(x=50, color='black', linestyle='--', linewidth = 1)
            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, prop={'size':20}, ncol=2, bbox_to_anchor=(1, 0.145), bbox_transform=plt.gcf().transFigure)

            xticks = axbig.get_xticks()
            xlabels = []
            for val in xticks:
                if val > 0:
                    val -= 1
                if val in steps:
                    xlabels.append(str(steps[val]))
                else:
                    xlabels.append("")
            axbig.set_xticklabels(xlabels)
    
        plt.savefig("metrics.pdf")

class Plot:
    def __init__(self): pass

    @staticmethod
    def plot(x,y,fig=None, ax=None, color=None, leg=None):
        l, = ax.plot(x, y, linewidth=1.7, color=color, label=leg)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax, l

if __name__ == '__main__':
    #if DEBUG:
    root = pathlib.Path(__file__).parent
    os.chdir(root)
    a = YV5_Train_Data()
    a.plot_cat_metrics()
    pass


# %%
