"""
YOLO EXPERIMENTS GENERAL CONFIGS
        - PATH CONFIGS
        - Experiment details configs
"""
import os
import pathlib
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import datetime

root = pathlib.Path(__file__).parent
SILENT = False # Show status messages while executing plots..

_di5 = {
    'yv5_0' : {'name'   :   'YOLOv5_UAVDT_0',
               'date'   :   '21_Feb_2021_18h_17m',
               'tb'     :   'events.out.tfevents.1613943087.febe.6899.0',
               'model'  :   'yv5_S'
            },
    'yv5_1' : {'name'   :   'YOLOv5_UAVDT_1',
               'date'   :   '21_Feb_2021_19h_26m',
               'tb'     :   'events.out.tfevents.1613947244.febe.4710.0',
               'model'  :   'yv5_M'
            },
    'yv5_2' : {'name'   :   'YOLOv5_UAVDT_2',
               'date'   :   '21_Feb_2021_21h_42m',
               'tb'     :   'events.out.tfevents.1613956332.febe.17826.0',
               'model'  :   'yv5_L'
            },
    'yv5_3' : {'name'   :   'YOLOv5_UAVDT_3',
               'date'   :   '22_Feb_2021_11h_36m',
               'tb'     :   'events.out.tfevents.1614005399.febe.29767.0',
               'model'  :   'yv5_X'
            },
    'yv5_4' : {'name'   :   'YOLOv5_UAVDT_4',
               'date'   :   '25_Feb_2021_13h_13m',
               'tb'     :   'events.out.tfevents.1614270487.febe.3970.0',
               'model'  :   'yv5_S'
            },
    'yv5_5' : {'name'   :   'YOLOv5_UAVDT_5',
               'date'   :   '26_Feb_2021_04h_26m',
               'tb'     :   'events.out.tfevents.1614325214.febe.32724.0',
               'model'  :   'yv5_M'
            },
    'yv5_6' : {'name'   :   'YOLOv5_UAVDT_6',
               'date'   :   '26_Feb_2021_04h_25m',
               'tb'     :   'events.out.tfevents.1614325182.febe.32230.0',
               'model'  :   'yv5_L'
            },
    'yv5_7' : {'name'   :   'YOLOv5_UAVDT_7',
               'date'   :   '26_Feb_2021_04h_25m',
               'tb'     :   'events.out.tfevents.1614325148.febe.31738.0',
               'model'  :   'yv5_X'
            },
    'yv5_8' : {'name'   :   'YOLOv5_UAVDT_8',
               'date'   :   '03_Mar_2021_03h_52m',
               'tb'     :   'events.out.tfevents.1614755251.febe.12105.0',
               'model'  :   'yv5_S'
            },
    'yv5_9' : {'name'   :   'YOLOv5_UAVDT_9',
               'date'   :   '03_Mar_2021_03h_53m',
               'tb'     :   'events.out.tfevents.1614755289.febe.12519.0',
               'model'  :   'yv5_M'
            },
    'yv5_10' : {'name'   :  'YOLOv5_UAVDT_10',
               'date'   :   '03_Mar_2021_03h_53m',
               'tb'     :   'events.out.tfevents.1614755322.febe.12613.0',
               'model'  :   'yv5_L'
            },
    'yv5_11' : {'name'   :  'YOLOv5_UAVDT_11',
               'date'   :   '03_Mar_2021_00h_06m',
               'tb'     :   'events.out.tfevents.1614741591.febe.27196.0',
               'model'  :   'yv5_X'
            }
    #         },
    # 'yv5_12' : {'name'   :  'YOLOv5_UAVDT_12',
    #            'date'   :   '07_Mar_2021_00h_06m',
    #            'tb'     :   'events.out.tfevents.1615087268.febe.22541.0',
    #            'model'  :   'yv5_S'
    #         },
    # 'yv5_13' : {'name'   :  'YOLOv5_UAVDT_13',
    #            'date'   :   '07_Mar_2021_00h_07m',
    #            'tb'     :   'events.out.tfevents.1615087295.febe.22748.0',
    #            'model'  :   'yv5_M'
    #         },
    # 'yv5_14' : {'name'   :  'YOLOv5_UAVDT_14',
    #            'date'   :   '07_Mar_2021_00h_08m',
    #            'tb'     :   'events.out.tfevents.1615087417.febe.23319.0',
    #            'model'  :   'yv5_L'
    #         },
    #  'yv5_15' : {'name'   :  'YOLOv5_UAVDT_15',
    #            'date'   :   '07_Mar_2021_00h_11m',
    #            'tb'     :   'events.out.tfevents.1615087647.febe.23551.0',
    #            'model'  :   'yv5_X'
    #        },
}
_di3 = {
    'yv3_0' : {'name'   :   'YOLOv3_UAVDT_0',
               'date'   :   '28_Feb_2021_04h_35m',
               'tb'     :   'events.out.tfevents.1614498572.febe.27725.0',
               'model'  :   'yv3_tiny'
            },
    'yv3_1' : {'name'   :   'YOLOv3_UAVDT_1',
               'date'   :   '28_Feb_2021_04h_36m',
               'tb'     :   'events.out.tfevents.1614498593.febe.28080.0',
               'model'  :   'yv3'
            },
    'yv3_2' : {'name'   :   'YOLOv3_UAVDT_2',
               'date'   :   '28_Feb_2021_04h_36m',
               'tb'     :   'events.out.tfevents.1614498628.febe.28202.0',
               'model'  :   'yv3_spp'
            },
    'yv3_3' : {'name'   :   'YOLOv3_UAVDT_3',
               'date'   :   '01_Mar_2021_11h_34m',
               'tb'     :   'events.out.tfevents.1614610260.febe.9687.0',
               'model'  :   'yv3_tiny'
            },
    'yv3_4' : {'name'   :   'YOLOv3_UAVDT_4',
               'date'   :   '01_Mar_2021_11h_34m',
               'tb'     :   'events.out.tfevents.1614610273.febe.9757.0',
               'model'  :   'yv3'
            },
    'yv3_5' : {'name'   :   'YOLOv3_UAVDT_5',
               'date'   :   '01_Mar_2021_11h_35m',
               'tb'     :   'events.out.tfevents.1614610272.febe.9751.0',
               'model'  :   'yv3_spp'
            },
    'yv3_6' : {'name'   :   'YOLOv3_UAVDT_6',
               'date'   :   '04_Mar_2021_16h_41m',
               'tb'     :   'events.out.tfevents.1614887743.febe.29787.0',
               'model'  :   'yv3_tiny'
            },
    'yv3_7' : {'name'   :   'YOLOv3_UAVDT_7',
               'date'   :   '04_Mar_2021_16h_42m',
               'tb'     :   'events.out.tfevents.1614887743.febe.29790.0',
               'model'  :   'yv3'
            },
    'yv3_8' : {'name'   :   'YOLOv3_UAVDT_8',
               'date'   :   '04_Mar_2021_18h_12m',
               'tb'     :   'events.out.tfevents.1614893298.febe.13458.0',
               'model'  :   'yv3_spp'
            }
}

_pd_df_di3 = pd.DataFrame(_di3).T
_pd_df_di5 = pd.DataFrame(_di5).T

class YOLO_UAVDT_CONFIGS:
    def __init__(self): pass

    @staticmethod
    def movies_teste():
        return ["M0203", "M0205", "M0208", "M0209", "M0403", "M0601", \
                "M0602", "M0606", "M0701", "M0801", "M0802", "M1001", "M1004", \
                "M1007", "M1009", "M1101", "M1301", "M1302", "M1303", "M1401"]
    @staticmethod
    def det_data_file_yaml():
        return """
                # train and val datasets (image directory or *.txt file with image paths)

                #train: ../../../Datasets/UAVDT_YOLOv5/train/images/
                #test: ../../../Datasets/UAVDT_YOLOv5/test/images/
                val: ../../../Datasets/UAVDT_YOLOv5/test/images/

                # number of classes
                nc: 3

                # class names
                names: ['car', 'truck', 'bus']
                """

class _YC:#YoloConfigs
    def __init__(self): pass
    root = pathlib.Path(__file__).parent
    # dict for yolov5 paths
    @staticmethod
    def yv5df():
        return _pd_df_di5
    @staticmethod
    def yv3df():
        return _pd_df_di3


class _MP:#MakePaths
    def __init__(self): pass

    @staticmethod
    def exp_root_path(experiment:'str'):
        return _YC().root / experiment

    @staticmethod
    def train_path(experiment:'str', date:'str'):
        pt0 = "_".join(experiment.split("_")[:2])
        return _MP.exp_root_path(experiment) / f'{pt0}_train' / f'{str(experiment)}-{date}'
    
    @staticmethod
    def det_path(experiment:'str', date:'str'):
        pt0 = "_".join(experiment.split("_")[:2])
        return _MP.exp_root_path(experiment) / f'{pt0}_det' / f'{str(experiment)}-{date}'
    
    @staticmethod
    def tb_path(experiment:'str', date:'str', tb_file:'str'):
        pass
        return _MP.train_path(experiment, date) / tb_file
    

class _YOLO_PATHS:#MakePathsfor YOLO exps
    def __init__(self): pass
    @staticmethod
    def exp_root_paths(df):
        return {k:_MP.exp_root_path(v) for k,v in zip(df.index, df['name'])}
    @staticmethod
    def det_paths(df):
        return {k:_MP.det_path(n,d) for k,n,d in zip(df.index, df['name'], df['date'] )}
    @staticmethod
    def train_paths(df):
        return {k:_MP.train_path(n,d) for k,n,d in zip(df.index, df['name'], df['date'] )}
    @staticmethod
    def tb_paths(df):
        return {k:_MP.tb_path(n,d,t) for k,n,d,t in zip(df.index, df['name'], df['date'], df['tb'])}


class _NETsCONFIGS:
    def __init__(self): pass
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
    def fitness(ap_05, map_05_095):
        # From Submodules/yolov5/train.py.
        # This is the same fitness function for Submodules/yolov3/train.py.
        # The fitness function is equivalent to sum(0.1*map0.5 + 0.9*map0.5:0.95).
        # Ist est, a weighted combination of the AP metrics.
        #w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        #return (x[:, :4] * w).sum(1)
        return 0.1*ap_05 + 0.9*map_05_095
    
    @staticmethod
    def get_vline_pos():
        # Index of the end of the training sessions == epochs
        return [50, 100]

class _YV3_PATHS():
    def __init__(self): pass
    @staticmethod
    def get_df(): return _YC.yv3df()
    @staticmethod
    def get_exp_root_paths(): return _YOLO_PATHS.exp_root_paths(_YC.yv3df())
    @staticmethod
    def get_det_paths(): return _YOLO_PATHS.det_paths(_YC.yv3df())
    @staticmethod
    def get_train_paths(): return _YOLO_PATHS.train_paths(_YC.yv3df())
    @staticmethod
    def get_tb_paths(): return _YOLO_PATHS.tb_paths(_YC.yv3df())

class _YV3_EXP_CONFIGS():
    def __init__(self): pass

    @staticmethod
    def load_cats():
        return [
            ('yv3_0', 'yv3_3', 'yv3_6'),
            ('yv3_1', 'yv3_4', 'yv3_7'),
            ('yv3_2', 'yv3_5', 'yv3_8'),
            ]

    @staticmethod
    def plot_exp_title(name):
        table = {
            'yv3_0' : 'yv3_tiny',
            'yv3_1' : 'yv3',
            'yv3_2' : 'yv3_spp',
            'yv3_3' : 'yv3_tiny',
            'yv3_4' : 'yv3',
            'yv3_5' : 'yv3_spp',
            'yv3_6' : 'yv3_tiny',
            'yv3_7' : 'yv3',
            'yv3_8' : 'yv3_spp',
            }
        return table[name]
    
    @staticmethod
    def color_cycler(): #cmy#k
        colors = {
                'yv3-spp'  : 'y',
                'yv3'      : 'm',
                'yv3_tiny' : 'c',        
                'yv3_8'    : 'y',
                'yv3_7'    : 'm',
                'yv3_6'    : 'c',
                'yv3_5'    : 'y',
                'yv3_4'    : 'm',
                'yv3_3'    : 'c',
                'yv3_2'    : 'y',
                'yv3_1'    : 'm',
                'yv3_0'    : 'c'
                }
        return colors

    @staticmethod
    def learning_rates_schedules():
        return {
                'Scratch' : ['yv3_0', 'yv3_1', 'yv3_2'],
                'Finetune' : ['yv3_3', 'yv3_4', 'yv3_5', 'yv3_6', 'yv3_7', 'yv3_8']
                }

class _YV5_PATHS():
    def __init__(self): pass
    @staticmethod
    def get_df(): return _YC.yv5df()
    @staticmethod
    def get_exp_root_paths(): return _YOLO_PATHS.exp_root_paths(_YC.yv5df())
    @staticmethod
    def get_det_paths(): return _YOLO_PATHS.det_paths(_YC.yv5df())
    @staticmethod
    def get_train_paths(): return _YOLO_PATHS.train_paths(_YC.yv5df())
    @staticmethod
    def get_tb_paths(): return _YOLO_PATHS.tb_paths(_YC.yv5df())

class _YV5_EXP_CONFIGS():
    def __init__(self): pass

    @staticmethod
    def get_models():
        return ['yv5_S', 'yv5_M', 'yv5_L', 'yv5_X']
    @staticmethod
    def load_cats():
        # KEEP THE ORDER, yv5_S, yv5_M, yv5_L, yv5_X
        return [
            ('yv5_0', 'yv5_4', 'yv5_8' ),# 'yv5_12' ),#yv5_S
            ('yv5_1', 'yv5_5', 'yv5_9' ),# 'yv5_13' ),#yv5_M
            ('yv5_2', 'yv5_6', 'yv5_10'),# 'yv5_14'),#yv5_L
            ('yv5_3', 'yv5_7', 'yv5_11'),# 'yv5_15')#yv5_X
            ]

    @staticmethod
    def plot_exp_title(name):
        for model, nets in zip(_YV5_EXP_CONFIGS.get_models(), _YV5_EXP_CONFIGS.load_cats()):
            if name in nets: return model
    
    @staticmethod
    def color_cycler(): 
        return  {
                'yv5_X'  : 'orange',               
                'yv5_L'  : 'b',
                'yv5_M'  : 'g',
                'yv5_S'  : 'r',
                'yv5_15' : 'orange',               
                'yv5_14' : 'b',
                'yv5_13'  : 'g',
                'yv5_12'  : 'r',
                'yv5_11' : 'orange',               
                'yv5_10' : 'b',
                'yv5_9'  : 'g',
                'yv5_8'  : 'r',
                'yv5_7'  : 'orange', 'yv5_6'  : 'b', 'yv5_5'  : 'g', 'yv5_4'  : 'r',
                'yv5_3'  : 'orange', 'yv5_2'  : 'b', 'yv5_1'  : 'g', 'yv5_0'  : 'r',
                }           
        
    @staticmethod
    def learning_rates_schedules():
        return {
                'Scratch' : ['yv5_0', 'yv5_1', 'yv5_2', 'yv5_3'],
                'Finetune' : ['yv5_4', 'yv5_5', 'yv5_6', 'yv5_7',
                              'yv5_8', 'yv5_9', 'yv5_10', 'yv5_11'],
                'Custom1' : ['yv5_12', 'yv5_13', 'yv5_14', 'yv5_15']
                }


class YV5_CONFIGS(_NETsCONFIGS, _YV5_EXP_CONFIGS, _YV5_PATHS):
    def __init__(self): pass

class YV3_CONFIGS(_NETsCONFIGS, _YV3_EXP_CONFIGS, _YV3_PATHS):
    def __init__(self): pass



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
    def make_output_file_path(metric, metric_spec, net="Error", phase='train', ext='pdf'):
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
        return output_dir / f'{net}_{metric}_cat150_{time_stamp}.{ext}'
    
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
    def set_yticklabels(ax):#bar plot
        for ytick in ax.get_yticklabels():
            ytick.set_fontsize(14)
            ytick.set_fontweight('bold')
    @staticmethod
    def set_xticklabels(ax, labels):#bar plot
        ax.set_xticklabels(labels, fontsize = 14, fontweight='bold')
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
    print(YV5_CONFIGS.get_df())
    print(YV3_CONFIGS.get_df())
    pass