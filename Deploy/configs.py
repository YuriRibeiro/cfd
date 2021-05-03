import pathlib

# Configs root path
root = pathlib.Path(__file__).parent

# Compatible input types
compatible_video_types = ['.avi', '.mp4']
compatible_image_types = ['.png', '.jpg', '.jpeg']

#Default devices list
default_devices_list = {'cpu': 'cpu', 'GPU0':'0', 'GPU1':'1', 'GPU2':'2', 'GPU3':'3'}

# Compatible detectors
detectors_list = ['yv5_S', 'yv5_M', 'yv5_L', 'yv5_X', 'yv3_tiny', 'yv3', 'yv3_spp']

# Compatible Trackers
trackers_list = ['SORT']

weights_folder = root / 'Weights'
weight_paths = {'yv5_S'   : weights_folder / 'yv5_S(1).pt'    ,
                'yv5_M'   : weights_folder / 'yv5_M(1).pt'    ,
                'yv5_L'   : weights_folder / 'yv5_L(2).pt'    ,
                'yv5_X'   : weights_folder / 'yv5_X(3).pt'    ,
                'yv3_tiny': weights_folder / 'yv3_tiny(1).pt' ,
                'yv3'     : weights_folder / 'yv3_(1).pt'     ,
                'yv3_spp' : weights_folder / 'yv3_spp(0).pt'  }


# Detectors Default Params
yv5_detect_default_params = yv3_detect_default_params = {'device'     : 'cpu' ,
                                                        'img_size'    : 768   ,
                                                        'conf_thres'  : 0.4   ,
                                                        'iou_thres'   : 0.4   ,
                                                        'augment'     : False ,
                                                        'agnostic_nms': True  ,
                                                        'classes'     : None  }

yv5_S_detect_default_params = yv5_M_detect_default_params= yv5_detect_default_params.copy(); yv5_S_detect_default_params.update({'img_size': 1280})

detector_default_params ={ 'yv5_S'   : yv5_S_detect_default_params ,
                           'yv5_M'   : yv5_M_detect_default_params ,
                           'yv5_L'   : yv5_detect_default_params   ,
                           'yv5_X'   : yv5_detect_default_params   ,
                           'yv3_tiny': yv3_detect_default_params   ,
                           'yv3'     : yv3_detect_default_params   ,
                           'yv3_spp' : yv3_detect_default_params   }


# Trackers default params for the above networks
tracker_sort_default_params = { 'yv5_S':    {'max_age': 15, 'min_hits': 9, 'iou_threshold': 0.5} ,
                                'yv5_M':    {'max_age': 1 , 'min_hits': 9, 'iou_threshold': 0.5} ,
                                'yv5_L':    {'max_age': 15, 'min_hits': 6, 'iou_threshold': 0.5} ,
                                'yv5_X':    {'max_age': 15, 'min_hits': 9, 'iou_threshold': 0.5} ,
                                'yv3_tiny': {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3} ,
                                'yv3':      {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3} ,
                                'yv3_spp':  {'max_age': 15, 'min_hits': 3, 'iou_threshold': 0.5} }

tracker_default_params = { 'SORT' : tracker_sort_default_params }

# UAVDT class bbox colors BGR. 0: car, 1: Bus, 2: Truck
uavdt_class_colors_rgb = {0 : (255,0,0) ,
                          1 : (0,255,0) ,
                          2 : (0,0,255) }