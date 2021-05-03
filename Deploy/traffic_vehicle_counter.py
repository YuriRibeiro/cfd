#%%
"""
Formatos de entrada permitidos:
    *arquivo de vídeo [mp4, avi]
    *pasta com imagens numeradas
    *pasta com vídeos
    *pasta com sub-pastas contendo imagens
"""
import pathlib
import sys
import configs
import cv2
import numpy as np
import os
import random
import shutil
from IPython.display import HTML, display, Javascript
import ipywidgets as widgets
import matplotlib.pyplot as plt
import json

sys.path.append(str(configs.root / 'sort'))
sys.path.append(str(configs.root / 'yv5'))
sys.path.append(str(configs.root / 'yv3'))

import sort

class GatherData():
    def __init__(self, video_path=''):
        self.process = None
        video_types = configs.compatible_video_types
        img_types = configs.compatible_image_types

        self.input_organization = None
        video_file = False
        folder_with_images = False
        folder_with_videos = False
        folder_with_folders_of_images = False
        
        self.video_path = video_path = pathlib.Path(video_path)
        if video_path.is_dir():
            files = list(video_path.glob('*'))
            if len(files) == 0:
                raise Exception(f'Folder {video_path} empty.')
            first_file = files[0]
            if first_file.is_file():
                # check if all items are files
                file_type = [i.is_file() for i in files]
                if not all(file_type):
                    raise Exception(f'Not same fyle type for every item in {video_path}. Files and Folders mixed.')
                file_type = first_file.suffix
                if file_type in img_types:
                    file_types = [i.suffix in img_types for i in files]
                    if not all(file_types):
                        raise Exception(f'Not same fyle type for every item in {video_path}. Images and videos mixed.')
                    folder_with_images = True
                elif file_type in video_types:
                    file_types = [i.suffix in video_types for i in files]
                    if not all(file_types):
                        raise Exception(f'Not same fyle type for every item in {video_path}. Images and videos mixed.')
                    folder_with_videos = True
                else:
                    raise Exception(f'File type {file_type} not expected for {first_file}. Not supported file type.')
            elif first_file.is_dir():
                #check if all other items are folders:
                folders = [i.is_dir() for i in files]
                if not all(folders):
                    raise Exception(f'Not same fyle type for every item in {video_path}.')
                #check if inside all folders there is images
                for subFolder in files:
                    subFolderFiles = list(subFolder.glob('*'))
                    file_type = [i.is_file() for i in subFolderFiles]
                    if len(file_type) == 0:
                        raise Exception(f'Folder {subFolder} empty.')
                    if not all(file_type):
                        raise Exception(f'Not same fyle type for every item in {subFolder}. Files and Folders mixed.')
                    file_types = [i.suffix in img_types for i in subFolderFiles]
                    if not all(file_types):
                        raise Exception(f'Not same fyle type for every item in {subFolder}. Images and videos mixed.')
                folder_with_folders_of_images = True
            else:
                raise Exception(f'File type {file_type} not expected for {first_file}. It should be a directory or a video.')

        elif video_path.is_file():
            if video_path.suffix in video_types:
                video_file = True
            else:
                raise Exception(f'File type {file_type} not expected for {first_file}. It should be a directory or a video.')
        else:
            raise Exception(f'File {video_path} not found.')

        #sanity check
        if not (video_file ^ folder_with_images ^ folder_with_videos ^ folder_with_folders_of_images):
            raise Exception('Sanity check failed. "not (video_file ^ folder_with_images ^ folder_with_videos ^ folder_with_folders_of_images)"')
        
        if video_file:
            self.process = self._preprocess_video_file
            self.input_organization = 'video_file'
        elif folder_with_images:
            self.process = self._preprocess_folder_with_images
            self.input_organization = 'folder_with_images'
        elif folder_with_videos:
            self.process = self._preprocess_folder_with_videos
            self.input_organization = 'folder_with_videos'
        elif folder_with_folders_of_images:
            self.process = self._preprocess_folder_with_folders_of_images
            self.input_organization = 'folder_with_folders_of_images'
        else:
            raise Exception('Error. Unknown Condition.')

    def _preprocess_video_file(self):
        return [self.video_path]

    def _preprocess_folder_with_images(self):
        return [self.video_path]
    
    def _preprocess_folder_with_videos(self):
        videos = []
        for video_path in list(self.video_path.glob('*')): videos.append(video_path)
        return videos

    def _preprocess_folder_with_folders_of_images(self):
        return self._preprocess_folder_with_videos()

class Detector():
    def __init__(self, detector='yv5_S', params ={}):
        '''
        detector: 'yv5_S', 'yv5_M', 'yv5_L', 'yv5_X', ... (see configs.detectors_list)
        params: dict cotaining parameters to change from the default configs.'net'_detect_default_params.
                Example: {'img_size': 320, 'conf_thresh' : 0.45, 'device':'cpu'}
        '''
        if not detector in configs.detectors_list:
            raise Exception(f'Detector {detector} not in configs.detector_list.')
        self.detector = detector
        self.detector_family = detector[:3]

        if not isinstance(params, dict):
            raise Exception('Parameters should be dict instance. Empty dict means default parameters.')
        self.params = self._process_params(params)

        self.weights_path = self._load_weights_path()

    def _load_default_params(self):
        return configs.detector_default_params[self.detector]

    def _process_params(self, params):
        default_params = self._load_default_params()
        if params == {}: return default_params
        # Check params keys
        for k in params.keys():
            if not k in default_params.keys(): raise Exception(f'Parameter error: Parameter {k} not recognized')
        # Update default params with the new params and return
        default_params.update(params)
        return default_params
    
    def _load_weights_path(self):
        return configs.weight_paths[self.detector]

    def _load_detector(self, input_path):
        if self.detector_family == 'yv5':
            from yv5 import detect as yv5detect
            return yv5detect.detect(input_path, self.weights_path, **self.params)
        elif self.detector_family == 'yv3':
            from yv3 import detect as yv3detect
            return yv3detect.detect(input_path, self.weights_path, **self.params)
    
    def get_sort_tracker_optimal_parameters(self):
        return configs.tracker_sort_default_params[self.detector]

class TrackerBase():
    def __init__(self):
        self.mot_labels={} #list
        self.tracked_vehicles_trajectory={} #np.array
        self.vehicles_on_the_frame={} #np.array

class TrackerSort(TrackerBase):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        super().__init__()
        self._params = {'max_age':max_age, 'min_hits':min_hits, 'iou_threshold':iou_threshold}
        self.colors = {} #id:random_color
        self.colors_handler = self._default_colors_handler 
        self.tracker = self._create_tracker()
    
    def get_tracker_parameters(self):
        return self._params

    def _create_tracker(self):
        import sort
        return sort.Sort(**self._params)
    
    def _default_colors_handler(self):
        rdn = random.randint
        return (rdn(0,255), rdn(0,255), rdn(0,255))

    def update(self, dets, frame_number):
        if len(dets) == 0:
            bboxes_atual = np.zeros((0, 5))
        else:
            bboxes_atual = np.zeros((len(dets), 5))
            for idx, (x,y,w,h,classe,score) in enumerate(dets):
                bboxes_atual[idx] = [x, y, x+w, y+h, score]
        
        # Analisar o frame atual e identificar os bounding boxes id (update SORT)
        track_bbs_ids = self.tracker.update(bboxes_atual)

        # Passar as coordenadas para o padrão: [frame,x,y,w,h,idx]
        track_bbs_ids[:, 2] = track_bbs_ids[:, 2] - track_bbs_ids[:, 0]
        track_bbs_ids[:, 3] = track_bbs_ids[:, 3] - track_bbs_ids[:, 1]
        vec_xc = track_bbs_ids[:, 0] + track_bbs_ids[:, 2] / 2
        vec_yc = track_bbs_ids[:, 1] + track_bbs_ids[:, 3] / 2

        track_bbs_ids = np.rint(track_bbs_ids).astype(int)
        vec_xc = np.rint(vec_xc).astype(int)
        vec_yc = np.rint(vec_yc).astype(int)
        this_frame_ids = track_bbs_ids[:,-1]
        
        for idx, xc, yc  in zip(track_bbs_ids[:,-1], vec_xc, vec_yc):
            # Guardar a trajetória do centro do veículo IDx
            if idx in self.tracked_vehicles_trajectory:
                self.tracked_vehicles_trajectory[idx].append((frame_number,xc,yc))
            else:
                self.tracked_vehicles_trajectory[idx] =  [(frame_number,xc,yc)]
                self.colors[idx] = self.colors_handler()
        
        # Atualizar a variável global
        self.vehicles_on_the_frame[frame_number] = this_frame_ids.copy()
        self.mot_labels[frame_number] = track_bbs_ids.copy()


class CountBarrier():
    """
    # Driver program to test above functions: 
    p1 = Point(1, 1) 
    q1 = Point(10, 1) 
    p2 = Point(1, 2) 
    q2 = Point(10, 2) 

    if doIntersect(p1, q1, p2, q2): 
        print("Yes") 
    else: 
        print("No") 
        
    # This code is contributed by Ansh Riyal
    """
    def __init__(self,
                p1 : "tuple (x,y)",
                q1 : "tuple (x,y)",
                start_frame: "Frame number that the barrier will START to count cars on video sequences" = 0,
                end_frame : "Frame number that the barrier will STOP to count cars on video sequences" = 0,
                name : "Name to identify the barrier" = "0"):
        self.p1 = p1
        self.q1 = q1
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.name = name
        
        self.counter = 0
        self.intersection_frames = {}
    
    def add_counter(self, value = 1):
        self.counter += value
    
    def add_intersection_frame(self, frame_number, value=1):
        if frame_number in self.intersection_frames:
            self.intersection_frames[frame_number] += value
        else:
            self.intersection_frames[frame_number] = value
    
    def onSegment(self, p, q, r): 
        """
        Given three colinear points p, q, r, the function checks if 
        point q lies on line segment 'pr' 
        """
        if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
            return True
        return False

    def orientation(self, p, q, r):
        """
        to find the orientation of an ordered triplet (p,q,r) 
        function returns the following values: 
         0 : Colinear points 
         1 : Clockwise points 
         2 : Counterclockwise 
        
        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
        for details of below formula. 
        """
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
        if (val > 0): 
            # Clockwise orientation 
            return 1
        elif (val < 0): 
            # Counterclockwise orientation 
            return 2
        else:
            # Colinear orientation 
            return 0

    def doIntersect(self,p2,q2): 
        """
        The main function that returns true if 
        the line segment 'p1q1 (barrier)' and 'p2q2' intersect. 
        """
        p1, q1 = self.p1, self.q1

        # Find the 4 orientations required for 
        # the general and special cases 
        o1 = self.orientation(p1, q1, p2) 
        o2 = self.orientation(p1, q1, q2) 
        o3 = self.orientation(p2, q2, p1) 
        o4 = self.orientation(p2, q2, q1) 
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
        # Special Cases 
        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
        if ((o1 == 0) and self.onSegment(p1, p2, q1)): 
            return True
        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
        if ((o2 == 0) and self.onSegment(p1, q2, q1)): 
            return True
        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
        if ((o3 == 0) and self.onSegment(p2, p1, q2)): 
            return True
        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
        if ((o4 == 0) and self.onSegment(p2, q1, q2)): 
            return True
        # If none of the cases 
        return False

    def __str__(self):
        return f"Nome da Barreira: {self.name}. Quantidade de veículos que passaram: {self.counter}."

class Analysis():   
    def __init__(self,
                input_path,
                output_path,
                detector : 'Detector Object',
                tracker : 'Tracker Object'=None,
                barriers : 'Dict {video_name:[barriers]} of Count Barrier Objects'=None,
                save_detection_json=True,
                save_tracking_json=True,
                save_det_imgs_without_headers=False,
                save_det_imgs_with_headers=True,
                save_track_imgs_without_headers=False,
                save_track_imgs_with_headers=True,
                save_track_imgs_lines = False,
                save_track_img_barriers = True,
                save_det_vid_without_headers=False,
                save_det_vid_with_headers=False,
                save_track_vid_without_headers=False,
                save_track_vid_with_headers=False,
                save_track_vid_lines = False,
                save_track_vid_barriers = False,
                clear_output_folder=True
                ):
        if pathlib.Path(input_path) == pathlib.Path(output_path):
            raise Exception('Input path should be different of the output path.')
        
        rp1 = os.path.realpath(input_path).split(os.sep)
        rp2 = os.path.realpath(output_path).split(os.sep)
        check = [i1 == i2 for i1,i2 in zip(rp1,rp2)]
        if all(check):
            raise Exception('Output path should not be subpath of the input path.')

        # Inputsave_track_img_barriers_wl = True,
        input_videos = GatherData(video_path=input_path)
        self.input_videos_paths = input_videos.process()
        self.input_videos_organization = input_videos.input_organization

        # Output
        output_fopath = pathlib.Path(output_path)
        if not output_fopath.exists() or (output_fopath.exists() and len(list(output_fopath.glob('*'))) == 0):
            pass
        elif output_fopath.is_dir() and len(list(output_fopath.glob('*'))) != 0:
            if clear_output_folder:
                shutil.rmtree(str(output_fopath))
            else:
                raise Exception(f'Output folder {str(output_fopath)} not empty.')
        elif output_fopath.is_file():
            if clear_output_folder:
                os.remove(str(output_fopath))
            else:
                raise Exception(f'Output folder {str(output_fopath)} is a file.')
        else:
            raise Exception(f'Problems to create the output folder at {str(output_fopath)}.')
        
        output_fopath.mkdir(parents=True, exist_ok=True)
        self.output_root_fopath = output_fopath

        # Detector
        self.detector = detector

        # Tracker
        self.tracker = tracker

        # Count barriers
        self.barriers = barriers
        if not self.barriers == None and not isinstance(self.barriers, dict):
            raise Exception("Barriers should be a dict {'movie_name':[barriers], 'movie_name2':[barriers_2]} style.")
        # Check if count barriers have unique names
        if isinstance(self.barriers, dict):
            for vid in self.barriers:
                bnames = []
                for b in self.barriers[vid]:
                    if not b.name in bnames: bnames.append(b.name)
                    else:
                        raise Exception(f'Counting barriers NAMES of the video {vid} are not unique.')


        # Save Parameters
        self.save_detection_json            = save_detection_json
        self.save_tracking_json             = save_tracking_json
        self.save_det_imgs_without_headers  = save_det_imgs_without_headers
        self.save_det_imgs_with_headers     = save_det_imgs_with_headers
        self.save_track_imgs_without_headers= save_track_imgs_without_headers
        self.save_track_imgs_with_headers   = save_track_imgs_with_headers
        self.save_track_imgs_lines          = save_track_imgs_lines
        self.save_track_img_barriers        = save_track_img_barriers

        self.save_det_vid_without_headers   = save_det_vid_without_headers
        self.save_det_vid_with_headers      = save_det_vid_with_headers
        self.save_track_vid_without_headers = save_track_vid_without_headers
        self.save_track_vid_with_headers    = save_track_vid_with_headers
        self.save_track_vid_lines           = save_track_vid_lines
        self.save_track_vid_barriers        = save_track_vid_barriers

        if self.tracker == None:
            self.save_track_imgs_without_headers = False
            self.save_track_imgs_with_headers = False
            self.save_track_img_barriers = False
            self.save_track_vid_without_headers = False
            self.save_track_vid_with_headers = False
            self.save_track_vid_lines = False
            self.save_track_vid_barriers = False
        
        if (self.barriers == None) or (not isinstance(self.barriers, dict)) or (len(self.barriers) == 0):
            self.save_track_img_barriers = False
            self.save_track_vid_barriers = False

        self.tracking_activated = self.save_track_imgs_without_headers or \
                                  self.save_track_imgs_with_headers    or \
                                  self.save_track_vid_without_headers  or \
                                  self.save_track_vid_with_headers
        
        self.barriers_activated = self.save_track_img_barriers or \
                                  self.save_track_vid_barriers

        self.save_videos =  self.save_det_vid_without_headers   or \
                            self.save_det_vid_with_headers      or \
                            self.save_track_vid_without_headers or \
                            self.save_track_vid_with_headers    or \
                            self.save_track_vid_lines           or \
                            self.save_track_vid_barriers

    def _draw_bbox(self, img, xywh, place_headers=False, colors_handler=configs.uavdt_class_colors_rgb):
        img = img.copy()
        tl = round(0.001 * max(img.shape[0], img.shape[1])) + 1
        colors = colors_handler
        for x,y,w,h,classe,conf in xywh:
            cv2.rectangle(img, (x,y), (x+w, y+h), colors[classe], tl)
            if place_headers:
                label = f'{conf:.2f}'
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
                c2 = x + t_size[0], y - t_size[1] - 3
                cv2.rectangle(img, (x,y), c2, colors[classe], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x, y-2), 0, tl / 3, (0,0,0), thickness=tf, lineType=cv2.LINE_AA)
        return img

    def _draw_bbox_tracking(self, img, mot_data, place_headers=False, color_handler=None):
        img = img.copy()
        tl = round(0.001 * max(img.shape[0], img.shape[1])) + 1
        if color_handler == None: raise Exception('Color handler should be specified.')
        colors = color_handler
        for (x,y,w,h,idx) in mot_data:
            cv2.rectangle(img, (x,y), (x+w, y+h), colors[idx], tl)
            if place_headers:
                label = f'{idx}'
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
                c2 = x + t_size[0], y - t_size[1] - 3
                cv2.rectangle(img, (x,y), c2, colors[idx], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x, y-2), 0, tl / 3, (0,0,0), thickness=tf, lineType=cv2.LINE_AA)
        return img
    
    def _draw_bbox_and_line_tracking(self, draw_bbox_img, this_frame_ids, veh_trajectories, color_handler=None, barriers=None, frame=None):
        img = draw_bbox_img.copy()
        tl = round(0.001 * max(img.shape[0], img.shape[1])) + 1
        for idx in this_frame_ids:
            trajectory = veh_trajectories[idx]
            if len(trajectory) <= 1: continue
            color = color_handler[idx]
            for i in range(len(trajectory)-1):
                p1 = trajectory[i][1:]
                p2 = trajectory[i+1][1:]
                cv2.line(img, p1, p2, color, tl, cv2.LINE_AA)
        
        if isinstance(barriers,list):
            for bi in barriers:
                if not bi.start_frame <= frame <= bi.end_frame: continue
                p1 = bi.p1
                p2 = bi.q1
                color = (0,255,255) if frame in bi.intersection_frames else (255,0,0)
                cv2.line(img, p1, p2, color, tl, cv2.LINE_AA)
                if frame in bi.intersection_frames:
                    label = f'{bi.name}:{bi.counter}'
                    tf = max(tl - 1, 1)
                    x, y=p1
                    t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
                    pos1 = None
                    if x+t_size[0] <= img.shape[1]:
                        pos1 = x+t_size[0]
                    elif x-t_size[0] >= 0:
                        pos1 = x-t_size[0]
                    pos2 = None
                    if y-t_size[1] >= 0:
                        pos2 = y-t_size[1] -3
                    elif y+t_size[1] <= img.shape[0]:
                        pos2 = y+t_size[1] + 3
                    c2 = pos1, pos2
                    cv2.rectangle(img, (x,y), c2, (255,255,255), -1, cv2.LINE_AA)  # filled
                    x= min(x, pos1)
                    y= max(y, pos2)
                    cv2.putText(img, label, (x, y-2), 0, tl / 3, (0,0,0), thickness=tf, lineType=cv2.LINE_AA)
        return img

    def _plot_barriers_crossings_vs_frame(self, blist:'List'=None):
        if blist == None: return
        if not isinstance(blist, list): raise Exception(f'Wrong input type. Expected a list, and got {type(blist)}')
        # Plot
        plots = {}
        total_crossings = {}
        for bi in blist:
            start_frame = bi.start_frame
            end_frame = bi.end_frame
            instersection_frames = list(bi.intersection_frames.keys())
            intersections_count = bi.intersection_frames #dict
            
            frames = np.arange(start_frame, end_frame + 1)
            crossing = np.zeros((len(frames),))
            for i in instersection_frames:
                crossing[i-1] = intersections_count[i]

            with plt.style.context('bmh'):
                fig, ax = plt.subplots(1,1, figsize=(10,6), constrained_layout=True)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.plot(frames, crossing)
                ax.set_xlabel('Frame Number', fontsize=14)
                ax.set_ylabel('Crossings', fontsize=14)
                ax.set_title(f'Barrier {bi.name} -- Crossings vs Frame', fontsize=16)
                ax.set_xticks(frames)
            total_crossings[bi.name] = np.sum(crossing)
            plots[bi.name] = fig
        
        return plots, total_crossings


    def start(self):

        for input_vid in self.input_videos_paths:
            first_vid_iter = True
            # Create Output Folders
            video_name = input_vid.stem
            input_vid_type = 'image' if input_vid.is_dir() else 'video'

            output_fipath_dets_json = self.output_root_fopath / f'dets_{video_name}.json'
            first_line_detection_json = True
            output_det_json_file = open(str(output_fipath_dets_json),'w')

            output_fipath_tracking_json = self.output_root_fopath / f'tracking_{video_name}.json'

            output_fopath_det_imgs_without_headers = self.output_root_fopath / f'dets_{video_name}_imgs'
            if self.save_det_imgs_without_headers: output_fopath_det_imgs_without_headers.mkdir(parents=True, exist_ok=True)
            output_fopath_det_imgs_with_headers = self.output_root_fopath / f'dets_{video_name}_imgs_wh'
            if self.save_det_imgs_with_headers: output_fopath_det_imgs_with_headers.mkdir(parents=True, exist_ok=True)

            output_fopath_track_imgs_without_headers = self.output_root_fopath / f'track_{video_name}_imgs'
            if self.save_track_imgs_without_headers: output_fopath_track_imgs_without_headers.mkdir(parents=True, exist_ok=True)
            output_fopath_track_imgs_with_headers = self.output_root_fopath / f'track_{video_name}_imgs_wh'
            if self.save_track_imgs_with_headers: output_fopath_track_imgs_with_headers.mkdir(parents=True, exist_ok=True)

            output_fopath_track_imgs_lines = self.output_root_fopath / f'track_{video_name}_imgs_lines'
            if self.save_track_imgs_lines: output_fopath_track_imgs_lines.mkdir(parents=True, exist_ok=True)

            output_fopath_track_img_barriers = self.output_root_fopath / f'track_{video_name}_imgs_barriers'
            if self.save_track_img_barriers: output_fopath_track_img_barriers.mkdir(parents=True, exist_ok=True)

            output_fopath_barriers_plots = self.output_root_fopath / f'barriers_{video_name}_crossings_plots'
            if (self.barriers_activated) and (video_name in self.barriers): output_fopath_barriers_plots.mkdir(parents=True, exist_ok=True)

            output_fipath_det_vid = self.output_root_fopath / f'dets_{video_name}.mp4'
            output_fipath_det_vid_with_headers = self.output_root_fopath / f'dets_{video_name}_wh.mp4'
            output_fipath_track_vid = self.output_root_fopath / f'track_{video_name}.mp4'
            output_fipath_track_vid_with_headers = self.output_root_fopath / f'track_{video_name}_wh.mp4'
            output_fipath_track_vid_lines = self.output_root_fopath / f'track_{video_name}_lines.mp4'
            output_fipath_track_vid_barriers = self.output_root_fopath / f'track_{video_name}_barriers.mp4'
            

            # Set bariers for this movie
            this_video_barriers_activated = False
            if (self.barriers_activated) and (video_name in self.barriers):
                barriers = self.barriers[video_name]
                this_video_barriers_activated = True

            # Detections Processing
            detections = self.detector._load_detector(input_path=input_vid)
            for data in detections:
                if data == None: continue
                frame, img, dets = data #dets: (x,y,w,h,classe,conf)
                if self.save_detection_json:
                    det_str = str(dets)
                    det_table = det_str.maketrans({" ":"", "(":"[", ")":"]"})
                    if not first_line_detection_json:
                        string = f',{{"frame":{frame},"dets":{det_str.translate(det_table)}}}'
                    else:
                        string = f'[{{"frame":{frame},"dets":{det_str.translate(det_table)}}}'
                        first_line_detection_json = False
                    
                    output_det_json_file.write(string)
                
                if first_vid_iter: # Open Vid Recorders
                    first_vid_iter = False
                    fourcc = 'mp4v'  # output video codec
                    if self.save_videos and input_vid_type=='video':
                        cap = cv2.VideoCapture(str(input_vid))
                        if not cap.isOpened(): raise Exception(f"Error opening video {str(input_vid)}.")
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                    
                    elif self.save_videos and input_vid_type=='image':
                        fps=30
                        h = int(img.shape[0])
                        w = int(img.shape[1])

                    # Create VidWriter
                    if self.save_det_vid_without_headers:
                        vid_writer_det_without_headers = cv2.VideoWriter(str(output_fipath_det_vid), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if self.save_det_vid_with_headers:
                        vid_writer_det_with_headers = cv2.VideoWriter(str(output_fipath_det_vid_with_headers), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if self.save_track_vid_without_headers:
                        vid_writer_track_without_headers = cv2.VideoWriter(str(output_fipath_track_vid), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if self.save_track_vid_with_headers:
                        vid_writer_track_with_headers = cv2.VideoWriter(str(output_fipath_track_vid_with_headers), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if self.save_track_vid_lines:
                        vid_writer_track_lines = cv2.VideoWriter(str(output_fipath_track_vid_lines), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if self.save_track_vid_barriers:
                        vid_writer_track_barriers = cv2.VideoWriter(str(output_fipath_track_vid_barriers), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                
                # Save Outputs Detection
                if self.save_det_imgs_without_headers or self.save_det_vid_without_headers:
                    img_temp = self._draw_bbox(img, dets)
                    if self.save_det_imgs_without_headers:
                        cv2.imwrite(str(output_fopath_det_imgs_without_headers / f'img_{frame}.jpg'), img_temp)
                    if self.save_det_vid_without_headers:
                        vid_writer_det_without_headers.write(img_temp)

                if self.save_det_imgs_with_headers or self.save_det_vid_with_headers:
                    img_temp = self._draw_bbox(img, dets, place_headers=True)
                    if self.save_det_imgs_with_headers:
                        cv2.imwrite(str(output_fopath_det_imgs_with_headers / f'img_{frame}.jpg'), img_temp)
                    if self.save_det_vid_with_headers:
                        vid_writer_det_with_headers.write(img_temp)

                # Track and save Tracking Outputs
                if self.tracking_activated:
                    self.tracker.update(dets, frame)

                    if self.save_track_imgs_without_headers or self.save_track_vid_without_headers:
                        img_temp = self._draw_bbox_tracking(img, self.tracker.mot_labels[frame], color_handler=self.tracker.colors)
                        if self.save_track_imgs_without_headers:
                            cv2.imwrite(str(output_fopath_track_imgs_without_headers / f'img_{frame}.jpg'), img_temp)
                        if self.save_track_vid_without_headers:
                            vid_writer_track_without_headers.write(img_temp)
                        if self.save_track_imgs_lines or self.save_track_vid_lines:
                            _img_track_lines = img_temp

                    if self.save_track_imgs_with_headers or self.save_track_vid_with_headers:
                        img_temp = self._draw_bbox_tracking(img, self.tracker.mot_labels[frame], color_handler=self.tracker.colors, place_headers=True)
                        if self.save_track_imgs_with_headers:
                            cv2.imwrite(str(output_fopath_track_imgs_with_headers / f'img_{frame}.jpg'), img_temp)
                        if self.save_track_vid_with_headers:
                            vid_writer_track_with_headers.write(img_temp)

                    if self.save_track_imgs_lines or self.save_track_vid_lines:
                        if not self.save_track_imgs_without_headers:
                            _img_track_lines = self._draw_bbox_tracking(img, self.tracker.mot_labels[frame], color_handler=self.tracker.colors)
                        img_temp = self._draw_bbox_and_line_tracking(_img_track_lines, self.tracker.vehicles_on_the_frame[frame], self.tracker.tracked_vehicles_trajectory, color_handler=self.tracker.colors)
                        if self.save_track_imgs_lines:
                            cv2.imwrite(str(output_fopath_track_imgs_lines / f'img_{frame}.jpg'), img_temp)
                        if self.save_track_vid_lines:
                            vid_writer_track_lines.write(img_temp)

                    # Barriers Processing and Outputs
                    if this_video_barriers_activated:
                        if frame > 1: # Check for crossings
                            for bi in barriers:
                                if  not bi.start_frame <= frame <= bi.end_frame:
                                    continue
                                for veh_idx in self.tracker.vehicles_on_the_frame[frame]:
                                    if len(self.tracker.tracked_vehicles_trajectory[veh_idx]) <= 1: continue
                                    #sanity check:
                                    assert self.tracker.tracked_vehicles_trajectory[veh_idx][-1][0] == frame, 'Problemas na indexação dos frames(Tracking)(1).'
                                    assert self.tracker.tracked_vehicles_trajectory[veh_idx][-2][0] <= frame-1, 'Problemas na indexação dos frames(Tracking)(2).'
                                    pos_atual = self.tracker.tracked_vehicles_trajectory[veh_idx][-1][1:]
                                    pos_anterior = self.tracker.tracked_vehicles_trajectory[veh_idx][-2][1:]
                                    if bi.doIntersect(pos_anterior, pos_atual):
                                        bi.add_counter()
                                        bi.add_intersection_frame(frame)

                        if self.save_track_img_barriers or self.save_track_vid_barriers:
                            if not self.save_track_imgs_without_headers:
                                _img_track_lines = self._draw_bbox_tracking(img, self.tracker.mot_labels[frame], color_handler=self.tracker.colors)
                            img_temp = self._draw_bbox_and_line_tracking(_img_track_lines, self.tracker.vehicles_on_the_frame[frame], self.tracker.tracked_vehicles_trajectory, color_handler=self.tracker.colors, barriers=barriers, frame=frame)
                            if self.save_track_img_barriers:
                                cv2.imwrite(str(output_fopath_track_img_barriers / f'img_{frame}.jpg'), img_temp)
                            if self.save_track_vid_barriers:
                                vid_writer_track_barriers.write(img_temp)

            # Post-processing
            if self.save_detection_json:
                with open(str(output_fipath_dets_json), 'a') as f:
                    f.write(']')
            
            if self.save_tracking_json:
                new_line='\n'
                first_line_tracking = True
                with open(str(output_fipath_tracking_json), 'a') as f:
                    for frame, dets in self.tracker.mot_labels.items(): #x,y,w,h,idx
                        det_str = repr(dets)
                        det_table = det_str.maketrans({" ":"", "(":"[", ")":"]", new_line:""})
                        if len(dets) == 0:
                            det_str = "xxxxx([])"
                        if not first_line_tracking:
                            string = f',{{"frame":{frame},"mot":{det_str.translate(det_table)[6:-1]}}}'
                        else:
                            string = f'[{{"frame":{frame},"mot":{det_str.translate(det_table)[6:-1]}}}'
                            first_line_tracking = False    
                        f.write(f'{string}')
                    f.write(']')

            # Save Barriers Plots
            if this_video_barriers_activated:
                barrier_plots, total_crossings = self._plot_barriers_crossings_vs_frame(barriers)
                text = ''
                str_start = '['
                for barrier in barriers:
                    bname = barrier.name
                    # Save plots
                    barrier_plots[bname].savefig(str(output_fopath_barriers_plots / f'{video_name}_{bname}.png'), bbox_inches='tight')
                    # Save Infos
                    text += f'{str_start}{{"barrier_name":"{bname}","total_crossings":{total_crossings[bname]},"frame_vs_cross_dict":{json.dumps(barrier.intersection_frames)}}}'
                    str_start=','

                with open(str(output_fopath_barriers_plots / f'{video_name}_barrier_infos.json'), 'w') as f:
                    f.write(text+']')
            
            # Release det_json file writer
            output_det_json_file.write(']')
            output_det_json_file.close()

            # Release VideoWriters
            if self.save_det_vid_without_headers:
                vid_writer_det_without_headers.release()
            if self.save_det_vid_with_headers:
                vid_writer_det_with_headers.release()
            if self.save_track_vid_without_headers:
                vid_writer_track_without_headers.release()
            if self.save_track_vid_with_headers:
                vid_writer_track_with_headers.release()
            if self.save_track_vid_lines:
                vid_writer_track_lines.release()
            if self.save_track_vid_barriers:
                vid_writer_track_barriers.release()

class JupyterInterface():
    def __init__(self):
        self._widgets()
        self._set_callbacks()

    def _widgets(self):

        self.file_selector = HTML('<input type="file" id="selectedFile" style="display: none; " /> \
                    <input type="button" value="Search Input/ Output Path ..." \
                    onclick="document.getElementById(\'selectedFile\').click();" />')

        self.input_path_widget = widgets.Text(
                                              value='',
                                              placeholder='Input Path...',
                                              description='Input Path:',
                                              disabled=False,
                                              layout=widgets.Layout(width='100%')
                                              )

        self.output_fopath_widget = widgets.Text(
                                              value='',
                                              placeholder='Output Folder Path...',
                                              description='Output Path:',
                                              disabled=False,
                                              layout=widgets.Layout(width='100%')
                                                )

        det0 = configs.detectors_list[0]
        tracker0 = configs.trackers_list[0]
        det_params = configs.detector_default_params[det0]

        tracker_params = configs.tracker_default_params[tracker0][det0]
        
        self.det_selector = widgets.Dropdown(
                                            options=configs.detectors_list,
                                            value=det0,
                                            description='Detector:',
                                            disabled=False,
                                        )
        self.tracker_selector = widgets.Dropdown(
                                    options=configs.trackers_list,
                                    value=tracker0,
                                    description='Tracker:',
                                    disabled=False,
                                )
        
        self.device_selector = widgets.Dropdown(
                                    options=list(configs.default_devices_list.keys()),
                                    value='cpu',
                                    description='Device',
                                    disabled=False,
                                )
        self.parameters_box = widgets.Textarea(
                                                value='',
                                                placeholder='Generated python code.',
                                                description='Python Code:',
                                                disabled=False,
                                                layout=widgets.Layout(width='90%', height='240px')
                                            )
        
        
        self.msg0 = '<br> <p style="font-size:15px"> Como salvar os resultados?: </p><br>'
        self.checkbox_msgs = ['detection_json',
                         'tracking_json',
                         'det_imgs_without_headers',
                         'det_imgs_with_headers', 
                         'track_imgs_without_headers',
                         'track_imgs_with_headers',
                         'track_imgs_lines',
                         'track_img_barriers',
                         'det_vid_without_headers',
                         'det_vid_with_headers',
                         'track_vid_without_headers',
                         'track_vid_with_headers',
                         'track_vid_lines',
                         'track_vid_barriers']
        make = lambda x: widgets.Checkbox(value=True,description=x,disabled=False,indent=True)        
        self.checkbox_widgets = [make(i) for i in self.checkbox_msgs]
        hbox1 = widgets.HBox(self.checkbox_widgets[:3])
        hbox2 = widgets.HBox(self.checkbox_widgets[3:6])
        hbox3 = widgets.HBox(self.checkbox_widgets[6:9])
        hbox4 = widgets.HBox(self.checkbox_widgets[9:12])
        hbox5 = widgets.HBox(self.checkbox_widgets[12:15])
        self.checkbox_vbox = widgets.VBox([hbox1, hbox2, hbox3, hbox4, hbox5])
        self._callback_det_or_tracker_dropdown_onchange(None)


    def _set_callbacks(self):
        self.det_selector.on_trait_change(self._callback_det_or_tracker_dropdown_onchange)
        self.tracker_selector.on_trait_change(self._callback_det_or_tracker_dropdown_onchange)
        self.device_selector.on_trait_change(self._callback_det_or_tracker_dropdown_onchange)
        self.input_path_widget.on_trait_change(self._callback_det_or_tracker_dropdown_onchange)
        self.output_fopath_widget.on_trait_change(self._callback_det_or_tracker_dropdown_onchange)
        for i in self.checkbox_widgets:
            i.on_trait_change(self._callback_det_or_tracker_dropdown_onchange)


    def _callback_det_or_tracker_dropdown_onchange(self, val):
        det = self.det_selector.value
        trk = self.tracker_selector.value

        if self.input_path_widget.value == '':
            input_path = ''
        else:
            input_ = pathlib.Path(self.input_path_widget.value)
            input_path = f'input_path = \'{str(input_)}\''

        if self.output_fopath_widget.value == '':
            output_fopath = ''
        else:
            output = pathlib.Path(self.output_fopath_widget.value) / f'{input_.stem}_processed_{det}-{trk}'
            output_fopath = f'output_path = \'{str(output)}\''

        device = self.device_selector.value
        det_params = configs.detector_default_params[det]
        det_params.update({'device' : configs.default_devices_list[device]})
        trk_params = configs.tracker_default_params[trk][det]

        temp = ''
        for k,v in trk_params.items():
            this = f"'{k}' : {v},\n"
            temp += f'{this:>40}'
        trk_params = temp[:-2]

        temp = ''
        for k,v in det_params.items():
            if isinstance(v, str):
                this = f"'{k}' : '{v}',\n"    
            else:
                this = f"'{k}' : {v},\n"
            temp += f'{this:>40}'            
        det_params = temp[:-2]
        
        temp = ''
        for w, item in zip(self.checkbox_widgets, self.checkbox_msgs):
            this = f"'save_{item}' : {str(w.value)},\n"
            temp += f'{this:>50}'
        save = temp[:-2]
        
        msg = f'''import traffic_vehicle_counter as tvc
CountBarrier = tvc.CountBarrier

input_path = {input_path}

output_path = {output_fopath}

save = {{
{save}
}}

detector_params = {{
{det_params}
}}

tracker_params = {{
{trk_params}
}}

## DEFINA AS BARREIRAS AQUI!
#barriers = 

detector = tvc.Detector(detector='{det}', params=detector_params)

tracker = tvc.TrackerSort(**tracker_params)

a = tvc.Analysis(
            input_path   = input_path,
            output_path  = output_path,
            detector     = detector,
            tracker      = tracker,
            barriers     = barriers,
            **save
)

## REMOVA O '#' PARA PERMITIR QUE O PROGRAMA RODE!
#a.start()'''

        self.parameters_box.value = msg

if __name__ == '__main__':
    def example1(device='cpu', det='yv5_S'):
        save_outputs = {'save_detection_json'             : True, 
                        'save_tracking_json'              : True, 
                        'save_det_imgs_without_headers'   : True, 
                        'save_det_imgs_with_headers'      : True,
                        'save_track_imgs_without_headers' : True, 
                        'save_track_imgs_with_headers'    : True,
                        'save_track_imgs_lines'           : True, 
                        'save_track_img_barriers'         : True,
                        'save_det_vid_without_headers'    : True, 
                        'save_det_vid_with_headers'       : True, 
                        'save_track_vid_without_headers'  : True, 
                        'save_track_vid_with_headers'     : True, 
                        'save_track_vid_lines'            : True, 
                        'save_track_vid_barriers'         : True 
                        }

        detector = Detector(detector=det, params={'conf_thres': 0.3, 'device': device})
        tracker = TrackerSort(**detector.get_sort_tracker_optimal_parameters())
        
        b1 = CountBarrier((100,100), (540,540), 1, 15, 'b0')
        b2 = CountBarrier((400,400), (500,800), 5, 20, 'b1')
        b3 = CountBarrier((100,400), (500,800), 5, 20, 'b1')

        #barriers = {'movie_name_1': [barriers_objects_1], 'movie_name_2': [barriers_objects_2], ...}
        barriers = {"M0101": [b1, b2],
                    "M0201":[b3]}

        a=Analysis(input_path   ='/home/yuri/Desktop/videos/',
                   output_path  =f'/home/yuri/Desktop/videos_processed_{det}/',
                   detector     = detector,
                   tracker      = tracker,
                   barriers     = barriers,
                   **save_outputs)
        a.start()

    def example2(device='cpu', det='yv5_S'):
        save_outputs = {'save_detection_json'             : True, 
                        'save_tracking_json'              : True, 
                        'save_det_imgs_without_headers'   : True, 
                        'save_det_imgs_with_headers'      : True,
                        'save_track_imgs_without_headers' : True, 
                        'save_track_imgs_with_headers'    : True,
                        'save_track_imgs_lines'           : True, 
                        'save_track_img_barriers'         : True,
                        'save_det_vid_without_headers'    : True, 
                        'save_det_vid_with_headers'       : True, 
                        'save_track_vid_without_headers'  : True, 
                        'save_track_vid_with_headers'     : True, 
                        'save_track_vid_lines'            : True, 
                        'save_track_vid_barriers'         : True 
                        }

        input_path  = '/home/yuri/Desktop/videos_mp4/20210417_205425.mp4'
        output_path = f'/home/yuri/Desktop/video_20210417_205425_processed_{det}/'
        detector    = Detector(detector=det, params={'device':device})
        tracker     = TrackerSort(**detector.get_sort_tracker_optimal_parameters())

        
        b0       = CountBarrier(p1=(441, 938), q1=(1608, 885), start_frame=80, end_frame=790, name="b0")
        b1       = CountBarrier(p1=(1452, 688), q1=(1864, 624), start_frame=80, end_frame=790, name="b1")
        b2       = CountBarrier(p1=(1560, 417), q1=(1845, 544), start_frame=80, end_frame=790, name="b2")
        barriers = {"20210417_205425": [b0, b1, b2]} 
        # P.S: barriers coords obtained via convenience_define_barriers.py

        a=Analysis(input_path   =input_path,
                   output_path  =output_path,
                   detector     =detector,
                   tracker      =tracker,
                   barriers     =barriers,
                   **save_outputs)
        a.start()

    def example3(device='cpu', det='yv5_X'):
        save_outputs = {'save_detection_json'             : True, 
                        'save_tracking_json'              : True, 
                        'save_det_imgs_without_headers'   : True, 
                        'save_det_imgs_with_headers'      : True,
                        'save_track_imgs_without_headers' : True, 
                        'save_track_imgs_with_headers'    : True,
                        'save_track_imgs_lines'           : True, 
                        'save_track_img_barriers'         : True,
                        'save_det_vid_without_headers'    : True, 
                        'save_det_vid_with_headers'       : True, 
                        'save_track_vid_without_headers'  : True, 
                        'save_track_vid_with_headers'     : True, 
                        'save_track_vid_lines'            : True, 
                        'save_track_vid_barriers'         : True 
                        }

        input_path  = '/home/yuri/Desktop/Videos/videos_cier'
        output_path = f'/home/yuri/Desktop/Videos_Processed/videos_cier_processed_{det}/'
        detector    = Detector(detector=det, params={'device':device})
        tracker     = TrackerSort(**detector.get_sort_tracker_optimal_parameters())

        dj_b0=CountBarrier(p1=(309, 168), q1=(357, 717), start_frame=1, end_frame=1199, name="b0")
        dj_b1=CountBarrier(p1=(1305, 115), q1=(1495, 710), start_frame=1, end_frame=1199, name="b1")
        
        v_cel1=CountBarrier(p1=(288, 813), q1=(1310, 672), start_frame=52, end_frame=6588, name="b0")
        v_cel2=CountBarrier(p1=(979, 482), q1=(1528, 412), start_frame=52, end_frame=6588, name="b1")

        barriers = {"dronejose": [dj_b0, dj_b1],
                    "video_cel1": [v_cel1, v_cel2]}
        # P.S: barriers coords obtained via convenience_define_barriers.py

        a=Analysis(input_path   =input_path,
                   output_path  =output_path,
                   detector     =detector,
                   tracker      =tracker,
                   barriers     =barriers,
                   **save_outputs)
        a.start()

    example1(det='yv3_tiny')
    #example2(det='yv3_tiny')
    #example3()
    pass
# %%
