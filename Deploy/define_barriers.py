#%%
import queue
import cv2
import numpy as np
import pathlib
import configs
from IPython.display import HTML, display, Javascript
import ipywidgets as widgets



class Widgets():
    def __init__(self):

        self.file_selector = HTML('<input type="file" id="selectedFile" style="display: none;" /> \
                                    <input type="button" value="Copy Input/ Output Path ..." \
                                    onclick="document.getElementById(\'selectedFile\').click();" />')
        self.input_path_widget = widgets.Text(value='Input Path...',
                                              placeholder='Input Path',
                                              description='Input:',
                                              disabled=False)
        self.output_path_widget = widgets.Text(value='Output Path...',
                                              placeholder='Output Path',
                                              description='Output:',
                                              disabled=False)
        
        

class InputHandler():
    def __init__(self, input_path, get_file_frame_number=None):
        self.input_path = pathlib.Path(input_path)
        self.input_type = self._check_input_type()

        # Dir Vars
        self.get_file_frame_number = get_file_frame_number
        if self.get_file_frame_number == None: self.get_file_frame_number = lambda x: int(x[3:])
        if self.input_type == 'dir':
            self.dir_img_paths = self._dir_collect_paths()#frame:path
        
        # Vid vars
        if self.input_type == 'vid':
            self._vid_first_frame = None
            self._vid_last_frame = None
            self._vid_qtt_frames = None
            self._vid_frame_atual = 0
            self._vid_get_frame_infos()
            self._vid_cap_atual = self._vid_get_cap()
            self._vid_img_atual = None

    def _check_input_type(self):
        if self.input_path.is_dir():
            for file in self.input_path.glob('*'):
                if file.is_dir():
                    raise Exception(f'Folder is not allowed inside input path: {str(file)}. The folder should have only images.')
                if not file.suffix in configs.compatible_image_types:
                    raise Exception(f'Input folder has file with incompatible type: {str(file)}')
            return 'dir'
        elif self.input_path.is_file():
            if not self.input_path.suffix in configs.compatible_video_types:
                raise Exception(f'Video type not allowed {str(self.input_path.suffix)}. Allowed types: {configs.compatible_video_types}.')
            return 'vid'
        else:
            raise Exception(f'Unrecognized input path: {str(self.input_path)}')
    
    def _vid_get_cap(self):
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise Exception("Error opening video")
        return cap

    def _vid_get_frame(self, frame):
        if frame > self._vid_last_frame:
            raise Exception(f'Video required frame ({frame}) greater than total frames ({self._vid_last_frame}).')
        elif frame < 1:
            raise Exception(f'Required frame number is lower than 1.')
        elif frame < self._vid_frame_atual:
            self._vid_cap_atual = self._vid_get_cap()
            self._vid_frame_atual = 0
        elif frame == self._vid_frame_atual:
            return self._vid_img_atual

        while self._vid_frame_atual < frame:
            status, self._vid_img_atual = self._vid_cap_atual.read()
            self._vid_frame_atual += 1
        return self._vid_img_atual

    def _dir_collect_paths(self):
        if not self.input_type == 'dir': return None
        dir_img_paths = {}
        for file in self.input_path.glob('*'):
            frame_num = self.get_file_frame_number(str(file.stem))
            dir_img_paths[frame_num] = file
        return dir_img_paths

    def _vid_get_frame_infos(self):
        if self.input_type == 'vid':
            cap = self._vid_get_cap()
            frame = 0
            # Count frames exactly
            while True:
                status, _ = cap.read()
                if not status: break
                frame += 1
            self._vid_first_frame = 1
            self._vid_last_frame = frame
            self._vid_qtt_frames = frame

    def get_frame_infos(self):
        if self.input_type == 'vid':
            first_frame = self._vid_first_frame
            last_frame = self._vid_last_frame
            qtt_frames = self._vid_qtt_frames

        elif self.input_type == 'dir':
            frame_nums = self.dir_img_paths.keys()
            first_frame = min(frame_nums)
            last_frame = max(frame_nums)
            qtt_frames = len(self.dir_img_paths)

        return first_frame, last_frame, qtt_frames
    
    def get_img_paths(self):
        return self.dir_img_paths
    
    def get_vid_frame(self, frame):
        return self._vid_get_frame(frame)

class Convenience():
    def __init__(self):
        max_fifo_size = 2
        self.click_coords_fifo = queue.Queue(maxsize=max_fifo_size)
        self.click_abs_coords = np.zeros((max_fifo_size,2))
        self.current_img = None
        self.current_img_gain = np.zeros((2,))
        self.BLUE = (255,0,0)
    
    def _to_abs_coords(self):
        self.click_abs_coords = np.array(list(self.click_coords_fifo))
        self.click_abs_coords[:,0] *= self.current_img_gain[0]
        self.click_abs_coords[:,1] *= self.current_img_gain[1]

    def _check_fifo_to_draw(self):
        ok = True
        if self.coords_fifo.qsize() < 2: ok = False
        return ok

    def get_abs_coords(self):
        temp = np.around(self.click_abs_coords)
        cac = self.click_abs_coords
        x1, y1 = int(cac[0,0]), int(cac[0,1])
        x2, y2 = int(cac[1,0]), int(cac[1,1])
        return [(x1,y1), (x2,y2)]
    
    def _draw(self, polygon):
        if not self._check_fifo_to_draw():
             print('Problems drawing the rectangle.')
             return
        else:
            pt1, pt2 = self.get_abs_coords()
        if polygon == 'rect':
            cv2.rectangle(self.current_img, pt1, pt2, self.BLUE, 2, cv2.LINEA_AA)
        elif polygon == 'circle':
            cv2.circle(self.current_img, pt1, 5, 1, cv2.LINE_AA)
            cv2.circle(self.current_img, pt2, 5, 1, cv2.LINE_AA)
    


