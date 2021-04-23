#%%
import cv2
import pathlib
import configs
import numpy as np

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

class ConvenienceBarriers(InputHandler):
    def __init__(self, input_path, get_file_frame_number=None):
        super().__init__(input_path, get_file_frame_number)
        
    def convenience_define_barriers(self):
        """
        input_path : should be a video, or a folder containing more than one image with ordered names.

        Facilitador para a tarefa de definir barreiras em um vídeo.
        
        1) Detectar os mouses clicks e selecionar as barreiras.
        2) Botão Draw para desenhar a barreira.
        3) Botão select para entrar no modo de seleção de barreira.
        4) Quando uma barreira for selecionada, mostrar opções de exclusão,
            set_name, set_first_frame, set_end_frame, edit_coords.
        """
        window_name = 'Define Barriers'
        show = lambda img: cv2.imshow(window_name, img)
        nothing = lambda x: None

        color  = (0,0,255)
        selected_color = (125,0,125)
        thickness = 2
        
        last_x = None
        last_y = None
        barriers = {}
        current_selection = set([])

        if self.input_type == 'dir':
            self.video_frames_paths = self.get_img_paths() #dict:: frames:path

        first_frame_num, last_frame_num, qtt_of_frames = self.get_frame_infos()
        

        def click_callback(event, x, y, flags, params):
            nonlocal img, last_x, last_y, barriers, current_selection
            nonlocal color, thickness
            h, w = img.shape[:2]

            radius = int(h * 0.02)
            # Add Barriers Handler
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x,y), radius, color, thickness)

                if last_x == None and last_y == None:
                    last_x = x
                    last_y = y  
                else:
                    barriers[len(barriers)+1] = [(last_x, last_y), (x, y), (1, qtt_of_frames)] 
                    last_x = None
                    last_y = None

            # Select Barrier Handler
            if event == cv2.EVENT_MBUTTONDOWN:
                lower_r = 1E10
                selected_barrier_num = -1
                
                for barrier_num in barriers:
                    x1,y1 = barriers[barrier_num][0]
                    x2,y2 = barriers[barrier_num][1]

                    midp_x, midp_y = (x1+x2)/2, (y1+y2)/2

                    r = np.linalg.norm([x - midp_x, y-midp_y], 2)
                    #print(r, lower_r)
                    if r < lower_r:
                        lower_r = r
                        selected_barrier_num = barrier_num

                x1,y1 = barriers[selected_barrier_num][0]
                x2,y2 = barriers[selected_barrier_num][1]

                if selected_barrier_num in current_selection:
                    # Deselect
                    current_selection.remove(selected_barrier_num)
                else:
                    # Select    
                    current_selection.add(selected_barrier_num)
                
                #Set Trackbar Positions
                if len(current_selection) == 1:
                    start, end = barriers[selected_barrier_num][2]
                else:
                    start, end = 1,1
                cv2.setTrackbarPos("Barrier Start Frame", "Define Barriers", start)
                cv2.setTrackbarPos("Barrier End Frame", "Define Barriers", end)
                #print("Selected barrier number: ", selected_barrier_num)

            # Draw Functions            
            show(img)
            return None

        img = None
        cv2.namedWindow("Define Barriers", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800,600)

        cv2.createTrackbar("Frame","Define Barriers",1, qtt_of_frames,nothing)
        cv2.setTrackbarMin("Frame", "Define Barriers", 1)

        cv2.createTrackbar("Barrier Start Frame","Define Barriers",1,qtt_of_frames,nothing)
        cv2.setTrackbarMin("Barrier Start Frame", "Define Barriers", 1)

        cv2.createTrackbar("Barrier End Frame", "Define Barriers",1,qtt_of_frames,nothing)
        cv2.setTrackbarMin("Barrier End Frame", "Define Barriers", 1)

        cv2.setMouseCallback("Define Barriers", click_callback)

        frame = 0
        atualizar_frame = False
        while True:
            past_frame = frame

            # Get Trackbar Frame
            frame = cv2.getTrackbarPos("Frame", "Define Barriers")
            if frame == 0: frame = 1

            #Take Care of Start and End Frame Trackbars
            start_frame_pos = cv2.getTrackbarPos("Barrier Start Frame", "Define Barriers")
            end_frame_pos = cv2.getTrackbarPos("Barrier End Frame", "Define Barriers")

            if  start_frame_pos > end_frame_pos:
                cv2.setTrackbarPos("Barrier End Frame", "Define Barriers", start_frame_pos)

            #Update Image
            if frame != past_frame or atualizar_frame:
                if self.input_type == 'vid':
                    img = self.get_vid_frame(frame)
                else:
                    frame_path = str(self.video_frames_paths[frame])
                    img = cv2.imread(frame_path)
                atualizar_frame = False
            
            # Draw Barriers and Deselect Barriers out of Vision
            for barrier_num in barriers:
                (x1,y1), (x2,y2), (start_frame, end_frame) = barriers[barrier_num]
                if not start_frame <= frame <= end_frame:
                    if barrier_num in current_selection:
                        current_selection.remove(barrier_num)
                    continue
                if not barrier_num in current_selection:
                    cv2.line(img, (x1,y1), (x2,y2), color, thickness)
                else:
                    cv2.line(img, (x1,y1), (x2,y2), selected_color, thickness)
            
            show(img)
            
            
            # Commands Handler
            k = cv2.waitKey(60)

            if k == 27: # ESC
                cv2.destroyWindow("Define Barriers")
                break

            elif k == 32: # Space
                new_frame = frame+1 if frame < qtt_of_frames else frame
                cv2.setTrackbarPos("Frame","Define Barriers", new_frame)

            elif k == 100: # d
                #(d) Delete current selection of barriers
                deleted = []
                for barrier_num in current_selection:
                    barriers.pop(barrier_num)
                    deleted.append(barrier_num)

                for barrier_num in deleted:
                    current_selection.remove(barrier_num)
                
                atualizar_frame = True
                #print(f"Barriers {deleted} deleted.")

            elif k == 102: 
            #(f) Set frame limits to the current barriers selection.
                for barrier_num in current_selection:
                    start_frame_Trackbar = cv2.getTrackbarPos("Barrier Start Frame", "Define Barriers")
                    end_frame_Trackbar = cv2.getTrackbarPos("Barrier End Frame", "Define Barriers")
                    barriers[barrier_num][2] = (start_frame_Trackbar, end_frame_Trackbar)
                    
                current_selection.clear()
                cv2.setTrackbarPos("Barrier Start Frame", "Define Barriers", 1)
                cv2.setTrackbarPos("Barrier End Frame", "Define Barriers", 1)
                atualizar_frame = True

        # Now, summarize barrier informations
        list_of_barriers = [infos for infos in barriers.values()]
        # Order in ascending start_frame value
        list_of_barriers.sort(key = lambda x: x[2][0])
        
        for idx, barrier_infos in enumerate(list_of_barriers):
            p1, q1, (start_frame, end_frame) = barrier_infos
            name = f"b{idx}"
            print(f'\n{name}=CountBarrier(p1={p1}, q1={q1}, start_frame={start_frame}, end_frame={end_frame}, name="{name}")')
        return list_of_barriers



if __name__ == '__main__':
    pass
    #a = InputHandler('/home/yuri/Desktop/videos/M0101', get_file_frame_number=lambda x: int(x[3:]))
    #(first,last, qtt) = a.get_frame_infos()
    #imgs_paths = a.get_img_paths() #frame: path

    #a = ConvenienceBarriers('/home/yuri/Desktop/videos/M0101', get_file_frame_number=lambda x: int(x[3:]))
    #a.convenience_define_barriers()

    #a = ConvenienceBarriers('/home/yuri/Desktop/videos_mp4/20210417_205425.mp4')
    #a.convenience_define_barriers()
# %%
