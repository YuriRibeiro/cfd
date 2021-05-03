#%%
import cv2
import pathlib
import configs
import numpy as np
from IPython.display import HTML, display, Javascript
import ipywidgets as widgets


class JupyterInterface:
    def __init__(self):
        # Call Widgets and Callbacks Constructor
        self._widgets()
        self._callbacks()


    def _widgets(self):
        self.file_selector = HTML('<input type="file" id="selectedFile" style="display: none; " /> \
                            <input type="button" value="Search Input Path ..." \
                            onclick="document.getElementById(\'selectedFile\').click();" />')
        self.input_path_widget = widgets.Text(
                                        value='',
                                        placeholder='Input Path...',
                                        description='Input Path:',
                                        disabled=False,
                                        layout=widgets.Layout(width='100%')
                                        )
        self.input_path_button = widgets.Button(
                                        description='Run via OpenCV!',
                                        disabled=False,
                                        button_style='success',
                                        tooltip='Start to define the barriers.',
                                        icon='',
                                        layout=widgets.Layout(width='30%', height='40px')
                                        )
        self.print_barrier_results_text = widgets.Textarea(
                                                        value='',
                                                        placeholder='Barrier Info Results',
                                                        description='Status:',
                                                        disabled=False,
                                                        layout=widgets.Layout(width='90%', height='120px')
                                                    )
        
    
    def _callbacks(self):
        # Set Up All Types of Callbacks
        self._callbacks_on_click()

    def _callbacks_on_click(self):
        # Set up _on_click Callbacks
        self.input_path_button.on_click(self._input_path_button_callback)
    
    def _input_path_button_callback(self, val):
        value = self.input_path_widget.value
        try:
            self.ih = ConvenienceBarriers(value, jupyter=self)
            self.ih.convenience_define_barriers()
        except Exception as e:
            self.print_barrier_results_text.value= str(e)




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
                    raise Exception(f'Input folder has incompatible file types: {str(file)}')
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

    def _vid_get_frame(self, frame, mode='cap_properties'):
        if frame > self._vid_last_frame:
            raise Exception(f'Video required frame ({frame}) greater than total frames ({self._vid_last_frame}).')
        elif frame < 1:
            raise Exception(f'Required frame number is lower than 1.')
        if mode == 'exactly':
            return self._vid_get_frame_exactly(frame)
        elif mode == 'cap_properties':
            return self._vid_get_frame_cap_props(frame)
        else:
            raise Exception(f'Unknow mode: {mode}.')
    
    def _vid_get_frame_cap_props(self, frame):
        cap = self._vid_get_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1) #cap_prop_pos_frames is zero indexed
        status, self._vid_img_atual = cap.read()
        if not status: raise Exception(f'Problems reading frame {frame}.')
        self._vid_frame_atual = frame
        return self._vid_img_atual

    def _vid_get_frame_exactly(self, frame):
        if frame < self._vid_frame_atual:
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

    def _vid_get_frame_infos(self, mode='cap_properties'):
        if self.input_type == 'vid':
            if mode == 'exactly':
                self._vid_get_frames_infos_exactly()
            elif mode == 'cap_properties':
                self._vid_get_frames_infos_cap_props()
            else:
                raise Exception(f'Unknow mode: {mode}.')

    def _vid_get_frames_infos_cap_props(self):
        cap = self._vid_get_cap()
        self._vid_first_frame = 1
        self._vid_last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._vid_qtt_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def _vid_get_frames_infos_exactly(self):
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

        return int(first_frame), int(last_frame), int(qtt_frames)
    
    def get_img_paths(self):
        return self.dir_img_paths
    
    def get_vid_frame(self, frame):
        return self._vid_get_frame(frame)


class ConvenienceBarriers(InputHandler):
    def __init__(self, input_path, get_file_frame_number=None, jupyter=None):
        super().__init__(input_path, get_file_frame_number)
        self.jupyter = jupyter
        
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
        window_w = 800
        window_h = 600
        jupyter = self.jupyter

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
            nonlocal color, thickness, jupyter
            h, w = img.shape[:2]

            radius = int(h * 0.02)
            # Add Barriers Handler
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x,y), radius, color, thickness)

                if last_x == None and last_y == None:
                    last_x = x
                    last_y = y  
                else:
                    barrier_names = barriers.keys()
                    if not (last_x == x and last_y == y):
                        for num in range(len(barriers)+1):
                            if num not in barrier_names: break

                        barriers[num] = [(last_x, last_y), (x, y), (1, qtt_of_frames)] 
                        last_x = None
                        last_y = None

            # Select Barrier Handler
            if event == cv2.EVENT_MBUTTONDOWN:
                lower_r = 1E10
                selected_barrier_num = -1

                if not len(barriers)==0:
                    for barrier_num in barriers:
                        (x1,y1), (x2,y2), (start_frame, end_frame) = barriers[barrier_num]
                        if start_frame > frame > end_frame:
                            continue # Select barriers only at the current frame.
                        
                        # Search for the nearest barrier midpoint
                        midp_x, midp_y = (x1+x2)/2, (y1+y2)/2
                        r = np.linalg.norm([x - midp_x, y-midp_y], 2)
                        if r < lower_r:
                            lower_r = r
                            selected_barrier_num = barrier_num

                    if selected_barrier_num in current_selection:
                        # Deselect
                        current_selection.remove(selected_barrier_num)
                    else:
                        # Select    
                        current_selection.add(selected_barrier_num)
                        if not isinstance(jupyter, JupyterInterface):                            
                            print('Última Barreira Selecionada: ', selected_barrier_num)
                    
                    #Set Trackbar Positions
                    if len(current_selection) == 1:
                        start, end = barriers[selected_barrier_num][2]
                    else:
                        start, end = 1,1
                    cv2.setTrackbarPos("Barrier Start Frame", "Define Barriers", start)
                    cv2.setTrackbarPos("Barrier End Frame", "Define Barriers", end)

            # Draw Functions            
            show(img)
            return None

        img = None
        cv2.namedWindow("Define Barriers", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_w, window_h)

        cv2.createTrackbar("Frame","Define Barriers",1, qtt_of_frames,nothing)
        cv2.setTrackbarMin("Frame", "Define Barriers", 1)

        cv2.createTrackbar("Barrier Start Frame","Define Barriers",1,qtt_of_frames,nothing)
        cv2.setTrackbarMin("Barrier Start Frame", "Define Barriers", 1)

        cv2.createTrackbar("Barrier End Frame", "Define Barriers",1,qtt_of_frames,nothing)
        cv2.setTrackbarMin("Barrier End Frame", "Define Barriers", 1)

        cv2.createTrackbar("//----------------------- INFOS -----------------------//", "Define Barriers", 1,1 , nothing)        
        cv2.createTrackbar("L-Mouse: Draw // Sroll Mouse Button : Select Nearest (Midpoint of the) Barrirer // f: Save Selected Barrier Frame Range //", "Define Barriers", 1, 1, nothing)        
        cv2.createTrackbar("Space: Next Frame // d: Delete Selected Barrier // Esc: Quit and Print Results //", "Define Barriers", 1, 1, nothing)


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

            if isinstance(jupyter, JupyterInterface):
                print_cs = [f'b{i}' for i in current_selection]
                jupyter.print_barrier_results_text.value = f'Barreiras Selecionadas: {print_cs}'    
            
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

            elif k == 100 and len(current_selection)>=1: # d
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
        list_of_barriers = [(name,infos) for (name,infos) in barriers.items()]
        # Order in ascending start_frame value
        list_of_barriers.sort(key = lambda x: x[1][2][0])

        # Print String
        string = f'==== LISTA DE BARREIRAS ====\nbarriers={{"{self.input_path.stem}" : ['
        if not len(list_of_barriers) == 0:
            for (name, barrier_infos) in list_of_barriers:
                p1, q1, (start_frame, end_frame) = barrier_infos
                string+=(f'CountBarrier(p1={p1}, q1={q1}, start_frame={start_frame}, end_frame={end_frame}, name="b{name}"),\n')
            string = string[:-2] + ']}'
        else:
            string = string + ']}'
        if isinstance(self.jupyter, JupyterInterface):                            
            self.jupyter.print_barrier_results_text.value = string
            return
        else:
            print(string)
        return list_of_barriers

if __name__ == '__main__':
    pass
    # Folder With Images numbered like img001.jpg, img002.jpg, ...
    #a = ConvenienceBarriers('/home/yuri/Desktop/videos/M0101', get_file_frame_number=lambda x: int(x[3:]))
    #a.convenience_define_barriers()

    # A video file (.mp4)
    #a = ConvenienceBarriers('/home/yuri/Desktop/Datasets/20210417_205724.mp4')
    #a.convenience_define_barriers()

    ## Known Problems:
    # 1) Images are not resized to the display_window size before processing. So, high resolution images will
    # in high resolution (slowly) to be resized at the display. Solution: Resize images to the display size
    # before the drawing process. For video_caps, use cv2.CAP_PROP_XX to set the video width and height.

    

# %%
