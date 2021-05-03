#%%
import queue
import cv2
import numpy as np
import pathlib
import configs
from IPython.display import HTML, display, Javascript
import ipywidgets as widgets
import base64


class Barriers():
    def __init__(self, p1:'tuple(x,y)', p2:'tuple(x,y)', start_frame=0, end_frame=0, name=''):
        self.p1 = self._round_to_int(p1)
        self.p2 = self._round_to_int(p2)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.name = name
    def _round_to_int(self,pi):
        return int(np.around(pi[0])),int(np.around(pi[1]))
    
class Widgets():
    def __init__(self):
        self.x1, y1, x2, y2 =0.0, 0.0, 0.0, 0.0
        self.w = 0
        self.h = 0
        self.disp_w = 600
        self.disp_h = 400

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
                                                description='Run!',
                                                disabled=False,
                                                button_style='success',
                                                tooltip='Start to define the barriers.',
                                                icon='',
                                                layout=widgets.Layout(width='30%', height='40px')
                                                )
        self.input_path_button.on_click(self._input_path_button_callback)

        self.status_text_box_widget = widgets.HTML(
                                                value='',
                                                placeholder='',
                                                description='Status: ',
                                                disabled=False,
                                                layout=widgets.Layout(width='100%')
                                                )
        
        self.frame_atual = f'''
                            <div>
                            <img src="" style="width: {self.disp_w}px; height: {self.disp_h}px;" alt="Image" id="def_count_barriers" onclick="coords()">
                            </div>
                            <p id="coords_clicked"> x1: 0; y1: 0</p>
                            <p id="coords_clicked2"> x2: 0; y2: 0</p>

                            <script type="text/Javascript">
                                var kernel = IPython.notebook.kernel;
                                var x=0;
                                var y=0;
                                function coords(){{
                                    document.getElementById("coords_clicked").innerHTML = "x1: "+x+"; y1: "+y;
                                    kernel.execute("widgets.x1="+x);
                                    kernel.execute("widgets.y1="+y);
                                    var rect = document.getElementById('def_count_barriers').getBoundingClientRect();
                                    x = event.pageX - rect.left;
                                    y = event.pageY - rect.top;
                                    document.getElementById("coords_clicked2").innerHTML = "x2: "+x+"; y2: "+y;
                                    kernel.execute("widgets.x2="+x);
                                    kernel.execute("widgets.y2="+y);
                                    kernel.execute("update_coords()")
                                }}
                            </script>
                            '''

        self.draw_current_coords = widgets.Button(description='Draw Circle Around Current Coords',
                                                disabled=False,
                                                button_style='warning',
                                                tooltip='',
                                                icon='',
                                                layout=widgets.Layout(width='30%')
                                                )
        self.draw_current_coords.on_click(self._draw_current_coords_callback)

        self.clear_current_coords = widgets.Button(description='Clear Current Coords Drawn Circles',
                                                disabled=False,
                                                button_style='',
                                                tooltip='',
                                                icon='',
                                                layout=widgets.Layout(width='30%')
                                                )
        self.clear_current_coords.on_click(self._clear_current_coords_drawn_callback)

        self.add_barrier_button = widgets.Button(description='Add Counting Barrier.',
                                                disabled=False,
                                                button_style='success',
                                                tooltip='',
                                                icon='',
                                                layout=widgets.Layout(width='30%')
                                                )
        self.add_barrier_button.on_click(self._add_barrier_button_callback)

        self.select_barrier_dropdown = widgets.Dropdown(
                                                        options=[''],
                                                        value='',
                                                        description='Barrier:',
                                                        disabled=False,
                                                    )
        self.select_barrier_dropdown.on_trait_change(self._update_barrier_control_parameters)


        self.frames_barrier_slider = widgets.IntRangeSlider(
                                                    value=[1, 2],
                                                    min=1,
                                                    max=2,
                                                    step=1,
                                                    description='',
                                                    disabled=False,
                                                    continuous_update=False,
                                                    orientation='horizontal',
                                                    readout=True,
                                                    readout_format='d',
                                                    layout=widgets.Layout(width='80%')
                                                )

        self.barrier_name_text = widgets.Text(
                                        value='',
                                        placeholder='Barrier Name',
                                        description='Barrier Name',
                                        disabled=False,
                                        layout=widgets.Layout(width='30%')
                                        )
        
        self.set_barrier_info = widgets.Button(description='Save This Barrier Infos',
                                        disabled=False,
                                        button_style='info',
                                        tooltip='Set this barrier name and frame range',
                                        icon='',
                                        layout=widgets.Layout(width='20%')
                                        )
        self.set_barrier_info.on_click(self._set_barrier_info_callback)

        self.delete_barrier = widgets.Button(description='Delete This Barrier',
                                        disabled=False,
                                        button_style='danger',
                                        tooltip='Delete this barrier',
                                        icon='',
                                        layout=widgets.Layout(width='20%')
                                        )
        self.delete_barrier.on_click(self._del_barrier_callback)

        self.print_barrier_info = widgets.Button(description='Print Barrier Infos',
                                        disabled=False,
                                        button_style='success',
                                        tooltip='Print All Barrier Infos',
                                        icon='',
                                        layout=widgets.Layout(width='30%', height='40px')
                                        )
        self.print_barrier_info.on_click(self._print_barrier_info_callback)

        self.print_barrier_results_text = widgets.Textarea(
                                                        value='',
                                                        placeholder='Barrier Info Results',
                                                        description='',
                                                        disabled=False,
                                                        layout=widgets.Layout(width='90%', height='120px')
                                                    )
                                                    
        self.play_widget_slider = widgets.IntSlider(continuous_update=False,
                                                    description='Frame: ',
                                                    min=1,
                                                    max=1,
                                                    step=1,
                                                    layout=widgets.Layout(width='100%',height='40px')
                                                    )
        self.play_widget_slider.on_trait_change(self._play_slider_onchange_callback)
        
        

        self.msg0 = '<br> <p style="font-size:30px"> Input Path: </p><br>'
        self.hbox1 = widgets.HBox([self.draw_current_coords, self.clear_current_coords, self.add_barrier_button])

        self.hbox2 = widgets.HBox([self.play_widget_slider])

        self.msg1 = '<br> <p style="font-size:30px"> Pick Coordinates and Add Barriers: </p><br>'

        self.msg2 = '<br> <p style="font-size:30px"> Edit Counting Barrier Parameters and Print Results: </p><br>'

        self.hbox3 = widgets.HBox([widgets.Label('Frame Range '), self.frames_barrier_slider])

        self.vbox1  = widgets.VBox([self.hbox1, self.hbox2])
        self.vbox2  = widgets.VBox([self.select_barrier_dropdown, self.hbox3 , self.barrier_name_text,
                                    self.set_barrier_info, self.delete_barrier, self.print_barrier_info, self.print_barrier_results_text ])

        # Object Vars
        self.input_path = None #str
        self.input_handler = None
        self.img_atual_clean = None
        self.img_atual_with_barriers = None
        self.barriers = []
        self.current_barrier_selection = None
        self.frame_num_atual = None

    def _del_barrier_callback(self, val):
        current_barrier = self.select_barrier_dropdown.value
        if self.select_barrier_dropdown.options == ['']: return
        
        # Remove from self.barriers list
        for idx,bi in enumerate(self.barriers):
            if bi.name == current_barrier:
                self.barriers.pop(idx)
                break
        # Remove frmom drop down menu
        lista = list(self.select_barrier_dropdown.options)
        for idx, value in enumerate(lista):
            if value == current_barrier:
                lista.pop(idx)
        if len(lista) == 0: lista=['']
        self.select_barrier_dropdown.options = lista
        self.select_barrier_dropdown.value = lista[0]
        self.barrier_name_text.value = self.select_barrier_dropdown.value
        self._update_select_barriers_dropdown()
        self._draw_all_barriers_frame_atual()
        self._encode_and_display_img(self.img_atual_with_barriers)

    def _set_barrier_info_callback(self, val):
        old_name = self.select_barrier_dropdown.value
        new_name = self.barrier_name_text.value
        if new_name == '': return
        sf, ef = self.frames_barrier_slider.value
        
        def make_match(new_name):
            m_counts = 0
            mat = None
            for bi in self.barriers:
                if bi.name == new_name:
                    m_counts += 1
                if bi.name == old_name:
                    mat = bi
            return mat, m_counts
        
        match, match_counts = make_match(new_name)
        itera = 1
        max_itera = 1000
        while match_counts > 1:
            new_name = new_name + f'_{itera}'
            match, match_counts = make_match(new_name)
            itera += 1
            if itera > max_itera:
                raise Exception('Problemns trying to find a valid name for the barrier. Max Iteration.')
        
        match.start_frame = sf
        match.end_frame = ef
        match.name = new_name
        lista = list(self.select_barrier_dropdown.options)
        for idx,val in enumerate(lista):
            if val == old_name:
                lista[idx] = new_name
                break
        self.select_barrier_dropdown.options = lista
        self.select_barrier_dropdown.value = new_name
        self._update_select_barriers_dropdown()
        self._draw_all_barriers_frame_atual()
        self._encode_and_display_img(self.img_atual_with_barriers)

    def _print_barrier_info_callback(self, val):
        vid_name = self.input_handler.input_path.stem
        res = f"{{'{vid_name}' : ["
        for bi in self.barriers:
            res += f'CountBarrier(p1={bi.p1}, q1={bi.p2}, start_frame={int(bi.start_frame)}, end_frame={int(bi.end_frame)}, name=\'{bi.name}\'),\n '
        res= res[:-3] + ']}'
        self.print_barrier_results_text.value=res

    def _play_slider_onchange_callback(self,val):
        frame = self.play_widget_slider.value
        if frame <= 0 : return
        self._update_frame_atual(frame)

    def _get_barrier_unique_name(self):
        count = len(self.barriers)+1
        initial = f'B{count}'
        barriers_names = [bi.name for bi in self.barriers]
        it_limit=1000
        while (initial in barriers_names):
            count+=1
            initial=f'B{count}'
            if count > it_limit: raise Exception('Problems trying to generate a unique name for the barrier. Max iteration limit reached.')
        return initial
        
    def _add_barrier_button_callback(self, val):
        p1,p2 = self._get_current_coords()
        p1_abs, p2_abs = self._get_current_coords_abs()
        # Add barriers to self.barriers
        _,_,qtt_of_frames = self.input_handler.get_frame_infos()
        barrier_unique_name = self._get_barrier_unique_name()
        bi = Barriers(p1_abs, p2_abs, start_frame=0, end_frame = qtt_of_frames, name=barrier_unique_name)
        self.barriers.append(bi)
        self._draw_all_barriers_frame_atual()
        self._encode_and_display_img(self.img_atual_with_barriers)
        self._update_select_barriers_dropdown()

    def _update_select_barriers_dropdown(self):
        current_value = self.select_barrier_dropdown.value
        if len(self.barriers) == 0:
            self.select_barrier_dropdown.options = ['']
            self.select_barrier_dropdown.value = ''
        else:
            self.select_barrier_dropdown.options = [bi.name for bi in self.barriers]
            if not current_value in self.select_barrier_dropdown.options: #current_value removed
                self.select_barrier_dropdown.value = self.select_barrier_dropdown.options[0]
        self._update_barrier_control_parameters()
    
    def _update_barrier_control_parameters(self,val=None):
        current_barrier = ''
        for bi in self.barriers:
            if bi.name == self.select_barrier_dropdown.value:
                current_barrier = bi
                break
        if current_barrier == '': return
        #Frame Range Slider
        sf = current_barrier.start_frame
        ef = current_barrier.end_frame
        _,_,max_frames = self.input_handler.get_frame_infos()
        self.frames_barrier_slider.max = max_frames
        self.frames_barrier_slider.set_trait('value', [sf, ef])
        # Barrier Name Text Widget
        name = current_barrier.name
        self.barrier_name_text.value = name
        self._draw_all_barriers_frame_atual()
        self._encode_and_display_img(self.img_atual_with_barriers)
    
    def _draw_all_barriers_frame_atual(self):
        if len(self.barriers) == 0:
            self.img_atual_with_barriers = self.img_atual_clean.copy()
            return
        img = self.img_atual_clean.copy()
        for bi in self.barriers:
            if not (bi.start_frame <= self.frame_num_atual <= bi.end_frame): continue
            p1, p2 = self._to_rel_coords(bi.p1), self._to_rel_coords(bi.p2)
            color = (0,0,255)
            if bi.name == self.select_barrier_dropdown.value:
                color = (0,255,255)
            img = cv2.line(img, p1, p2, color,2)
        self.img_atual_with_barriers = img.copy()   

    def _round_to_int(self,p_i):
        return int(np.around(p_i[0])),int(np.around(p_i[1]))

    def _to_rel_coords(self,p_i):
        return self._round_to_int((p_i[0]*self.disp_w/self.w, p_i[1]*self.disp_h/self.h))

    def _to_abs_coords(self,p_i):
        return p_i[0]*self.w/self.disp_w, p_i[1]*self.h/self.disp_h

    def _get_current_coords_abs(self):
        p1 = self._round_to_int(self._to_abs_coords((self.x1, self.y1)))
        p2 = self._round_to_int(self._to_abs_coords((self.x2, self.y2)))
        return p1, p2

    def _get_current_coords(self):
        p1 = self._round_to_int((self.x1, self.y1))
        p2 = self._round_to_int((self.x2, self.y2))
        return p1, p2 

    def _draw_current_coords_callback(self, val):
        p1, p2 =  self._get_current_coords()
        img = self.img_atual_with_barriers.copy()
        cv2.circle(img, p1, 6, (0,0,255), 3)
        cv2.circle(img, p2, 6, (0,0,255), 3)
        self._encode_and_display_img(img)
    
    def _clear_current_coords_drawn_callback(self, val):
        self._encode_and_display_img(self.img_atual_with_barriers)

    def _input_path_button_callback(self, value):
        self.input_path = self.input_path_widget.value
        try:
            ih = InputHandler(self.input_path)
            self.input_handler = ih
        except Exception as e:
            self.status_text_box_widget.value= str(e)
            return

        # Status Message
        file_name = ih.input_path
        folder_or_file = 'File' if ih.input_type == 'vid' else 'Folder'
        file_type = 'Video' if ih.input_type == 'vid' else 'Images'
        _,_,qtt_of_frames = ih.get_frame_infos()
        self.status_text_box_widget.value = f'{folder_or_file} Name: {file_name.stem + file_name.suffix}. File type: {file_type}. Frames: {qtt_of_frames}'

        # Draw first frame
        self._update_frame_inicial()
        # Update frame_num_atual
        self.frame_num_atual = 1    

        # Play Widget Update
        self.play_widget_slider.value = 1
        self.play_widget_slider.max = qtt_of_frames
        # Barrier Slider Update
        self.frames_barrier_slider.max = qtt_of_frames

    def _update_frame_inicial(self):
        self._update_frame_atual(1, frame_inicial=True)

    def _update_frame_atual(self, frame_num, frame_inicial=False):
        if self.input_handler.input_type == 'dir':
            path = str(self.input_handler.dir_img_paths[frame_num])
            self.img_atual_clean = cv2.imread(path)
        elif self.input_handler.input_type == 'vid':
            self.img_atual_clean = self.input_handler.get_vid_frame(frame_num)
        
        # Check if the video or images keep frame shapes constant 
        if frame_inicial: self.h, self.w = self.img_atual_clean.shape[:2]
        h,w = self.img_atual_clean.shape[:2]
        if (h != self.h) or (w != self.w):
            raise Exception(f'Initial frame ({self.w}x{self.h}) and frame {frame_num} ({w},{h}) have different shapes.')
        
        # Reshape to display size
        self.img_atual_clean = cv2.resize(self.img_atual_clean, (self.disp_w, self.disp_h))

        self.frame_num_atual = frame_num
        self._draw_all_barriers_frame_atual()
        self._encode_and_display_img(self.img_atual_with_barriers)

    def _encode_and_display_img(self, img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        encoded_img = str(base64.b64encode(img))
        display(Javascript(f'document.getElementById("def_count_barriers").src = "data:image/jpg;base64,{encoded_img[2:-1]}"'))


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
            self._vid_cap_atual = self._vid_get_new_cap()
            self._vid_get_frame_infos()
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
    
    def _vid_get_new_cap(self):
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
        cap = self._vid_cap_atual
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1) #cap_prop_pos_frames is zero indexed
        status, self._vid_img_atual = cap.read()
        if not status: raise Exception(f'Problems reading frame {frame}.')
        self._vid_frame_atual = frame
        return self._vid_img_atual

    def _vid_get_frame_exactly(self, frame):
        if frame < self._vid_frame_atual:
            self._vid_cap_atual = self._vid_get_new_cap()
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
        cap = self._vid_cap_atual
        self._vid_first_frame = 1
        self._vid_last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._vid_qtt_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def _vid_get_frames_infos_exactly(self):
        cap = self._vid_get_new_cap()
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
