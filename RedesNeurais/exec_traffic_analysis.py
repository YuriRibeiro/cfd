from glob import glob
import os
import re
import numpy as np
import cv2
import pathlib
from matplotlib import pyplot as plt

# DEBUG ##########################################
DEBUG_MODE = True
if __name__ != "__main__":
    # If not main, deactivate debug automatically
    DEBUG_MODE = False

if DEBUG_MODE:
    os.chdir("./RedesNeurais")
##################################################


# Importar módulos da pasta Submodules
import sys
sys.path.append("../Submodules/sort/") # Adicionar Submodules ao PATH:
import sort
import time


# class AdHocUtils():
#     """
#     This class provides adhoc functions to commom tasks.
#     """
#     @staticmethod
#     def check_folder_existence(self, folder_path, create_folder=True):
#         """
#         Check if a folder path exists, and if this path leads to a folder.
#         If it doesn't exist, then this function returns false.
#         """
#         pass

#     @staticmethod
#     def yolov5_resume_labels_to_one_file(self, input_folder_path, output_file_path):
#         """
#         Take all txt files with labels for one image each, and resume all these labels
#         in one txt file.

#         Input: folder path
#         Output: results file path
#         """
#         ## Check Input/ Output folders and files

#         #Enforce folder path to terminate with os.sep
#         input_folder_path = input_folder_path+os.sep if input_folder_path[-1] != os.sep else input_folder_path
#         #Enforce file path to do not terminate with os.sep
#         output_file_path = output_file_path[:-1] if output_file_path[-1]==os.sep else output_file_path

#         # check if input folder doesn't exists:
#         if not os.path.isfolder(input_folder_path):
#             raise Exception(f"Input folder '{input_folder_path}' not found.")

#         # check if output file path exists:
#         if os.path.isfile(output_file_path):
#             raise Exception(f"Ouput file '{output_file_path}' already exists.")

#         # check if output folder doesn't exists:
#         # if it doesn't exists, create it.
#         output_folder_path = f"{os.sep}".join(output_file_path.split(os.sep)[:-1])

#         if not os.path.isfolder(output_folder_path):
#             pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        
#         ## Collect all txt files with labels and resume it to one resulting file

#         # Collect all txt files
#         txt_file_paths = glob(output_folder_path + "*.txt")

#         pattern = re.compile("\d+\.")
#         frame_number_aux = lambda name: pattern.search(name)
#         frame_number_lambda = lambda name: int(frame_number_aux(name).group()[:-1])
        
#         with open(output_file_path, 'w') as output_arq:

#             for file_path in txt_file_paths:
#                 file_name = file_path.split(os.sep)[-1]
#                 frame_number = frame_number_lambda()

#                 with open(file_path) as f:
#                     lines = f.readlines()
#                     for line in lines:
#                         x = float(line[1])
#                         y = float(line[2])
#                         w = float(line[3])
#                         h = float(line[4])
#                         classe = int(line[5])

                        

class GatherData():
    """
    Gather all data for the given paths to perform traffic analysis.

    Each line of the detections and gt files should have their data organized as follows:
        [frame, x, y, w, h, class]

    All coordinates should be absolute, i.e., not normalized between 0 and 1 or anything else.

    Input:
        video_path : path to the folder containing the video frames.
        detections_path :  path to the text file containing the neural network detections for all frames.
        gt_path : path to the text file containing groun truth detections for all frames.
        image_file_format : the image file format of the files inside the input folder.
    Output:
        self.gather_data(): returns dictionaries "frame indexed".
         video_frames_paths, gt_labels, predt_labels = obj.gather_data()

    """
    def __init__(self, video_path = "../Datasets/UAVDT/UAV-benchmark-M/M0101/",
                       detections_path = "./YOLOv5_UAVDT_0/resultados/inferencia_uavdt/M0101_results/",
                       gt_path = "../Datasets/UAVDT/UAV-benchmark-MOTD_v1.0/GT/M0101_gt_whole.txt",
                       image_file_format = "jpg"
                ):
        # Object Vars
        self.video_path = video_path if video_path[-1] == os.sep else video_path + os.sep
        self.detections_path = detections_path
        self.gt_path = gt_path
        self.image_file_format = image_file_format

        # Check for the existence of the folder and files:
        if not os.path.isdir(self.video_path):
             raise Exception(f"Video path '{self.video_path}' not found.")
        if not os.path.isfile(self.detections_path):
             raise Exception(f"Detections file '{self.detections_path}' not found.")
        if self.gt_path != "" and not os.path.isfile(self.gt_path):
             raise Exception(f"Ground truth file '{self.gt_path}' not found.")

        # Initialize the fields with None values:
        # Original Images Paths
        self.video_frames_paths = None
        # Ground Truth Labels for each frame, starting from frame number 1
        self.gt_labels = None
        # Predicted Labels (detections)
        self.predt_labels = None

    def collect_video_frames_paths(self):
        pattern = re.compile("\d+\.")
        frame_number_aux = lambda name: pattern.search(name)
        frame_number = lambda name: int(frame_number_aux(name).group()[:-1])

        frames_paths = glob(self.video_path + f"*.{self.image_file_format}")

        temp = {}
        for path in frames_paths:
            temp[frame_number(path)] = path
        return temp

    def collect_labels(self, file_path):
        temp = {}
        with open(file_path) as f:
            for line in f.readlines():
                line = line.split(",")
                frame = int(line[0])
                x = float(line[1])
                y = float(line[2])
                w = float(line[3])
                h = float(line[4])
                classe = int(line[5])

                if frame in temp:
                    temp[frame].append([frame, x, y, w, h, classe])
                else:
                    temp[frame] = [[frame, x, y, w, h, classe]]
        return temp
    
    def gather_data(self):
        # Original Images Paths
        self.video_frames_paths = self.collect_video_frames_paths()
        
        # Ground Truth Labels for each frame, starting from frame 1
        self.gt_labels = {}
        if self.gt_path != "":
            self.gt_labels = self.collect_labels(self.gt_path)
        else:
            self.gt_labels = None
        
        # Predicted Labels (detections)
        self.predt_labels = self.collect_labels(self.detections_path)

        return self.video_frames_paths, self.gt_labels, self.predt_labels


class TrackingData():
    def __init__(self):
        pass
    
    @staticmethod
    def SORT(self, predictions):
        qtt_frames = len(predictions)
        tracker = sort.Sort()

        tracked_vehicles_trajectory = {} # Trajetória de cada ID identificado
        vehicles_on_the_frame = {} # Veículos que estão presentes no frame X

        mot_labels = [[0, 0, 0, 0, 0, 0, 0] for _ in range(qtt_frames + 1)]

        for frame_number in range(1, qtt_frames+1):

            bboxes_atual = predictions[frame_number][:]

            # Formatar a lista para alimentar o Sort
            # np.array( [ [x1,y1,x2,y2,score1], [x3,y3,x4,y4,score2], ... ] )

            if len(bboxes_atual) == 0:
                bboxes_atual = np.zeros((0, 5)) # Requerido pelo Algoritmo Sort
            else:
                for idx in range(len(bboxes_atual)):
                    x1, y1, w, h, classe = bboxes_atual[idx][1:]
                    x2 = x1 + w
                    y2 = y1 + h
                    score = np.random.randint(50, 100)/100 # Temporariamente setar score como random.
                    bboxes_atual[idx] = [x1, y1, x2, y2, score, classe]
                
                # Numpy array requerido pelo Sort
                bboxes_atual = np.array(bboxes_atual)
                
                #last_col = bboxes_atual[:, -1]
                #find_rows = lambda classe: bboxes_atual[last_col == classe].copy()
                #bboxes_cars = find_rows(1)
                #bboxes_truck = find_rows(2)
                #bboxes_bus = find_rows(3)
            
                # Analisar o frame atual e identificar os bounding boxes id (update SORT)
                track_bbs_ids = tracker.update(bboxes_atual[:,:-1])
                this_frame_ids = track_bbs_ids[:,-1]
                #track_bbs_ids_cars = tracker_car.update(bboxes_cars)
                #track_bbs_ids_truck = tracker_truck(bboxes_truck)
                #track_bbs_ids_bus = tracker_bus(bboxes_bus)
                
                # Passar as coordenadas para o padrão: [frame,x,y,w,h,idx]
                newboxes_list = [[0,0,0,0,0,0,0] for _ in range(len(track_bbs_ids))]
                for i, newbox in enumerate(track_bbs_ids):
                    x1,y1,x2,y2,idx = newbox
                    x, y, w, h = x1, y1, abs(x2-x1), abs(y2-y1)
                    x,y,w,h,idx = int(x), int(y), int(w), int(h), int(idx)
                    newboxes_list[i] = [frame_number, x, y, w, h, classe, idx]

                    # Guardar a trajetória do centro do veículo IDx
                    xc, yc = int(x + w/2) , int(y + h/2)
                    if idx in tracked_vehicles_trajectory:
                        tracked_vehicles_trajectory[idx].append((frame_number,xc,yc))
                    else:
                        tracked_vehicles_trajectory[idx] =  [(frame_number,xc,yc)]
                
                # Atualizar a variável global
                vehicles_on_the_frame[frame_number] = this_frame_ids
                mot_labels[frame_number] = newboxes_list[:]

        return mot_labels, tracked_vehicles_trajectory, vehicles_on_the_frame 

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
        self.intersection_frames = set([])
    
    def add_counter(self, value = 1):
        self.counter += value
    
    def add_intersection_frame(self, frame_number):
        self.intersection_frames.add(frame_number)
    
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


class TrafficAnalysis(GatherData):
    def __init__(self, class_color={1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}, **kwargs):
        super().__init__(**kwargs)

        # Class Colors: Dictionay containing the color of the box.
        self.class_color = class_color

        # Inference Data
        self.video_frames_paths = None
        self.gt_labels = None
        self.predt_labels = None
        
        # Calculate Tracking Data
        self.mot_labels = None
        self.tracked_vehicles_trajectory = None
        self.vehicles_on_the_frame = None

        # Define the List of Barriers
        self.__barriers = None
    
    def __track_with_SORT(self):
        #tracker = TrackingData()
        self.mot_labels, self.tracked_vehicles_trajectory, self.vehicles_on_the_frame = TrackingData.SORT(None,self.predt_labels)
    
    def add_count_barriers(self, barrier_objects : "List of barrier objects"):
        """
        Define a lista de barreiras a ser utilizada no vídeo.
        """
        if self.__barriers == None:
            self.__barriers = []
        
        for bobj in barrier_objects:
            if not isinstance(bobj, CountBarrier):
                raise Exception("The barrier must be object of the CountBarrier Class.") 
            self.__barriers.append(bobj)

    def __count_barrier_crossings(self):
        """
        Conta quantos carros cruzaram as barreiras.
        """
        for vID in self.tracked_vehicles_trajectory:
            trajectory = self.tracked_vehicles_trajectory[vID]
            trajectory.sort(key=lambda x: x[0]) #Sort by frame...

            for barrier in self.__barriers:
                start_frame = barrier.start_frame
                end_frame = barrier.end_frame

                for idx in range(len(trajectory) - 1):
                    position_past = trajectory[idx][1:]
                    position_present = trajectory[idx + 1][1:]
                    frame_present = trajectory[idx+1][0]

                    if barrier.doIntersect(position_past, position_present):
                        barrier.add_counter(value=1)
                        barrier.add_intersection_frame(frame_present)


    def start_analysis(self, tracking_alg="SORT"):
        """
        Main Function to Handle Analysis Procedures.
        """
        # 0) Check for possible errors
        
        ## Check for the existence of barriers and make sure their counter starts with zero.
        if len(self.__barriers) == 0: raise Exception("No counting barrier defined.")

        # 1) Gather data about the image paths, ground truth and inference.
        self.video_frames_paths, self.gt_labels, self.predt_labels = self.gather_data()
        
        # 2) Gather data about tracking.
        if tracking_alg == "SORT":
            self.__track_with_SORT()

        # 3) Count vehicles that have crossed the barriers
        self.__count_barrier_crossings()
        
        # 4) Make Data Analysis

        # 5)
    
    @staticmethod
    def draw_boxes(self, img, boxes_list, color_dict, vars_cfg=0):
        # Possible boxes vars list configurations
        cfg = {0: {"x":1, "y":2, "w":3, "h":4, "classe":5},
               1: {"x":1, "y":2, "w":3, "h":4, "classe":5, "idx":6},
               2: {"x":2, "y":3, "w":4, "h":5, "classe":8, "idx":1}}
        box_vars_positions = cfg[vars_cfg]
        
        #Parse vars positions
        xpos = box_vars_positions["x"]
        ypos = box_vars_positions["y"]
        wpos = box_vars_positions["w"]
        hpos = box_vars_positions["h"]

        classepos = box_vars_positions["classe"] if "classe" in box_vars_positions else None
        idxpos = box_vars_positions["idx"] if "idx" in box_vars_positions else None
        
        # Draw
        for box in boxes_list:
            # Box
            x, y, w, h = int(box[xpos]), int(box[ypos]), int(box[wpos]), int(box[hpos])
            classe = box[classepos] if classepos != None else None
            color = color_dict[classe] if classe != None else (255,255,255)
            img = cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            
            # Idx
            if idxpos != None:
                idx = box[idxpos]
                cv2.putText(img, f"{idx}", (x+w,y),
                             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, 2)
    
    def convenience_define_barriers(self, countBarrier_objs=[]):
        """
        Facilitador para a tarefa de definir barreiras em um vídeo.
        
        1) Detectar os mouses clicks e selecionar as barreiras.
        2) Botão Draw para desenhar a barreira.
        3) Botão select para entrar no modo de seleção de barreira.
        4) Quando uma barreira for selecionada, mostrar opções de exclusão,
           set_name, set_first_frame, set_end_frame, edit_coords.
        """
        show = lambda img: cv2.imshow("Define Barriers", img)
        nothing = lambda x: None

        color  = (0,0,255)
        selected_color = (125,0,125)
        thickness = 2
        
        last_x = None
        last_y = None
        barriers = {}
        current_selection = set([])

        if not countBarrier_objs == []:
            pass # Parse objs to barriers style.

        if self.video_frames_paths == None:
            self.video_frames_paths, _, _ = self.gather_data()

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
                    barriers[len(barriers)+1] = [(last_x, last_y), (x, y), (1, len(self.video_frames_paths))] 
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
        cv2.namedWindow("Define Barriers")

        cv2.createTrackbar("Frame","Define Barriers",1, len(self.video_frames_paths),nothing)
        cv2.setTrackbarMin("Frame", "Define Barriers", 1)

        cv2.createTrackbar("Barrier Start Frame","Define Barriers",1,len(self.video_frames_paths),nothing)
        cv2.setTrackbarMin("Barrier Start Frame", "Define Barriers", 1)

        cv2.createTrackbar("Barrier End Frame", "Define Barriers",1,len(self.video_frames_paths),nothing)
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
                frame_path = self.video_frames_paths[frame]
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
                new_frame = frame+1 if frame < len(self.video_frames_paths) else frame
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
        
        list_of_CountBarrier_objects = []
        for idx, barrier_infos in enumerate(list_of_barriers):
            p1, q1, (start_frame, end_frame) = barrier_infos
            name = f"b{idx}"
            list_of_CountBarrier_objects.append(CountBarrier(p1,q1,start_frame, end_frame, name))
        
        return list_of_CountBarrier_objects


    def view(self, output_dir =f".{os.sep}view_results{os.sep}", frame_by_frame = False,
             draw_boxes_vars_config = 0):
        """
        Visual Results.
        Three outputs:
            1) Ground Truth boxes (if ground truth labels are given);
            2) Detected boxes + ID;
            3) Tracking ids and predt. by tracking bounding boxes
            4) Barrier + line of trajectory.
        """
        # If the output_dir doesn't exist, create it.
        if os.path.exists(output_dir): raise Exception(f"Output folder {output_dir} already exists.")
        else: pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        txt_origin = lambda img: (int(0.05*img.shape[0]), int(0.05*img.shape[1]))
        txt_font = cv2.FONT_HERSHEY_SIMPLEX
        txt_scale = 1
        txt_color = (0,255,255)
        txt_thickness = 2

        for frame in range(1, len(self.video_frames_paths) + 1):
            video_frame_path = self.video_frames_paths[frame]
            
            # GT Boxes
            img_gt = cv2.imread(video_frame_path)
            if self.gt_labels != None:
                self.draw_boxes(self, img_gt, self.gt_labels[frame], self.class_color,
                                vars_cfg=draw_boxes_vars_config)
                gt_text = "GT Boxes"
            else: gt_text = "GT Boxes Not Provided"
            cv2.putText(img_gt, gt_text, txt_origin(img_gt),
                        txt_font, txt_scale, txt_color, txt_thickness)
            
            # Predt. Labels
            if self.predt_labels != None:
                img_predt = cv2.imread(video_frame_path)
                self.draw_boxes(self, img_predt, self.predt_labels[frame], self.class_color,
                                 vars_cfg=0)
                cv2.putText(img_predt, "Predt. Labels", txt_origin(img_predt),
                            txt_font, txt_scale, txt_color, txt_thickness)

            # Tracking -- Predicted boxes and IDs
            if self.mot_labels != None:
                img_track = cv2.imread(video_frame_path)
                self.draw_boxes(self, img_track, self.mot_labels[frame], self.class_color,
                                vars_cfg=1)
                cv2.putText(img_track, "Tracking IDs", txt_origin(img_track),
                            txt_font, txt_scale, txt_color, txt_thickness)
            
            # Line of Trajectory + Barrier
            if self.tracked_vehicles_trajectory != None and self.__barriers != None:
                img_barriers = cv2.imread(video_frame_path)
                cv2.putText(img_barriers, "Counting and Tracking", txt_origin(img_barriers),
                            txt_font, txt_scale, txt_color, txt_thickness)
                # Draw barriers:
                for barrier in self.__barriers:
                    if barrier.start_frame <= frame <= barrier.end_frame:
                        if frame in barrier.intersection_frames:
                            color = (255,0,0)
                        else:
                            color = (0,0,255)
                        cv2.line(img_barriers, barrier.p1, barrier.q1, color, 2)
                        cv2.putText(img_barriers, barrier.name, barrier.p1, txt_font, txt_scale, txt_color, txt_thickness)

                # Check vehicles on the present frame:
                vehicles_on_present_frame = self.vehicles_on_the_frame[frame]

                # For each vehicle, draw lines connecting all positions from the first
                # to the present frame.
                for vehicle in vehicles_on_present_frame:
                    trajectory =[]
                    for positions in self.tracked_vehicles_trajectory[vehicle]:
                        veh_frame, veh_x, veh_y = positions
                        if veh_frame <= frame:
                            trajectory.append((veh_x, veh_y))
                    for i in range(len(trajectory)-1):
                        cv2.line(img_barriers, trajectory[i], trajectory[i+1], (0,255,255), 2)


            # Save the image view
            output_frame_path = output_dir + f"teste_{frame}.jpg"

            sub0 = np.concatenate((img_gt, img_predt), axis = 1)
            sub1 = np.concatenate((img_track, img_barriers), axis = 1)
            allimg = np.concatenate((sub0, sub1), axis = 0)
            
            if os.path.exists(output_frame_path):
                os.remove(output_frame_path)
            cv2.imwrite(output_frame_path, allimg)
            
            if frame_by_frame:
                # Show images frame by frame
                cv2.imshow(f"Frame {frame} of {len(self.video_frames_paths)}.", allimg)
                k = cv2.waitKey(0)
                if k == 27:
                    break
                cv2.destroyAllWindows()
        pass
    
    def statistical_analysis(self, output_dir = f".{os.sep}view_results{os.sep}", video_fps = 30):
        """
        1) Criar gráficos, estilo histograma, que mostrem a quantidade de veículos a cada X frames.
        2) Para esses X frames, criar um outro histograma com a quantidade de veículos cruzando cada barreira;

        FUTURE:
            change nbins to a correct value.
        """
        
        # Check for possible problems
        if self.vehicles_on_the_frame == None:
            raise Exception("Please, use the function gather_data() first.")
        if output_dir[-1] != os.sep:
            output_dir += os.sep
        if not os.path.exists(output_dir):
            raise Exception("Output dir doesn't exists.")
        if not os.path.isdir(output_dir):
            raise Exception("Output dir is not a folder.")
            
        # the barchart of the vehicles data
        qtt_frames = len(self.vehicles_on_the_frame)
        qtt_of_vehicles_by_frame = [len(x) for x in self.vehicles_on_the_frame.values()]

        fig, ax = plt.subplots(1,1)
        ax.bar(range(1, qtt_frames + 1), qtt_of_vehicles_by_frame)

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Quantity of Vehicles')
        ax.set_title('Vehicles by Frame')

        output_file_path = f"{output_dir}traffic_bar_chart.png"
        plt.savefig(output_file_path, format = "png")

        # Barchart Crossing Vehicles x Barriers
        pass



if DEBUG_MODE:                          
    """tf_data = TrafficAnalysis(video_path="./TesteFolder",         
                    gt_path="./TesteFolder/TesteFolder_results.txt",
                    detections_path="./TesteFolder/TesteFolder_results.txt",
                    image_file_format="jpg")

    tf_data = TrafficAnalysis(video_path="../Datasets/UAVDT/UAV-benchmark-M/M0101",
                        gt_path="../Datasets/UAVDT/UAV-benchmark-MOTD_v1.0/GT/M0101_gt_whole.txt",
                        detections_path="./YOLOv5_UAVDT_0/results/inference/UAVDT_Inference/M0101_results.txt",
                        image_file_format="jpg")
    """

    tf_data = TrafficAnalysis(video_path="./TesteFolder",
                        gt_path="",
                        detections_path="./ViewResults/TesteFolder_results.txt",
                        image_file_format="jpg")


    barriers = tf_data.convenience_define_barriers()
    tf_data.add_count_barriers(barriers)
    
    tf_data.start_analysis()
    tf_data.view(output_dir="./view_results2/", frame_by_frame = False, draw_boxes_vars_config=0)

    for b in barriers:
        print(b)

    tf_data.statistical_analysis()
pass