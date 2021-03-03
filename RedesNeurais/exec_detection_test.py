"""
Dados de entrada no script

experimento : Define quais pesos deverão ser utilizados. Exemplo: "YOLOv5_UAVDT_0"


Exemplo para rodar na linha de comando:
$ python exec_detection_test.py YOLOv5_UAVDT_0
"""

import sys
import os
import shutil
import subprocess
import argparse
import json
import glob

this_file_dir = os.path.abspath(os.path.dirname(__file__))

class YOLOv5_UAVDT_CONFIG:
    this_file_dir = this_file_dir

    yv5_weigth_path = lambda exp, date: os.path.join(this_file_dir, exp,
                                    "YOLOv5_UAVDT_train",f"{exp}-{date}","weights","best.pt")
    
    # Relatitve Weights Paths "Database"
    weight_paths = {"YOLOv5_UAVDT_0" : yv5_weigth_path("YOLOv5_UAVDT_0", "21_Feb_2021_18h_17m"),
                    "YOLOv5_UAVDT_1" : yv5_weigth_path("YOLOv5_UAVDT_1", "21_Feb_2021_19h_26m"),
                    "YOLOv5_UAVDT_2" : yv5_weigth_path("YOLOv5_UAVDT_2", "21_Feb_2021_21h_42m"),
                    "YOLOv5_UAVDT_3" : yv5_weigth_path("YOLOv5_UAVDT_3", "22_Feb_2021_11h_36m"),
                    "YOLOv5_UAVDT_4" : yv5_weigth_path("YOLOv5_UAVDT_4", "25_Feb_2021_13h_13m"),
                    "YOLOv5_UAVDT_5" : yv5_weigth_path("YOLOv5_UAVDT_5", "26_Feb_2021_04h_26m"),
                    "YOLOv5_UAVDT_6" : yv5_weigth_path("YOLOv5_UAVDT_6", "26_Feb_2021_04h_25m"),
                    "YOLOv5_UAVDT_7" : yv5_weigth_path("YOLOv5_UAVDT_7", "26_Feb_2021_04h_25m"),
                  "YOLOv5_UAVDT_301" : yv5_weigth_path("YOLOv5_UAVDT_301", "13_October_2020_14h_48m_24s"),
                  "YOLOv5_UAVDT_302" : yv5_weigth_path("YOLOv5_UAVDT_302", "09_October_2020_15h_11m_52s")}
            
    
    movies_teste = ["M0203", "M0205", "M0208", "M0209", "M0403", "M0601", \
                    "M0602", "M0606", "M0701", "M0801", "M0802", "M1001", "M1004", \
                    "M1007", "M1009", "M1101", "M1301", "M1302", "M1303", "M1401"]
    data_yaml = """
                # train and val datasets (image directory or *.txt file with image paths)

                #train: ../../../Datasets/UAVDT_YOLOv5/train/images/
                #test: ../../../Datasets/UAVDT_YOLOv5/test/images/
                val: ../../../Datasets/UAVDT_YOLOv5/test/images/

                # number of classes
                nc: 3

                # class names
                names: ['car', 'truck', 'bus']
                """
    
    def __init__(self): pass
                
    def argparser(self):
        # Argument Parser
        opt = argparse.ArgumentParser()
        opt.add_argument('experiment', action='store', default="")
        opt.add_argument('-r', '--resolution', action='store', dest="resolution", default= 640,
                        help="Resolução da imagem para utilizar na detecção. Default = 640.")
        opt.add_argument('-t', '--iou-threshold', action='store', dest="iou_thresh", default= 0.4,
                        help="Máximo IOU threshold permitido para utilizar no torchvision.ops.nms. Default = 0.4.")
        opt.add_argument('-c', '--conf-threshold', action='store', dest="conf_thresh", default= 0.4,
                        help="Mínima detection confidende permitida nas detecções. Default = 0.4.")
        opt.add_argument('-b', '--batch-size', action='store', dest="batch_size", default= 16,
                        help="Batch size para utilizar na detecção. Default = 16.")
        opt.add_argument('--task', action='store', dest="task", default= "study",
                        help="Detection task: test, study, speed. Default = study")
                        # Study task: Varia as resoluções das imagens e anota o mAP para cada size.
                        # Sizes estudados: range(256, 1536 + 128, 128))  # x axis (image sizes)
        opt.add_argument('-d','--device', action='store', dest="device", default= "cpu",
                        help="Detection using device: cpu, 0, 1, 2, 3. Default = cpu")
        opt.add_argument('-s','--single-class', action='store_true', dest="single_class",
                        help="Detection treat the dataset as single class")
        opt.add_argument('--save-txt', action='store_true', help='save results to *.txt')

        opt.add_argument('--verbose', action='store_false', help='report mAP by class')
        
        opt.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
        opt.add_argument('--save-json', action='store_false', help='save a cocoapi-compatible JSON results file')

        #opt.add_argument('--project', default='runs/test', help='save to project/name')
        #opt.add_argument('--name', default='exp', help='save to project/name')
        opt = opt.parse_args()
        return opt


class YOLOv5_UAVDT_DET(YOLOv5_UAVDT_CONFIG):
    def __init__(self):
        self.opt = self.argparser()
        self.experimento = self.opt.experiment
        if self.experimento not in self.weight_paths.keys():
            raise Exception(f"Experimento '{self.experimento}' não encontrado." +\
                             f"Experimentos disponíveis: {self.weight_paths.keys()}.")
        self.project = "YOLOv5_UAVDT_det"
        self.name = self.weight_paths[self.experimento].split(os.sep)[-3]
        self.run_inference()

    def run_inference(self):
        # Remove old yolov5 files and replace by a new one copy.
        
        yv5_path = os.path.join(self.this_file_dir, self.experimento, f"yolov5_w_det_temp")
        if os.path.exists(yv5_path):
            shutil.rmtree(yv5_path)
        yv5w_submodules_path = os.path.join("..","Submodules","yolov5_w")
        shutil.copytree(yv5w_submodules_path, yv5_path)

        # Remove old output dirs:
        self.output_dir_path = os.path.join(self.this_file_dir, self.experimento, self.project, self.name)
        if os.path.exists(self.output_dir_path): shutil.rmtree(self.output_dir_path)

        # Now, run the inference
        weights_file_path = self.weight_paths[self.experimento]
        test_py_path = os.path.join(yv5_path, "test.py")
        data_yaml_path = os.path.join(self.this_file_dir, self.experimento, "data_detection.yaml")
        with open(data_yaml_path, 'w') as arq:
            arq.write(self.data_yaml)

        opt = self.opt
        os.chdir(yv5_path)

        shell_command = ["python", str(test_py_path),
                        "--weights", str(weights_file_path),
                        "--data", str(data_yaml_path),
                        "--batch-size", str(opt.batch_size),
                        "--img-size", str(opt.resolution),
                        "--conf-thres", str(opt.conf_thresh),
                        "--iou-thres", str(opt.iou_thresh),
                        "--task", str(opt.task),
                        "--device", str(opt.device),
                        "--project", os.path.join("..", self.project),
                        "--name", self.name]
        """
            store_false args:
                        "--single-cls",str(opt.single_class),
                        "--verbose", str(opt.verbose),
                        "--save-conf", str(opt.save_conf),
                        "--save-json", str(opt.save_json),
            store_true args:
                        "--single-class",str(opt.single_class),
                        "--save-txt", str(opt.save_txt)
        """
        
        for k,v in opt.__dict__.items():
            if isinstance(v, bool) and v==True:
                mytable = k.maketrans("_", "-")
                shell_command.append(f"--{k.translate(mytable)}")

        process = subprocess.Popen(shell_command, 
                        stdout=subprocess.PIPE,
                        universal_newlines=True)

        while True:
            output = process.stdout.readline()
            print(output.strip())
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output 
                for output in process.stdout.readlines():
                    print(output.strip())
                break
        
        os.chdir(self.this_file_dir)
        
        # Create one file with detetions for each one video:
        # Open the "coco format" det json file
        videos = os.listdir(self.output_dir_path)

        for video in videos:
            output_path = os.path.join(self.output_dir_path, video)

            json_path = os.path.join(output_path, "best_predictions.json")
            test_set_det = { k:[] for k in self.movies_teste}
            with open(json_path) as f:
                data = json.load(f)

            for ann in data:
                vid, img_id = ann['image_id'].split("_")
                if not vid in self.movies_teste: continue
                x,y,w,h = ann['bbox']
                x, w = float(x), float(w)
                y, h = float(y), float(h)
                score = ann['score']
                # Note that the class doesn't appear.
                test_set_det[vid].append(f"{int(img_id[3:])},-1,{x},{y},{w},{h},{score},1,-1")

            output_dir_path = os.path.join(output_path, f"det_{self.name}")
            if os.path.exists(output_dir_path): shutil.rmtree(output_dir_path)
            os.mkdir(output_dir_path)

            for k,v in test_set_det.items():
                output_file_path = os.path.join(output_dir_path, f"{k}.txt")
                with open(output_file_path, "w") as f:
                    if not len(v) == 0:
                        f.write(v[0])
                        if len(v) >= 1:
                            for det in v[1:]:
                                f.write(f"\n{det}")
                    else:
                        f.write("")
            
        det_parameters_file_path = os.path.join(self.output_dir_path, 'detection_parameters.txt')
        # save args in "fromfile_prefix_chars" format
        # https://docs.python.org/3/library/argparse.html
        print(f"[INFO] Salvando argumentos em {det_parameters_file_path}.")
        with open(det_parameters_file_path, "w") as f:
            f.write(f"{shell_command[2]}")
            for item in shell_command[3:]:
                f.write(f"\n{item}")

        print(f"[INFO] Removendo {yv5_path}, {data_yaml_path}")
        shutil.rmtree(yv5_path)
        os.remove(data_yaml_path)
        print(f"[INFO] Resultados salvos em {output_path}")
        # Final msg
        print("[INFO] Fim da execução.")


class YOLOv3_UAVDT_CONFIG:
    this_file_dir = this_file_dir

    yv3_weigth_path = lambda exp, date: os.path.join(this_file_dir, exp,
                                    "YOLOv3_UAVDT_train",f"{exp}-{date}","weights","best.pt")
    
    # Relatitve Weights Paths "Database"
    weight_paths = {"YOLOv3_UAVDT_0" : yv3_weigth_path("YOLOv3_UAVDT_0", "28_Feb_2021_04h_35m"),
                    "YOLOv3_UAVDT_1" : yv3_weigth_path("YOLOv3_UAVDT_1", "28_Feb_2021_04h_36m"),
                    "YOLOv3_UAVDT_2" : yv3_weigth_path("YOLOv3_UAVDT_2", "28_Feb_2021_04h_36m"),
                    "YOLOv3_UAVDT_3" : yv3_weigth_path("YOLOv3_UAVDT_3", "01_Mar_2021_11h_34m"),
                    "YOLOv3_UAVDT_5" : yv3_weigth_path("YOLOv3_UAVDT_4", "01_Mar_2021_11h_34m"),
                    "YOLOv3_UAVDT_4" : yv3_weigth_path("YOLOv3_UAVDT_5", "01_Mar_2021_11h_35m")}

    movies_teste = ["M0203", "M0205", "M0208", "M0209", "M0403", "M0601", \
                    "M0602", "M0606", "M0701", "M0801", "M0802", "M1001", "M1004", \
                    "M1007", "M1009", "M1101", "M1301", "M1302", "M1303", "M1401"]
    data_yaml = """
                # train and val datasets (image directory or *.txt file with image paths)

                #train: ../../../Datasets/UAVDT_YOLOv5/train/images/
                #test: ../../../Datasets/UAVDT_YOLOv5/test/images/
                val: ../../../Datasets/UAVDT_YOLOv5/test/images/

                # number of classes
                nc: 3

                # class names
                names: ['car', 'truck', 'bus']
                """
    
    def __init__(self): pass
                
    def argparser(self):
        # Argument Parser
        opt = argparse.ArgumentParser()
        opt.add_argument('experiment', action='store', default="")
        opt.add_argument('-r', '--resolution', action='store', dest="resolution", default= 640,
                        help="Resolução da imagem para utilizar na detecção. Default = 640.")
        opt.add_argument('-t', '--iou-threshold', action='store', dest="iou_thresh", default= 0.4,
                        help="Máximo IOU threshold permitido para utilizar no torchvision.ops.nms. Default = 0.4.")
        opt.add_argument('-c', '--conf-threshold', action='store', dest="conf_thresh", default= 0.4,
                        help="Mínima detection confidende permitida nas detecções. Default = 0.4.")
        opt.add_argument('-b', '--batch-size', action='store', dest="batch_size", default= 16,
                        help="Batch size para utilizar na detecção. Default = 16.")
        opt.add_argument('--task', action='store', dest="task", default= "study",
                        help="Detection task: test, study, speed. Default = study")
                        # Study task: Varia as resoluções das imagens e anota o mAP para cada size.
                        # Sizes estudados: range(256, 1536 + 128, 128))  # x axis (image sizes)
        opt.add_argument('-d','--device', action='store', dest="device", default= "cpu",
                        help="Detection using device: cpu, 0, 1, 2, 3. Default = cpu")
        opt.add_argument('-s','--single-class', action='store_true', dest="single_class",
                        help="Detection treat the dataset as single class")
        opt.add_argument('--save-txt', action='store_true', help='save results to *.txt')

        opt.add_argument('--verbose', action='store_false', help='report mAP by class')
        
        opt.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
        opt.add_argument('--save-json', action='store_false', help='save a cocoapi-compatible JSON results file')

        #opt.add_argument('--project', default='runs/test', help='save to project/name')
        #opt.add_argument('--name', default='exp', help='save to project/name')
        opt = opt.parse_args()
        return opt

class YOLOv3_UAVDT_DET(YOLOv3_UAVDT_CONFIG):
    def __init__(self):
        self.opt = self.argparser()
        self.experimento = self.opt.experiment
        if self.experimento not in self.weight_paths.keys():
            raise Exception(f"Experimento '{self.experimento}' não encontrado." +\
                             f"Experimentos disponíveis: {self.weight_paths.keys()}.")
        self.project = "YOLOv3_UAVDT_det"
        self.name = self.weight_paths[self.experimento].split(os.sep)[-3]
        self.run_inference()

    def run_inference(self):
        # Remove old yolov3 files and replace by a new one copy.
        
        yv3_path = os.path.join(self.this_file_dir, self.experimento, f"yolov3_w_det_temp")
        if os.path.exists(yv3_path):
            shutil.rmtree(yv3_path)
        yv3w_submodules_path = os.path.join("..","Submodules","yolov3_w")
        shutil.copytree(yv3w_submodules_path, yv3_path)

        # Remove old output dirs:
        self.output_dir_path = os.path.join(self.this_file_dir, self.experimento, self.project, self.name)
        if os.path.exists(self.output_dir_path): shutil.rmtree(self.output_dir_path)

        # Now, run the inference
        weights_file_path = self.weight_paths[self.experimento]
        test_py_path = os.path.join(yv3_path, "test.py")
        data_yaml_path = os.path.join(self.this_file_dir, self.experimento, "data_detection.yaml")
        with open(data_yaml_path, 'w') as arq:
            arq.write(self.data_yaml)

        opt = self.opt
        os.chdir(yv3_path)

        shell_command = ["python", str(test_py_path),
                        "--weights", str(weights_file_path),
                        "--data", str(data_yaml_path),
                        "--batch-size", str(opt.batch_size),
                        "--img-size", str(opt.resolution),
                        "--conf-thres", str(opt.conf_thresh),
                        "--iou-thres", str(opt.iou_thresh),
                        "--task", str(opt.task),
                        "--device", str(opt.device),
                        "--project", os.path.join("..", self.project),
                        "--name", self.name]
        """
            store_false args:
                        "--single-cls",str(opt.single_class),
                        "--verbose", str(opt.verbose),
                        "--save-conf", str(opt.save_conf),
                        "--save-json", str(opt.save_json),
            store_true args:
                        "--single-class",str(opt.single_class),
                        "--save-txt", str(opt.save_txt)
        """
        
        for k,v in opt.__dict__.items():
            if isinstance(v, bool) and v==True:
                mytable = k.maketrans("_", "-")
                shell_command.append(f"--{k.translate(mytable)}")

        process = subprocess.Popen(shell_command, 
                        stdout=subprocess.PIPE,
                        universal_newlines=True)

        while True:
            output = process.stdout.readline()
            print(output.strip())
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output 
                for output in process.stdout.readlines():
                    print(output.strip())
                break
        
        os.chdir(self.this_file_dir)
        
        # Create one file with detetions for each one video:
        # Open the "coco format" det json file
        videos = os.listdir(self.output_dir_path)

        for video in videos:
            output_path = os.path.join(self.output_dir_path, video)

            json_path = os.path.join(output_path, "best_predictions.json")
            test_set_det = { k:[] for k in self.movies_teste}
            with open(json_path) as f:
                data = json.load(f)

            for ann in data:
                vid, img_id = ann['image_id'].split("_")
                if not vid in self.movies_teste: continue
                x,y,w,h = ann['bbox']
                x, w = float(x), float(w)
                y, h = float(y), float(h)
                score = ann['score']
                # Note that the class doesn't appear.
                test_set_det[vid].append(f"{int(img_id[3:])},-1,{x},{y},{w},{h},{score},1,-1")

            output_dir_path = os.path.join(output_path, f"det_{self.name}")
            if os.path.exists(output_dir_path): shutil.rmtree(output_dir_path)
            os.mkdir(output_dir_path)

            for k,v in test_set_det.items():
                output_file_path = os.path.join(output_dir_path, f"{k}.txt")
                with open(output_file_path, "w") as f:
                    if not len(v) == 0:
                        f.write(v[0])
                        if len(v) >= 1:
                            for det in v[1:]:
                                f.write(f"\n{det}")
                    else:
                        f.write("")
            
        det_parameters_file_path = os.path.join(self.output_dir_path, 'detection_parameters.txt')
        # save args in "fromfile_prefix_chars" format
        # https://docs.python.org/3/library/argparse.html
        print(f"[INFO] Salvando argumentos em {det_parameters_file_path}.")
        with open(det_parameters_file_path, "w") as f:
            f.write(f"{shell_command[2]}")
            for item in shell_command[3:]:
                f.write(f"\n{item}")

        print(f"[INFO] Removendo {yv3_path}, {data_yaml_path}")
        shutil.rmtree(yv3_path)
        os.remove(data_yaml_path)
        print(f"[INFO] Resultados salvos em {output_path}")
        # Final msg
        print("[INFO] Fim da execução.")





class Main:
    available_networks = ["YOLOv5_UAVDT", "YOLOv3_UAVDT"]
    experimento = None
    network = None

    def __init__(self):
        try:
            self.experimento = sys.argv[1]
            # Check experiment neural network model name:
            self.network = self.experimento.split("_")[0:2]
            self.network = "_".join(self.network)    
            if not self.network in self.available_networks:
                raise Exception(f"Rede '{self.network}' não encontrada. Redes disponíveis: {self.available_networks}.")
        
        except Exception as e:
            print(e)
        
        if self.network == "YOLOv5_UAVDT":
            YOLOv5_UAVDT_DET()
        elif self.network == "YOLOv3_UAVDT":
            YOLOv3_UAVDT_DET()



### Debug mode for Visual Studio Code..
debug_mode = False
if debug_mode:
    os.chdir(this_file_dir)
    sys.argv.append("YOLOv5_UAVDT_0")
Main()