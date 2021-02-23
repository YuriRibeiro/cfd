"""
Dados de entrada no script

experimento : Define quais pesos deverão ser utilizados. Exemplo: experimento = "YOLOv5_UAVDT_0"

input_folder : Define a pasta contendo as imagens a serem classificadas pela rede treinada no experimento.
               Pode ser uma pasta com imagens.
               Pode ser uma pasta com outras pastas em seu interior contendo imagens. Como se fosse uma 
               espécie de dataset, contendo vários vídeos.

output_folder : Pasta para guardar o arquivo de texto único contendo informações a respeito
                das bounding boxes para cada frame. O nome do arquivo de texto é dado automaticamente: 
                nome_do_filme_sendo_analisado + "results.txt"    dataset_name = input_folder.split("/")[-2]

Filosofia:
    Dado um input_path contendo imagens ou pastas com imagens, para cada vídeo analisado será criado um arquivo 
    txt com os bounding boxes detectados pela rede e será colocado na pasta output_path. Simples assim.
    Demais rotinas devem ser executadas pelo usuário que chama este script.

Padrões utilizados:
    1) Todos as imagens possuem em seu nome o frame a que correspondem.
        O número do frame está localizado imediatamente antes do "." que separa o nome do formato da imagem.
        Exemplos: img_00001.jpg => Frame = 1
                  img3013uav000021.png => Frame = 21
    2) Os frames começam a ser contados do número 1 e não do número zero.
    3) Supor que todas as images de um vídeo possuem as mesmas dimensões de altura e largura.

Exemplo de execução:

    experimento = "YOLOv5_UAVDT_0"
    input_folder = "../Datasets/UAVDT/UAV-benchmark-M/M0101"
    output_folder = "./YOLOv5_UAVDT_0/results/inference/UAVDT_Inference/"

$ python this_file_name.py -e {experimento} -i {input_folder} -o {output_folder}

# Futuro:
Carregar o endereço dos pesos a partir de um banco de dados local.
Dado o experimento, localizar o endereço dos pesos no banco de dados.

# Debug:
    python exec_inference.py -e YOLOv5_UAVDT_0 -i ./TesteFolder -o ./TesteFolder
"""

import os
import shutil
import subprocess
import pathlib
import tempfile
import re
import argparse
from PIL import Image as PILImage
from glob import glob
import random

data_yaml = """
# train and val datasets (image directory or *.txt file with image paths)

train: ../../../Datasets/UAVDT_YOLOv5/train/images/
test: ../../../Datasets/UAVDT_YOLOv5_Dataset/test/images/

# number of classes
nc: 3

# class names
names: ['car', 'truck', 'bus']
"""

# Weights Relatitve Paths "Database"
yv5_weigth_path = lambda exp, date: os.path.join(".",exp,"YOLOv5_UAVDT_train",f"{exp}-{date}","weights","best.pt")

weight_paths = {"YOLOv5_UAVDT_0" : yv5_weigth_path("YOLOv5_UAVDT_0", "21_Feb_2021_18h_17m"),
                "YOLOv5_UAVDT_1" : yv5_weigth_path("YOLOv5_UAVDT_1", "21_Feb_2021_19h_26m")}

# Argument Parser
opt = argparse.ArgumentParser()
opt.add_argument('-o', '--output', action='store', dest="output_folder", default="")
opt.add_argument('-i', '--input', action='store', dest="input_folder", default="")
opt.add_argument('-e', '--experiment', action='store', dest="experiment", default="")

opt.add_argument('-r', '--resolution', action='store', dest="resolution", default= 640,
                help="Resolução da imagem para utilizar na detecção. Default = 640.")
opt.add_argument('-t', '--iou-threshold', action='store', dest="iou_thresh", default= 0.5,
                help="IOU threshold para utilizar na detecção. Default = 0.5.")
opt.add_argument('-c', '--conf-threshold', action='store', dest="conf_thresh", default= 0.5,
                help="Detection confidende para utilizar na detecção. Default = 0.5.")
opt.add_argument('-b', '--batch-size', action='store', dest="batch_size", default= 16,
                help="Batch size para utilizar na detecção. Default = 16.")
opt.add_argument('--task', action='store', dest="task", default= "test",
                help="Detection task: test, study, speed. Default = test")
                # Study task: Varia as resoluções das imagens e anota o mAP para cada size.
                # Sizes estudados: range(256, 1536 + 128, 128))  # x axis (image sizes)
opt.add_argument('-d','--device', action='store', dest="device", default= "cpu",
                help="Detection using device: cpu, 0, 1, 2, 3. Default = cpu")
opt.add_argument('-s','--single class', action='store_true', dest="single_class",
                help="Detection trear the dataset as single class")
opt.add_argument('--verbose', action='store_true', help='report mAP by class')
opt.add_argument('--save-txt', action='store_false', help='save results to *.txt')
opt.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
opt.add_argument('--save-json', action='store_false', help='save a cocoapi-compatible JSON results file')
opt.add_argument('--project', default='runs/test', help='save to project/name')
opt.add_argument('--name', default='exp', help='save to project/name')

opt = opt.parse_args()

if opt.output_folder == "" or opt.input_folder == "" or opt.experiment == "":
    raise Exception("The input, output or experiment is an empty strings.")

# Main Function
class main():

    def __init__(self, opt, clear_io=True):
        self.opt = opt
        self.experimento = self.opt.experiment
        if self.experimento not in weight_paths.keys():
            raise Exception(f"Experimento '{self.experimento}' não encontrado." +\
                             "Experimentos disponíveis: {weight_paths.keys()}.")

        self.input_folder = opt.proje
        self.output_folder = self.opt.output_folder
        self.clear_io = clear_io #Clear ouput folder before saving new files
        self.package = False # False: Not a folder inside a folder. True, otherwise.
        self.videos_paths = None        

    def check_input_and_output_dirs(self):
        """
        Verify if the folders are ok. If they aren't, raise an Exception or fix the problem.
        """
        # Check if the folders paths terminate with a os.sep. If don't, force it.
        if not self.input_folder[-1] == os.sep: self.input_folder += os.sep
        if not self.output_folder[-1] == os.sep: self.output_folder += os.sep

        # Check if input folder is a valid folder, i.e, it exists.
        if not os.path.isdir(self.input_folder):
            raise Exception(f'O caminho do input_folder "{self.input_folder}" não é um diretório.')

        else: 
            ## Check for files/ folders inside input path
            ## If there is files and folders, raise an Exception.
            ## The folder should contain only files or only folders.
            objects = glob(self.input_folder + "*")
            self.package = os.path.isdir(objects[0])
            for obj in objects:
                # not XOR to check if all file are equals.
                if not (package ^ os.path.isdir(obj)):
                    pass
                else:
                    raise Exception(f"There is files and folders mixed together inside" +\
                                     "'{self.input_folder}'. That isn't permitted.")
            
            videos_paths = []
            if self.package:
                for directory in objects:
                    self.videos_paths.append(directory)
            else:
                self.videos_paths.append(self.input_folder)

        # Analysis of each video
        for video_path in videos_paths:
            # Check if the output folder exists. If doesn't, create it.
            self.craete_empty_folder(self.output_folder)
            
            # Create the output file path
            video_name = video_path.split(os.sep)[-2]
            output_file_path = self.output_folder + video_name + "_results.txt"

            # If for some reason the output file already exists, remove it.
            # If it's non empty directory, raise an Exception.
            if os.path.isdir(output_file_path):
                os.remove(output_file_path)
            if os.path.isfile(output_file_path):
                if self.clear_io:
                    os.remove(output_file_path)
                else:
                    raise Exception(f"{output_file_path} already exists and clear_io == False.")
        
        # Create output dir
        if os.path.exists(self.output_folder):
            if os.path.isdir(self.output_folder):
                if not os.listdir(self.output_folder):
                    # Empty directory, ok..
                    pass
                elif self.clear_io:
                    # Clear output directory
                    shutil.rmtree(self.output_folder)
                    pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
                else:
                    raise Exception(f"{self.output_folder} not empty and clear_io == False.")
            if os.path.isfile(self.output_folder):
                raise Exception(f"{self.output_folder} is a file, not a directory.")
        else:
            raise Exception(f"Unknown output folder/file: '{self.output_folder}'.")

    def run_inference(self):
        """
        This function makes the things happen.
        """
        # Check input and output dirs
        self.check_input_and_output_dirs()
        
        # Check experiment neural network model name:
        network = self.experimento.split("_")[0]

        if network == "YOLOv5":
            try:
                for video_path in self.videos_paths:    
                    # Remove old yolov5 files and replace by a new one copy.
                    yv5_path = os.path.join(".", self.experimento, f"yolov5_w_det_temp")
                    if os.path.exists(yv5_path):
                        shutil.rmtree(yv5_path)
                    os.symlink(os.path.join("..","..","Submodules","yolov5_w",), yv5_path)

                # Create a temporary directory to store intermediate results
                    tempdir = tempfile.mkdtemp()
                    print("\nDiretório temporário: ", tempdir)

                    # Now, run the inference
                    weights_file_path = weight_paths[self.experimento]
                    detect_py_path = os.path.join(yv5_path, "detect.py")
                    data_yaml_path = os.path.join(".", self.experimento, "data_detection.yaml")
                    with open(data_yaml_path, 'w') as arq:
                        arq.write(data_yaml)

                    opt = self.opt
                    original_path = os.getcwd()
                    os.chdir(yv5_path)
                    shell_command = ["python", detect_py_path,
                                    "--weights", weights_file_path,
                                    "--data", data_yaml_path,
                                    "--batch-size", opt.batch_size,
                                    "--img-size", opt.resolution,
                                    "--conf_thres", opt.conf_thresh,
                                    "--iou-thres", opt.iou_thres,
                                    "--task", opt.task,
                                    "--device", opt.device,
                                    "--single-cls",opt.single_cls,
                                    "--verbose", opt.verbose,
                                    "--save-txt", opt.save_txt,
                                    "--save-conf", opt.save_conf,
                                    "--save-json", opt.save_json,
                                    "--project", opt.project,
                                    "--name", opt.name]

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
                    os.chdir(original_path)
                    # Resume all outputs to one txt file
                    txt_files_paths = glob(os.path.join(tempdir, "*.txt"))
                    img_files_paths = glob(os.path.join(tempdir, "*.jpg"))
                    
                    pattern = re.compile("\d+\.")
                    frame_number_aux = lambda name: pattern.search(name)
                    frame_number = lambda name: int(frame_number_aux(name).group()[:-1])

                    txt_files_paths.sort(key = frame_number)
                    img_files_paths.sort(key = frame_number)

                    frame_boxes = {}
                    for txt_path, img_path in zip(txt_files_paths, img_files_paths):
                        if frame_number(txt_path) != frame_number(img_path):
                            raise Exception("Problema com a numeração das imagens e files preditos pela rede.")
                        frame = frame_number(txt_path)

                        with PILImage.open(img_path) as img:
                            width, height = img.size

                        with open(txt_path) as arq:
                            lines = arq.readlines()
                            for line in lines:
                                if line == "": continue
                                line = line.split(" ")
                                classe, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                                x = x - w/2
                                y = y - h/2
                                x, w = x*width, w*width
                                y, h = y*height, h*height
                                if frame in frame_boxes:
                                    frame_boxes[frame].append([frame, x, y, w, h, classe + 1])
                                else:
                                    frame_boxes[frame] =[ [frame, x, y, w, h, classe + 1] ]

                    print(f"[INFO] Salvando resultados em {self.output_file_path}")
                    counter = 0
                    with open(self.output_file_path, 'a') as arq:
                        for boxes in frame_boxes.values():
                            for box in boxes:
                                box = [str(i) for i in box]
                                if counter == 0:
                                    arq.write(",".join(box))
                                    counter += 1
                                else: arq.write("\n" +",".join(box))

            except Exception as e:
                print(e)

            finally:
                print("[INFO] Limpando arquivos...")
                # Clear all outputs txt files, except the resume file
                print("[INFO] Removendo diretório temporário.")
                shutil.rmtree(tempdir)
                # Clear yolov5 folder
                print(f"[INFO] Removendo o subdretório yolov5 do experimento {self.experimento}.")
                shutil.rmtree(yv5_path)
                # Final msg
                print("[INFO] Fim da execução.")



        ### Debug mode for Visual Studio Code..
        debug_mode = False
        if debug_mode:
            print("\n\nCurrent dir: ", os.getcwd())
            os.chdir("./RedesNeurais")
            opt.parse_args() # Put the args here
        ###

        run_inference(opt= opt)