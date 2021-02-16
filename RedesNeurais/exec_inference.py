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
import optparse
from PIL import Image as PILImage
from glob import glob


# Absolute Weight Paths "Database"
dropbox = os.path.expanduser("~/Dropbox/YoloV5_Runs/Experimentos/")
weight_paths = {"YOLOv5_UAVDT_0" : dropbox + "YOLOv5_UAVDT_1_28_August_2020_14h_26m_40s/runs/exp0/weights/best.pt"}


# Option Parser
opt = optparse.OptionParser()
opt.add_option('-o', '--output', action='store', dest="output_folder", default="")
opt.add_option('-i', '--input', action='store', dest="input_folder", default="")
opt.add_option('-e', '--experiment', action='store', dest="experiment", default="")
opt, remainder = opt.parse_args()

if opt.output_folder == "" or opt.input_folder == "" or opt.experiment == "":
    raise Exception("The input, output or experiment is an empty strings.")

# Main Function
def run_inference(opt):
    """
    This function makes the things happen.
    """
    # options
    experimento = opt.experiment
    input_folder = opt.input_folder
    output_folder = opt.output_folder

    # Check if the folders paths terminate with a os.sep. If don't, force it.
    if not input_folder[-1] == os.sep: input_folder += os.sep
    if not output_folder[-1] == os.sep: output_folder += os.sep

    # Check if input folder is a valid folder, i.e, it exists.
    if not os.path.isdir(input_folder):
        raise Exception(f'O caminho do input_folder "{input_folder}" não é um diretório.')
    
    # Check for files/ folders inside input path
    objects = glob(input_folder + "*")
    
    # If there is files and folders, raise an Exception.
    # The folder should contain only files or only folders.
    package = os.path.isdir(objects[0])
    for obj in objects:
        # not XOR to check if all file are equals.
        if not (package ^ os.path.isdir(obj)):
            pass
        else:
            raise Exception(f"There is files and folders mixed together inside '{input_folder}'. That isn't permitted.")

    videos_paths = []

    if package:
        for directory in objects:
            videos_paths.append(directory)
    else:
        videos_paths.append(input_folder)

    # Analysis of each video
    for video_path in videos_paths:
        
        # Check if the output folder exists. If doesn't, create it.
        if not os.path.exists(output_folder):
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        else:
            # If it exists, check if it is not a file.
            if not os.path.isdir(output_folder):
                raise Exception("Output path is not a directory.")
        
        # Create the output file path
        video_name = video_path.split(os.sep)[-2]
        output_file_path = output_folder + video_name + "_results.txt"

        # If for some reason the output file already exists, remove it.
        # Here I use os.rmdir because if the output_file_path already exists,
        # then I'll remove it. If it's actually a directory, if not empty,
        # it'll raise an Exception.
        if os.path.isdir(output_file_path): os.rmdir(output_file_path)
        if os.path.isfile(output_file_path): os.remove(output_file_path)
        
        # Check experiment neural network model name:
        network = experimento.split("_")[0]

        if network == "YOLOv5":
            # Remove old yolov5 files and replace by a new one copy.
            yv5_path = os.path.join(".", experimento, "yolov5")
            if os.path.exists(yv5_path):
                os.remove(yv5_path)
            os.symlink(os.path.join("..","..","Submodules","yolov5",), yv5_path)

            # Create a temporary directory to store intermediate results
            tempdir = tempfile.mkdtemp()
            print("\nDiretório temporário: ", tempdir)

            # Now, run the inference
            weights_file_path = weight_paths[experimento]
            detect_py_path = os.path.join(yv5_path, "detect.py")

            shell_command = ["python", detect_py_path,
                            "--weights", weights_file_path,
                            "--img", "640",
                            "--conf", "0.5",
                            "--save-txt",
                            "--source", video_path,
                            "--output", tempdir]

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

            print(f"[INFO] Salvando resultados em {output_file_path}")
            counter = 0
            with open(output_file_path, 'a') as arq:
                for boxes in frame_boxes.values():
                    for box in boxes:
                        box = [str(i) for i in box]
                        if counter == 0:
                            arq.write(",".join(box))
                            counter += 1
                        else: arq.write("\n" +",".join(box))

            # Clear all outputs txt files, except the resume file
            print("[INFO] Removendo diretório temporário.")
            shutil.rmtree(tempdir)

            # Clear yolov5 folder
            print(f"[INFO] Removendo o subdretório yolov5 do experimento {experimento}.")
            os.remove(yv5_path)

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