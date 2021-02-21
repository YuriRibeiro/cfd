# -*- coding: UTF-8 -*-
#%%
"""
Script Python para organizar o UAVDT Dataset na forma requerida pelo módulo YOLOv5.
"""
from glob import glob
from os import system, getcwd

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    YOLOv5 Code to Resize Image. Extracted from the source code.
    yolov5/utils/datasets.py
    """
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def UAVDT_To_YoloV5_Dataset_Format(conjunto="Treino", movies_to_transform=[],
                    dest="Editado", uavdt_path = "../Datasets/UAVDT", coco_labels = False,
                    resize = 0, nosymlink = False):
    """
    Transformar imagens para treinamento e teste da rede YoloV5 a partir do UAVDT Dataset.

    Parâmetros:
        conjunto: "Treino" ou "Teste".
        movies_to_transform: Lista com títulos dos clipes a serem traansformados.
        dest: Pasta de destino do dataset editado.
        uavd_path: Path do dataset original.
        coco_labels: classes correspondem à numeração do COCO Dataset.
        resize: Redimensior as imagens no destino.
    """
    print(f"[INFO] Iniciando processo de transformação para {conjunto}.")
    num_of_analised_files = 0
    num_of_transformed_files = 0
    num_of_images_without_labels = 0
    num_of_skipped_images = 0
    num_of_labels = 0
    num_total_of_labels = 0

    # Dimensões originais das imagens
    original_width = 1024
    original_height = 540

    # Definir Caminhos para Pastas de Treinamento e Teste no Destino:
    train_images_folder = f"{dest}/train/images"
    train_labels_folder = f"{dest}/train/labels"

    test_images_folder = f"{dest}/test/images"
    test_labels_folder = f"{dest}/test/labels"

    print(f"""
    CWD: {getcwd()}
    Train images folder: {train_images_folder}
    Train labels folder: {train_labels_folder}
    Test images folder: {test_images_folder}
    Test labels folder: {test_labels_folder}
    """)

    if conjunto == "Treino":
        # Limpar a pasta de destino:
        _ = system(f"rm -rf {train_images_folder}")
        _ = system(f"rm -rf {train_labels_folder}")
        # Criar pastas limpas para treino
        _ = system(f"mkdir -p {train_images_folder}")
        _ = system(f"mkdir -p {train_labels_folder}")
    
    elif conjunto == "Teste":    
        # Limpar a pasta de destino
        _ = system(f"rm -rf {test_images_folder}")
        _ = system(f"rm -rf {test_labels_folder}")
        # Criar pastas limpas para teste:
        _ = system(f"mkdir -p {test_images_folder}")
        _ = system(f"mkdir -p {test_labels_folder}")

    for movie in movies_to_transform:
        jpg_files = glob(f"{uavdt_path}/UAV-benchmark-M/{movie}/*.jpg")

        # Criar inventário de labels:
        # Label[0] foi ignorada, pois contagem começa a partir da img000001.jpg
        # Cada frame (índice da lista) possuíra uma string com todos os labels,
        # no formato requisitado pela YoloV5:

        #  YoloV5 Data Format:
        #  class x_center y_center width height
        #
        #  As coordenadas devem ser normalizadas entre 0 e 1
        #  Box coordinates must be in normalized xywh format (from 0 - 1).
        #  If your boxes are in pixels, divide x_center and width by image width,
        #  and y_center and height by image height.
        #  Class numbers are zero-indexed (start from 0).

        #  UAV-M Data format:
        #  <frame_index>,<target_id>,<bbox_left>,<bbox_top>,
        #  <bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
        #  class: 1 = Car; 2 = Truck; 3 = Bus.
        num_of_labels = 0
        num_of_images = len(jpg_files) 
        labels = [""]*(num_of_images+1)

        print(f"[INFO] Analisando imagens e labels do clipe: {movie} ({num_of_images} frames). ", end="")

        with open(f"{uavdt_path}/UAV-benchmark-MOTD_v1.0/GT/{movie}_gt_whole.txt") as f:
            for line in f.readlines():
                line = line.split(",")
                frame = int(line[0])
                if frame > num_of_images:
                    # Skip image, if we are working with a small version of the dataset(debug purposes)
                    num_of_skipped_images += 1
                    continue
                num_of_labels += 1
                
                classe = int(line[-1]) - 1 # Zero indexed in yolov5 format.
                if coco_labels == True:
                    if classe == 0: classe = 2
                    elif classe == 1: classe = 7
                    elif classe == 2: classe = 5

                if resize == 0:
                    bbox_width = float(line[4]) / original_width
                    bbox_height = float(line[5]) / original_height

                    xc = (float(line[2]) / original_width) + (bbox_width / 2)
                    yc = (float(line[3]) / original_height) + (bbox_height / 2)
                else:
                    bbox_width = float(line[4])
                    bbox_height = float(line[5])
                    
                    xc = float(line[2]) + bbox_width / 2
                    yc = float(line[3]) + bbox_height / 2

                yv5_line = f"{classe} {xc} {yc} {bbox_width} {bbox_height}\n"
                
                labels[frame] = labels[frame] + yv5_line
        
        print(f"Quantidade de Labels: {num_of_labels}.")
        num_total_of_labels += num_of_labels

        for jpg_file in jpg_files:
            # Informações da imagem:
            original_name = jpg_file.split("/")[-1]
            frame_number = int(original_name.split(".")[0][3:].strip("0"))
            new_file_name = movie+"_"+original_name

            # Checar a imagem para verificar se tem label:
            if len(labels[frame_number]) == 0:
                num_of_images_without_labels += 1
                continue
            
            if resize != 0:
                img = cv2.imread(jpg_file)
                resized_img, ratio, pad = letterbox(img)
                if conjunto == "Treino":
                    cv2.imwrite(train_images_folder+"/"+new_file_name, resized_img)
                elif conjunto == "Teste":
                    cv2.imwrite(test_images_folder+"/"+new_file_name, resized_img)

                h,w = resized_img.shape[:2]
                rh, rw = ratio[0], ratio[1]

                # Adjust Labels:
                old_labels = labels[frame_number].split("\n")
                new_labels = ""
                for img_labels in old_labels:
                    if img_labels == "": continue
                    (classe, xc, yc, bbox_width, bbox_height) = img_labels.split(" ")
                    classe, xc, yc = int(classe), float(xc), float(yc)
                    bbox_height, bbox_width = float(bbox_height), float(bbox_width)
                    
                    # Ajustar novos labels com padding e ratio:
                   # print(xc, yc, rw, rh, pad[0], pad[1], w, h)
                    xc = (pad[0] + rw * xc) / w # pad width
                    yc = (pad[1] + rh * yc) / h # pad height
                    bbox_width =  (rw*bbox_width) / w
                    bbox_height = (rh*bbox_height) / h
                   # print("xc,", xc,yc, bbox_width, bbox_height)
                    new_labels += f"{classe} {xc} {yc} {bbox_width} {bbox_height}\n"
               # print("new", new_labels)
                labels[frame_number] = new_labels

            # Criar um hard link para cada imagem
            if resize == 0:
                copy_type = "-a" if nosymlink else "-al"
                if conjunto == "Treino":
                    system(f"cp {copy_type} {jpg_file} {train_images_folder}/{new_file_name}")
                elif conjunto == "Teste":
                    system(f"cp {copy_type} {jpg_file} {test_images_folder}/{new_file_name}")

            # Criar um label para essa imagem:
            label = labels[frame_number][:-1] #excluir \n da última linha
            label_file_name = new_file_name[:-3]+"txt"

            if conjunto == "Treino":
                system(f"echo \"{label}\" > {train_labels_folder}/{label_file_name}")
            elif conjunto == "Teste":
                system(f"echo \"{label}\" > {test_labels_folder}/{label_file_name}")
            
            num_of_transformed_files += 1

        num_of_analised_files += len(jpg_files)
    
    final_msg = f"[INFO] Processo finalizado para o conjunto de {conjunto}.\n" + \
                f" Total de Imagens Analisadas: {num_of_analised_files}.\n" + \
                f" Total de Imagens Transformadas: {num_of_transformed_files}.\n" + \
                f" Total de Imagens Sem Labels: {num_of_images_without_labels}.\n" +\
                f" Total de Imagens Puladas: {num_of_skipped_images}.\n" +\
                f" Total de Labels: {num_total_of_labels}.\n"
                
    print(final_msg)


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description="UAVDT to YoloV5 Dataset Organization.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--conjunto", type=str, default="", nargs=1,
                    help="Selecionar subconjunto de imagens a serem transformadas.",
                    choices=["treinoEteste", "treino", "teste"])
    group.add_argument("--videos", type=str, default="", nargs="+",
                    help="Selecionar uma lista de vídeos específicos.")
    parser.add_argument("--coco-labels", action="store_true",
                        help="As classes serão numeradas de acordo com os rótulos do COCO Dataset.")
    parser.add_argument("--dest-folder", type=str, nargs=1, default="../Datasets/YOLOv5_UAVDT",
                        help="Pasta de destino da saída.")
    parser.add_argument("--uavdt-path", type=str, nargs=1, default="../Datasets/UAVDT",
                        help="Pasta de destino da saída.")
    parser.add_argument("--resize", type=int, nargs=1, default=0,
                        help="Redimensionar as imagens no destino.")
    parser.add_argument("--nosymlink", action='store_true',
                        help="Salvar as imagens sem utilizar sym links.")
    opt = parser.parse_args()

    convert_args = lambda x: x[0] if isinstance(x, (list)) else x
    dest = convert_args(opt.dest_folder)
    uavdt_path =convert_args(opt.uavdt_path)
    conjunto = convert_args(opt.conjunto)
    coco_labels = opt.coco_labels
    videos = opt.videos
    resize = opt.resize
    nosymlink = opt.nosymlink

    if resize != 0:
        import cv2
        import numpy as np
    
    movies_treino = ["M0101", "M0201", "M0202", "M0204", \
        "M0206", "M0207", "M0210", "M0301", "M0401", "M0402", \
        "M0501", "M0603","M0604", "M0605", "M0702", "M0703", \
        "M0704", "M0901", "M0902", "M1002", "M1003", "M1005", \
        "M1006", "M1008", "M1102", "M1201", "M1202", "M1304", \
        "M1305", "M1306"]

    movies_teste = ["M0203", "M0205", "M0208", "M0209", "M0403", "M0601", \
            "M0602", "M0606", "M0701", "M0801", "M0802", "M1001", "M1004", \
            "M1007", "M1009", "M1101", "M1301", "M1302", "M1303", "M1401"]
    
    # TimeStamp
    wallClock = datetime.datetime.now()
    wallClock = wallClock.strftime('%d_%m_%Y_%H_%M_%S')
    print(f"[INFO] Etiqueta de Tempo: {wallClock}.")

    if conjunto == "treino" or conjunto == "treinoEteste":
        UAVDT_To_YoloV5_Dataset_Format(conjunto="Treino",
                                     movies_to_transform=movies_treino,
                                     dest=dest,
                                     uavdt_path = uavdt_path,
                                     coco_labels=coco_labels,
                                     resize=resize,
                                     nosymlink=nosymlink)
        
    if conjunto == "teste" or conjunto == "treinoEteste":
        UAVDT_To_YoloV5_Dataset_Format(conjunto="Teste",
                                     movies_to_transform=movies_teste,
                                     dest=dest,
                                     uavdt_path = uavdt_path,
                                     coco_labels=coco_labels,
                                     resize=resize,
                                     nosymlink=nosymlink)
        
    if  isinstance(videos, (list)):
        for title in videos:
            if title in movies_treino: conj = "Treino"
            elif title in movies_teste: conj = "Teste"
            else: print(f"Vídeo não identificado: %{title}.")

            UAVDT_To_YoloV5_Dataset_Format(conjunto=conj,
                                        movies_to_transform=[title],
                                        dest=dest,
                                        uavdt_path = uavdt_path,
                                        coco_labels=coco_labels,
                                        resize=resize,
                                        nosymlink=nosymlink)

