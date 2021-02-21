# -*- coding: UTF-8 -*-
#%%
"""
Script Python para organizar o UAVDT Dataset na forma requerida pelo Gcloud AutoML Vision.

Exemplo de execução:
cd utils
python uavdt_to_gcloud.py --conjunto treinoEteste
"""
from glob import glob
from os import system
from imgs_to_video_opencv import convert_img_to_video


def UAVDT_To_Gcloud_Dataset_Format(movies_to_transform=[],
        dest="./Editado_gcloud", uavdt_path = "../Datasets/UAVDT",
        gscloud_videos_folder="gs://cfdy/gcloud_automl_vision/uavdt",
        dest_video_format="avi", dest_video_fps = 30,
        convert_to_video = False, clear_dest_folder_before=False):
    """
    Transformar imagens para treinamento e teste da rede YoloV5 a partir do UAVDT Dataset.

    Parâmetros:
        movies_to_transform: Lista com títulos dos clipes a serem traansformados.
        dest: Pasta de destino do dataset editado.
        uavd_path: Path do dataset original.

    """
    print(f"[INFO] Iniciando processo de transformação.")
    num_of_analised_files = 0
    num_of_analised_labels = 0
    num_of_frames_without_labels = 0
    total_video_time = 0


    # Dimensões originais das imagens
    original_width = 1024
    original_height = 540

    # Definir Caminhos para Pastas de Treinamento e Teste no Destino:
    train_video_folder = f"{dest}/train"
    test_video_folder = f"{dest}/test"

    # Criar pastas limpas para treino e teste
    if clear_dest_folder_before:
        _ = system(f"rm -rf {dest}")
    _ = system(f"mkdir -p {train_video_folder}")
    _ = system(f"mkdir -p {test_video_folder}")

    include_test_labels = False
    include_train_labels = False
    destination_video_fps = 30


    for movie in movies_to_transform:
        if movie in movies_treino:
            video_images_folder_path = f"{uavdt_path}/UAV-benchmark-M/{movie}/"
            include_train_labels = True
            conjunto = "Treino"
            conjunto_path = "train"
            if convert_to_video:
                destination_video_path = f"{train_video_folder}/{movie}.avi"
                convert_img_to_video(video_images_folder_path,
                                    destination_video_fps,
                                    destination_video_path)

        if movie in movies_teste:
            video_images_folder_path = f"{uavdt_path}/UAV-benchmark-M/{movie}/"
            include_test_labels = True
            conjunto = "Teste"
            conjunto_path = "test"
            if convert_to_video:
                destination_video_path = f"{test_video_folder}/{movie}.avi"
                convert_img_to_video(video_images_folder_path,
                                    destination_video_fps,
                                    destination_video_path)

        jpg_files = glob(f"{uavdt_path}/UAV-benchmark-M/{movie}/*.jpg")
        num_of_images = len(jpg_files)
        num_of_analised_files += num_of_images

        # Criar inventário de labels:
            # Label[0] ignored.
            #  gcloud Data Format:
#video_uri,label,instance_id,time_offset,x_relative_min,y_relative_min,
#x_relative_max,y_relative_min,x_relative_max,y_relative_max,x_relative_min,y_relative_max

            #Example:
            # gs://folder/video1.avi,car,,12.90,0.8,0.2,,,0.9,0.3,,
            # gs://folder/video1.avi,bike,,12.50,0.45,0.45,,,0.55,0.55,,

        
            #  UAV-M Data format:
            #  <frame_index>,<target_id>,<bbox_left>,<bbox_top>,
            #  <bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
            #  class: 1 = Car; 2 = Truck; 3 = Bus.

        labels = [""]*(len(jpg_files)+1)
        classes = {1:"Car", 2:"Bus", 3:"Truck"}

        with open(f"{uavdt_path}/UAV-benchmark-MOTD_v1.0/GT/{movie}_gt_whole.txt") as f:
            for line in f.readlines():
                num_of_analised_labels += 1
                line = line.split(",")
                frame = int(line[0])
                if frame > num_of_images:
                    # Skip frames if operating with the reduced dataset (debug purposes)
                    num_of_frames_without_labels += 1
                    continue

                time_offset = frame / dest_video_fps
                
                classe_nome = classes[int(line[-1])]

                x = float(line[2])
                y = float(line[3])
                bbox_width = float(line[4])
                bbox_height = float(line[5])

                xmin =  x / original_width
                xmax = (x + bbox_width)/ original_width

                ymin = y / original_height
                ymax = (y + bbox_height) / original_height

                gcloud_line = f"{gscloud_videos_folder}/{conjunto_path}/{movie}.{dest_video_format},{classe_nome},,{time_offset:.2f},{xmin},{ymin},,,{xmax},{ymax},,\n"
                
                labels[frame] = labels[frame] + gcloud_line
            
        
            # Append (or create, if it does not exists yet) csv file:
            if conjunto == "Treino":
                labels_dest = f"{dest}/train_labels.csv"
            if conjunto == "Teste":
                labels_dest = f"{dest}/test_labels.csv"

            with open(labels_dest,'a') as f:
                for idx,line in enumerate(labels):
                    f.write(line)
    
    with open(f"{dest}/labels_files_paths.csv","w") as f:
        if include_train_labels:
            f.write(f"TRAIN,{gscloud_videos_folder}/train_labels.csv")
        if include_test_labels:
            f.write(f"\nTEST,{gscloud_videos_folder}/test_labels.csv")

    total_video_time += num_of_analised_files/dest_video_fps
    
    final_msg = f"[INFO] Processo de transformação finalizado.\n" + \
                f" Total de Imagens Analisadas e Transformadas: {num_of_analised_files}." + \
                f" Total de Imagens Sem Labels: {num_of_frames_without_labels}." + \
                f" Total de Labels: {num_of_analised_labels}." + \
                f" Total de tempo dos vídeos de saída: {total_video_time} segundos."
    print(final_msg)


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description="UAVDT to Gcloud Dataset Organization.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--conjunto", type=str, default="", nargs=1,
                    help="Selecionar subconjunto de imagens a serem transformadas.",
                    choices=["treinoEteste", "treino", "teste"])
    group.add_argument("--videos", type=str, default="", nargs="+",
                    help="Selecionar uma lista de vídeos específicos.")
    parser.add_argument("--dest-folder", type=str, nargs=1, default="../Datasets/UAVDT_Gcloud",
                        help="Pasta de destino da saída.")
    parser.add_argument("--uavdt-path", type=str, nargs=1, default="../Datasets/UAVDT",
                        help="Pasta do Dataset.")
    parser.add_argument("--gcloud-bucket", type=str, nargs=1, default="gs://cfdy/gcloud_automl_vision/uavdt",
                        help="Pasta de destino no google cloud.")
    parser.add_argument("--convert-to-video", "-ctv", action="store_true",
                        help="Converter as imagens para vídeos.")
    parser.add_argument("--clear-output-before", "-cob", action="store_true",
                        help="Remover todos os arquivos na pasta de output antes de iniciar o algoritmo.")
    
    opt = parser.parse_args()

    convert_args = lambda x: x[0] if isinstance(x, (list)) else x
    dest = convert_args(opt.dest_folder)
    uavdt_path =convert_args(opt.uavdt_path)
    conjunto = convert_args(opt.conjunto)
    gcloud_bucket = convert_args(opt.gcloud_bucket)
    convert = opt.convert_to_video
    clear = opt.clear_output_before
    videos = opt.videos
    
    movies_treino = ["M0101", "M0201", "M0202", "M0204", \
        "M0206", "M0207", "M0210", "M0301", "M0401", "M0402", \
        "M0501", "M0603","M0604", "M0605", "M0702", "M0703", \
        "M0704", "M0901", "M0902", "M1002", "M1003", "M1005", \
        "M1006", "M1008", "M1102","M1201", "M1202", "M1304", \
        "M1305", "M1306"] #

    movies_teste = ["M0203", "M0205", "M0208", "M0209", "M0403", "M0601", \
            "M0602", "M0606", "M0701", "M0801", "M0802", "M1001", "M1004", \
            "M1007", "M1009", "M1101", "M1301", "M1302", "M1303", "M1401"]
    

    if conjunto == "treino":
        videos = movies_treino
    if conjunto == "teste":
        videos = movies_teste
    if conjunto == "treinoEteste":
        videos = movies_treino + movies_teste
    else:
        raise Exception("Conjunto não identificado.")


    # TimeStamp
    wallClock = datetime.datetime.now()
    wallClock = wallClock.strftime('%d_%m_%Y_%H_%M_%S')

    print(f"[INFO] Etiqueta de Tempo INÍCIO: {wallClock}.")
    
    for title in videos:
        if (title in movies_treino) or (title in movies_teste):
            pass
        else:
            raise Exception(f"Vídeo não identificado: %{title}.")
    
    UAVDT_To_Gcloud_Dataset_Format(movies_to_transform=videos,
                                dest=dest,
                                uavdt_path = uavdt_path,
                                gscloud_videos_folder=gcloud_bucket,
                                convert_to_video = False,
                                clear_dest_folder_before=False)

    wallClock = datetime.datetime.now()
    wallClock = wallClock.strftime('%d_%m_%Y_%H_%M_%S')
    print(f"[INFO] Etiqueta de Tempo FIM: {wallClock}.")