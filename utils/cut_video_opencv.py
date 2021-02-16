# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:33:25 2020

@author: Ribeiro

Corta pedaços de vídeos entre X e Y segundos.

Exemplo:
Cortar entre 4 e 10 segundos do vídeo_original.avi e
salvar em video_editado.avi.

python cut_video_opencv.py -i video_original.avi -o video_editado.avi -s 10 -f 15
"""

import cv2
import numpy as np
import argparse

# =============================================================================
#  PARAMETERS

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
                help="path to input video")
ap.add_argument("-o", "--output", required=True, type=str,
                help="path to output video")
ap.add_argument("-s", "--start_second", required=True, type=int,
                help="Initial second to cut the video")
ap.add_argument("-f", "--final_second", required=True, type=int,
                help="Final second to cut the video")
args = vars(ap.parse_args())

original_video_config = {"file path": args["input"],
                         "fps": 0.0,
                         "frame shape": (0, 0, 0)
                         }

dst_video_configs = {"file path": args["output"],
                     "start second": args["start_second"],
                     "final second": args["final_second"],
                     "time length": args["final_second"] - args["start_second"]
                     }
if dst_video_configs["time length"] <= 0:
    raise Exception("Time length lower or equal zero.")

# Buffer size
buffer_config = {"size": 50}

# =============================================================================
# Start the VideoCapture of the original file

original_video = cv2.VideoCapture(original_video_config["file path"])

# Video Properties (fps)
original_video_config["fps"] = original_video.get(cv2.CAP_PROP_FPS)
ret, img = original_video.read()
if ret is False:
    raise Exception("File not found or problems with the video.")
original_video_config["frame shape"] = img.shape

# Initialize the image buffer array
img_buffer = np.zeros((buffer_config["size"],
                      *original_video_config["frame shape"]
                       ),
                      dtype=np.uint8
                      )
img_buffer_idx = 0

# Initialize the video file writer object
output = cv2.VideoWriter(dst_video_configs["file path"],
                         cv2.VideoWriter_fourcc(*'DIVX'),
                         original_video_config["fps"],
                         original_video_config["frame shape"][:2][::-1])
try:
    print("[INFO] Starting video edition.")
    analysed_frame_number = 0
    processed_frame_number = 0
    total_frames_to_process = int(original_video_config["fps"]*dst_video_configs["time length"])
    while True:
        video_time = analysed_frame_number / original_video_config["fps"]
        analysed_frame_number += 1

        if video_time < dst_video_configs["start second"]:
            ret, img = original_video.read()
            if ret is False:
                break
            continue
        elif video_time > dst_video_configs["final second"]:
            break
        else:
            pass

        processed_frame_number += 1
        img_buffer[img_buffer_idx] = img.copy()
        img_buffer_idx += 1

        if img_buffer_idx % buffer_config["size"] == 0:
            info = "%d of approx. %d." % (processed_frame_number,
                                          total_frames_to_process)
            print("[INFO] Processed frames: ", info)
            for idx in range(buffer_config["size"]):
                output.write(img_buffer[idx])
            img_buffer_idx = 0

        ret, img = original_video.read()
        if ret is False:
            break

    # Write the last frames
    for idx in range(img_buffer_idx):
        output.write(img_buffer[idx])

    print("[INFO] Video edition finished.")

except Exception as err:
    print(err)
    output.release()
