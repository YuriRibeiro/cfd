# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:33:25 2020

@author: Ribeiro

GRAVA VIDEOS EM FORMATO AVI A PARTIR DE IMAGENS

# Supor que todas as imagens possuem as mesmas dimensões e mesmo formato de g_
ravação (jpg, png, etc.).
# Assumir que as imagens são RGB.
"""

import cv2
import numpy as np
from glob import glob

# =============================================================================
#  PARAMETERS
images_folder   = 'D:\\Datasets\\UAV-benchmark-M\\UAV-benchmark-M\\M0209\\'
destination_video_fps = 30 #(same as the original video)
destination_video_name = "destination_video"+".avi"
# =============================================================================


# Pick all the image files that will compose the video:
images_paths  = glob(images_folder + '*.jpg')
num_of_frames = len(images_paths)

# Organize the list of paths in ascending order (0,1,2,3, ...)
# sort in place. key = image number.
images_paths.sort(key = lambda x: int(x.split("\\")[-1][3:-4]))

# Buffer size
buffer_size = 50

# Pick the first image dimensions:
# Suppose all the images have the same properties, such as dimensions,png, etc.
img  = cv2.imread(images_paths[0])
height, width, layers = img.shape
size = (width,height)

# Initialize the image array to pass to video file writer
img_array = np.zeros((buffer_size, height, width, layers), dtype = np.uint8)
# Image array index:
imga_idx  = 0

# Initialize the video file writer object
out = cv2.VideoWriter(destination_video_name,
                      cv2.VideoWriter_fourcc(*'DIVX'),
                      destination_video_fps,
                      size)
try:    
    for filepath in images_paths:
        img_array[imga_idx] = cv2.imread(filepath)
        imga_idx += 1
        
        if (imga_idx % buffer_size == 0):
            for i in range(buffer_size):
                out.write(img_array[i])
            imga_idx = 0
    
    # Write the last frames
    for i in range(imga_idx+1):
        out.write(img_array[i])
    
    # Close video writer
    out.release()

except Exception as e:
    print(e)
    out.release()