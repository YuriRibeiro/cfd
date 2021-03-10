
#logs/*.ipynb
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_11/logs/*.ipynb gs://cfdy/Experimentos/YOLOv3_UAVDT_11/logs/
#logs/*.txt
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_11/logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv3_UAVDT_11/logs/
#train files
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_11/YOLOv3_UAVDT_train/YOLOv3_UAVDT_11-09_Mar_2021_02h_12m/ gs://cfdy/Experimentos/YOLOv3_UAVDT_11/YOLOv5_UAVDT_train/YOLOv3_UAVDT_11-09_Mar_2021_02h_12m
