
#logs/*.ipynb
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_9/logs/*.ipynb gs://cfdy/Experimentos/YOLOv3_UAVDT_9/logs/
#logs/*.txt
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_9/logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv3_UAVDT_9/logs/
#train files
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_9/YOLOv3_UAVDT_train/YOLOv3_UAVDT_9-09_Mar_2021_02h_11m/ gs://cfdy/Experimentos/YOLOv3_UAVDT_9/YOLOv5_UAVDT_train/YOLOv3_UAVDT_9-09_Mar_2021_02h_11m
