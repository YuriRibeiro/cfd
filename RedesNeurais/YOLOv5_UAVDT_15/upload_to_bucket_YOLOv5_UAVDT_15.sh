
#logs/*.ipynb
!gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_15/logs/*.ipynb gs://cfdy/Experimentos/YOLOv5_UAVDT_15/logs/
#logs/*.txt
!gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_15/logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv5_UAVDT_15/logs/
#train files
!gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_15/YOLOv5_UAVDT_train/YOLOv5_UAVDT_15-07_Mar_2021_00h_11m/ gs://cfdy/Experimentos/YOLOv5_UAVDT_15/YOLOv5_UAVDT_train/YOLOv5_UAVDT_15-07_Mar_2021_00h_11m
