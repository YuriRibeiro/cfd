
#logs/*.ipynb
!gsutil -m cp -r logs/*.ipynb gs://cfdy/Experimentos/YOLOv3_UAVDT_6/logs
#logs/*.txt
!gsutil -m cp -r logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv3_UAVDT_6/logs
#train files
!gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_6/YOLOv3_UAVDT_train/YOLOv3_UAVDT_6-04_Mar_2021_16h_41m/ gs://cfdy/Experimentos/YOLOv3_UAVDT_6/YOLOv3_UAVDT_6-04_Mar_2021_16h_41m
