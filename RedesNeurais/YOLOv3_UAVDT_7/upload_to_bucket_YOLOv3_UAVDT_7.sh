
#logs/*.ipynb
!gsutil -m cp -r logs/*.ipynb gs://cfdy/Experimentos/YOLOv3_UAVDT_7/logs
#logs/*.txt
!gsutil -m cp -r logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv3_UAVDT_7/logs
#train files
!gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv3_UAVDT_7/YOLOv3_UAVDT_train/YOLOv3_UAVDT_7-04_Mar_2021_16h_31m/ gs://cfdy/Experimentos/YOLOv3_UAVDT_7/YOLOv3_UAVDT_7-04_Mar_2021_16h_31m
