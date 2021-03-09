
#logs/*.ipynb
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_13/logs/*.ipynb gs://cfdy/Experimentos/YOLOv5_UAVDT_13/logs/
#logs/*.txt
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_13/logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv5_UAVDT_13/logs/
#train files
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_13/YOLOv5_UAVDT_train/YOLOv5_UAVDT_13-07_Mar_2021_00h_07m/ gs://cfdy/Experimentos/YOLOv5_UAVDT_13/YOLOv5_UAVDT_train/YOLOv5_UAVDT_13-07_Mar_2021_00h_07m
