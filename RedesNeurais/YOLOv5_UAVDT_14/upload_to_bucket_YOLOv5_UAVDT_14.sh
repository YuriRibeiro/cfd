
#logs/*.ipynb
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_14/logs/*.ipynb gs://cfdy/Experimentos/YOLOv5_UAVDT_14/logs/
#logs/*.txt
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_14/logs/*.txt gs://cfdy/Experimentos_logs_txt/YOLOv5_UAVDT_14/logs/
#train files
gsutil -m cp -r /home/yuri/Desktop/cfd/RedesNeurais/YOLOv5_UAVDT_14/YOLOv5_UAVDT_train/YOLOv5_UAVDT_14-07_Mar_2021_00h_08m/ gs://cfdy/Experimentos/YOLOv5_UAVDT_14/YOLOv5_UAVDT_train/YOLOv5_UAVDT_14-07_Mar_2021_00h_08m

