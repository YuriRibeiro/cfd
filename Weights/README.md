# Introdução
Pasta para guardar pesos pré-treinados.

# YoloV5
Pesos do YOLOv5 rev4

-Pesos pré-treinados no COCO Dataset por 300 épocas:
  - yolov5s.pt
  - yolov5m.pt
  - yolov5l.pt
  - yolov5x.pt

```bash
revnum="4"; outpdir="YOLOv5_rev${revnum}"; mkdir ${outpdir} ; weights_file="yolov5s.pt"  && curl -L https://storage.googleapis.com/cfdy/Weights/YOLOv5_rev${revnum}/COCO/${weights_file} -o ./${outpdir}/${weights_file};
```

# YoloV3
Pesos do YOLOv3

-Pesos pré-treinados no COCO Dataset por 300 épocas:
  - yolov3.pt
  - yolov3-tiny.pt
  - yolov3-spp.pt


```bash
outpdir="YOLOv3"; mkdir ${outpdir} ; weights_file="yolov3.pt"  && curl -L https://storage.googleapis.com/cfdy/Weights/YOLOv3/COCO/${weights_file} -o ./${outpdir}/${weights_file};
```



