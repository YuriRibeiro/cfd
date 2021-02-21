# INTRODUÇÃO

Nesta pasta são guardados os experimentos que serão apresentados na dissertação.
Cada nome do experimento é formado pelo nome da rede, pelo dataset em que foi treinada, e pelo número único do experimento, ou seja, `Rede_DATASET_X`.


# Resumo dos Experimentos
Nesta seção são descritos as informações mais relevantes do experimento, com o intuito de criar um registro de fácil consulta.

| Experimento    | Épocas | Pré-Treino       | Modelo   | Batch Size |
|----------------|--------|------------------|----------|------------|
| YOLOv5_UAVDT_0 | 50     | COCO (300 Épocas)| YOLOv5 S | 16         |
| YOLOv5_UAVDT_1 | 50     | COCO (300 Épocas)| YOLOv5 M | 16         |
| YOLOv5_UAVDT_3 | 50     | COCO (300 Épocas)| YOLOv5 L | 16         |
| YOLOv5_UAVDT_4 | 50     | COCO (300 Épocas)| YOLOv5 X | 16         |
| YOLOv5_UAVDT_5 | 50     | NÃO              | YOLOv5 S | 16         |
| YOLOv5_UAVDT_6 | 50     | NÃO              | YOLOv5 M | 16         |
| YOLOv5_UAVDT_7 | 50     | NÃO              | YOLOv5 L | 16         |
| YOLOv5_UAVDT_8 | 50     | NÃO              | YOLOv5 X | 16         |
<br>

## Comentários
Nesta seção, são expostas as principais características exploradas em cada experimento.

- YOLOv5_UAVDT_X  (X = 1,2,...,8)
  - Vídeos do Dataset:
    - Conténdo todas as situações (dia, noite, etc.)
  - Pré-processamento das imagens
    - Zonas de ignore borradas fortemente com filtro gaussiano.
  - Características da rede:
    - Resolução de entrada: 640
    - Otimizador: adam

<br>

## Resultados Detecção e Rastreamento (MOT)
<br>

| Experimento    | mAP    | AP@0.5 | AP@0.75 | Curva PR |
|----------------|--------|--------|---------|----------|
| YOLOv5_UAVDT_0 |        |        |         |          |
| YOLOv5_UAVDT_1 |        |        |         |          |
| YOLOv5_UAVDT_3 |        |        |         |          |
| YOLOv5_UAVDT_4 |        |        |         |          |
| YOLOv5_UAVDT_5 |        |        |         |          |
| YOLOv5_UAVDT_6 |        |        |         |          |
| YOLOv5_UAVDT_7 |        |        |         |          |
| YOLOv5_UAVDT_8 |        |        |         |          |

<sub>mAP = AP@0.5:0.95</sub>
<br>
<br>


| Experimento    | MOTA   | MOTP   | MT      |ML       |FMT      |IDSW     | FP      |TP       |
|----------------|--------|--------|---------|---------|---------|---------|---------|---------|
| YOLOv5_UAVDT_0 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_1 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_3 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_4 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_5 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_6 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_7 |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_8 |        |        |         |         |         |         |         |         |
