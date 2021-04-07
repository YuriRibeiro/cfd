# INTRODUÇÃO

Nesta pasta são guardados os experimentos que serão apresentados na dissertação.
Cada nome do experimento é formado pelo nome da rede, pelo dataset em que foi treinada, e pelo número único do experimento, ou seja, `Rede_DATASET_X`.

## Download dos Resultados

Para sincronizar todos os resultados disponíveis na nuvem: `cd` para esta pasta e `gsutil -m rsync -r gs://cfdy/Experimentos/ ./`.Para resultados específicos `exp="YOLOv5_UAVDT_0"; gsutil -m rsync -r gs://cfdy/Experimentos/$exp ./$exp`.

# Resumo dos Experimentos
Nesta seção são descritos as informações mais relevantes do experimento, com o intuito de criar um registro de fácil consulta.

| Experimento      | Épocas | Pré-Treino       | Modelo   | Batch Size | Hyperparams|
|------------------|--------|------------------|----------|------------|------------|
| YOLOv5_UAVDT_0   | 50     | COCO (300 Épocas)| YOLOv5 S | 16         | Scratch    |
| YOLOv5_UAVDT_1   | 50     | COCO (300 Épocas)| YOLOv5 M | 16         | Scratch    |
| YOLOv5_UAVDT_2   | 50     | COCO (300 Épocas)| YOLOv5 L | 8          | Scratch    |
| YOLOv5_UAVDT_3   | 50     | COCO (300 Épocas)| YOLOv5 X | 8          | Scratch    |
|                  |        |                  |          |            |            |
| YOLOv5_UAVDT_4   | 50     |    YOLOv5_UAVDT_0| YOLOv5 S | 16         | Finetune   |
| YOLOv5_UAVDT_5   | 50     | YOLOv5_UAVDT_1   | YOLOv5 M | 16         | Finetune   |
| YOLOv5_UAVDT_6   | 50     | YOLOv5_UAVDT_2   | YOLOv5 L | 8          | Finetune   |
| YOLOv5_UAVDT_7   | 50     | YOLOv5_UAVDT_3   | YOLOv5 X | 8          | Finetune   |
|                  |        |                  |          |            |            |
| YOLOv5_UAVDT_8   | 50     | YOLOv5_UAVDT_4   | YOLOv5 S | 16         | Finetune   |
| YOLOv5_UAVDT_9   | 50     | YOLOv5_UAVDT_5   | YOLOv5 M | 16         | Finetune   |
| YOLOv5_UAVDT_10  | 50     | YOLOv5_UAVDT_6   | YOLOv5 L | 8          | Finetune   |
| YOLOv5_UAVDT_11  | 50     | YOLOv5_UAVDT_7   | YOLOv5 X | 8          | Finetune   |


<br>


| Experimento      | Épocas | Pré-Treino       | Modelo     | Batch Size | Hyperparams|
|------------------|--------|------------------|------------|------------|------------|
| YOLOv3_UAVDT_0   | 50     | COCO (300 Épocas)| YOLOv3-tiny| 16         | Scratch    |
| YOLOv3_UAVDT_1   | 50     | COCO (300 Épocas)| YOLOv3     | 16         | Scratch    |
| YOLOv3_UAVDT_2   | 50     | COCO (300 Épocas)| YOLOv3-spp | 16         | Scratch    |
|                  |        |                  |            |            |            |
| YOLOv3_UAVDT_3   | 50     | YOLOv3_UAVDT_0   | YOLOv3-tiny| 16         | Finetune   |
| YOLOv3_UAVDT_4   | 50     | YOLOv3_UAVDT_1   | YOLOv3     | 16         | Finetune   |
| YOLOv3_UAVDT_5   | 50     | YOLOv3_UAVDT_2   | YOLOv3-spp | 16         | Finetune   |
|                  |        |                  |            |            |            |
| YOLOv3_UAVDT_6   | 50     | YOLOv3_UAVDT_3   | YOLOv3-tiny| 16         | Finetune   |
| YOLOv3_UAVDT_7   | 50     | YOLOv3_UAVDT_4   | YOLOv3     | 16         | Finetune   |
| YOLOv3_UAVDT_8   | 50     | YOLOv3_UAVDT_5   | YOLOv3-spp | 16         | Finetune   |

<br>

## Comentários
Nesta seção, são expostas as principais características exploradas no treinamento de cada experimento.

- Vídeos do Dataset:
  - Conténdo todas as situações (dia, noite, etc.).
- Pré-processamento das imagens
  - Zonas de ignore substituídas pela cor preta.
- Características da rede:
  - Resolução de entrada: 640
  - Otimizador: adam

<br>

# Resultados Detecção
<br>
 
 ## Treinamento

| Experimento      | mAP    | AP@0.5 | AP@0.75 | Fitness  |
|------------------|--------|--------|---------|----------|
| YOLOv3_UAVDT_0   |        |        |         |          |
| YOLOv3_UAVDT_1   |        |        |         |          |
| YOLOv3_UAVDT_3   |        |        |         |          |
| YOLOv3_UAVDT_4   |        |        |         |          |
| YOLOv3_UAVDT_5   |        |        |         |          |
| YOLOv3_UAVDT_6   |        |        |         |          |
| YOLOv3_UAVDT_7   |        |        |         |          |
| YOLOv3_UAVDT_8   |        |        |         |          |


| Experimento      | mAP    | AP@0.5 | AP@0.75 | Fitness  |
|------------------|--------|--------|---------|----------|
| YOLOv5_UAVDT_0   |        |        |         |          |
| YOLOv5_UAVDT_1   |        |        |         |          |
| YOLOv5_UAVDT_3   |        |        |         |          |
| YOLOv5_UAVDT_4   |        |        |         |          |
| YOLOv5_UAVDT_5   |        |        |         |          |
| YOLOv5_UAVDT_6   |        |        |         |          |
| YOLOv5_UAVDT_7   |        |        |         |          |
| YOLOv5_UAVDT_8   |        |        |         |          |
| YOLOv5_UAVDT_9   |        |        |         |          |
| YOLOv5_UAVDT_10  |        |        |         |          |
| YOLOv5_UAVDT_11  |        |        |         |          |
<sub>mAP = AP@0.5:0.95</sub>

## UAVDT_Benchmark

| Experimento      | AP_Overall | AP_attributes |
|------------------|------------|---------------|
| YOLOv3_UAVDT_0   |            |               |
| YOLOv3_UAVDT_1   |            |               |
| YOLOv3_UAVDT_3   |            |               |
| YOLOv3_UAVDT_4   |            |               |
| YOLOv3_UAVDT_5   |            |               |
| YOLOv3_UAVDT_6   |            |               |
| YOLOv3_UAVDT_7   |            |               |
| YOLOv3_UAVDT_8   |            |               |


| Experimento      | AP_Overall | AP_attributes |
|------------------|------------|---------------|
| YOLOv5_UAVDT_0   |            |               |
| YOLOv5_UAVDT_1   |            |               |
| YOLOv5_UAVDT_3   |            |               |
| YOLOv5_UAVDT_4   |            |               |
| YOLOv5_UAVDT_5   |            |               |
| YOLOv5_UAVDT_6   |            |               |
| YOLOv5_UAVDT_7   |            |               |
| YOLOv5_UAVDT_8   |            |               |
| YOLOv5_UAVDT_9   |            |               |
| YOLOv5_UAVDT_10  |            |               |
| YOLOv5_UAVDT_11  |            |               |


<br>
<br>

## Resultados Rastreamento

| Experimento      | MOTA   | MOTP   | MT      |ML       |FMT      |IDSW     | FP      |TP       |
|------------------|--------|--------|---------|---------|---------|---------|---------|---------|
| YOLOv3_UAVDT_0   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_1   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_3   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_4   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_5   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_6   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_7   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_8   |        |        |         |         |         |         |         |         |
| YOLOv3_UAVDT_9   |        |        |         |         |         |         |         |         |


| Experimento      | MOTA   | MOTP   | MT      |ML       |FMT      |IDSW     | FP      |TP       |
|------------------|--------|--------|---------|---------|---------|---------|---------|---------|
| YOLOv5_UAVDT_0   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_1   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_3   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_4   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_5   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_6   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_7   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_8   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_9   |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_10  |        |        |         |         |         |         |         |         |
| YOLOv5_UAVDT_11  |        |        |         |         |         |         |         |         |

