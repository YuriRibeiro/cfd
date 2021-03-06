# Introdução

Nesta pasta são salvos os datasets completos, contendo todas as imagens, vídeos e arquivos de apoio.

Observação:
O arquivo .gitignore está configurado para ignorar todas as subpastas deste diretório.


## UAV-M Dataset

Nome dos arquivos para download:
- Dataset (6.3 GB): `fname="UAV-benchmark-M.zip"`
- Atributos do UAV Dataset (9,3 KB): `fname="M_attr.zip"`
- Devkit do UAV Datset (234,3 MB): `fname="UAV-benchmark-MOTD_v1.0.zip"`
- (EXTRA) Vídeos de treino em formato AVI (1.7 GB): `fname="train_videos_avi.zip"`
- (EXTRA) Vídeos de teste em formato AVI (1.1 GB): `fname="test_videos_avi.zip"`

```bash 
fname="UAV-benchmark-M.zip" && ds_folder="UAVDT" && fpath="./${ds_folder}/${fname}" && mkdir $ds_folder ; curl -L "https://storage.googleapis.com/cfdy/Datasets/UAVDT/${fname}" -o $fpath && unzip -q $fpath -d "./${ds_folder}" && rm -f $fpath && echo "FIM do download de ${fname}"
```

Para criar uma cópia simbólica do UAVDT no formato requerido pelo YOLOv5 para detecções.
```bash
python ../utils/uavdt_to_yolov5.py --conjunto treinoEteste --dest-folder ./UAVDT_YOLOv5
```

Para criar uma cópia simbólica do UAVDT com apenas os primeiros 20 frames de cada vídeo (útil para testes rápidos).
```bash
#cd into the full dataset, UAVDT/UAV-benchmark-M
dest="$HOME/Desktop/Datasets/UAVDT_minimal/UAV-benchmark-M";
echo $dest;
for i in $(ls); do for k in $(ls ./$i/ | head -20); do mkdir -p $dest/$i && cp -al ./$i/$k  $dest/$i/$k ; done; done
```
