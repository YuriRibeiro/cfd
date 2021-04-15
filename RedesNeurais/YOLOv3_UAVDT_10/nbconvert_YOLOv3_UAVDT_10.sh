conda activate yv5r4; mkdir logs; experimento="YOLOv3_UAVDT_10_TRAIN"; outputFileName=$(date "+"$experimento"_ipynb_%d_%B_%Y_%Hh_%Mm_%Ss"); echo "Salvando log de execucao em: $outputFileName.txt"; nohup jupyter nbconvert --to notebook --execute --allow-errors --output "./logs/$outputFileName" --ExecutePreprocessor.timeout=-1 --Application.log_level=10 $experimento.ipynb &> "./logs/$outputFileName.txt" &