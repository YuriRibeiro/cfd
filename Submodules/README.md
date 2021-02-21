# Pastas de Trabalho:
Pasta com módulos de terceiros utilizados neste trabalho.

Os "git submodules" aqui servem de referência para a versão que estou utilizando atualmente.

As pastas de trabalho são precedidas por `_w`. São apenas cópias das pastas originais, sem a presença dos arquvos `*.git.*`.

```bash
cp -ar yolov5 yolov5_w;
cp -ar sort sort_w;
cp -ar HOTA-metrics HOTA-metrics_w;
search='yolov5_w|sort_w|HOTA-metrics'; find -type f -regex './.*git.*' | grep -E ./${search} | awk '!/gitignore/' | xargs rm -f
```