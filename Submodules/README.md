# Pastas de Trabalho:
Pasta com módulos de terceiros utilizados neste trabalho.

Os "git submodules" aqui servem de referência para a versão que estou utilizando atualmente.

As pastas de trabalho são precedidas por `_w`. São apenas cópias das pastas originais, sem a presença dos arquvos `*.git.*`.

```
declare -a submodules=("yolov3" "yolov5" "sort" "HOTA-metrics"); for module in ${submodules[@]}; do echo "Submodule: ${module}"; rm -rf ${module}_w; cp -ar ${module} ${module}_w; cp -ar ${module}_w_mod/. ${module}_w/; search="${module}_w"; find -type f -regex './.*git.*' | grep -E ./${search} | awk '!/gitignore/' | xargs rm -f; diff -r ${module} ${module}_w; done;
```