Esta página contém backups de páginas Web com instruções para treinamento de redes neurais.

Bloco de código Html a ser colocada logo abaixo do `<body>` nas páginas clonadas.

```html
<table bgcolor="#FFFF00" style="z-index: 2;height:2.5%;width:100%; position: absolute; top: 0; bottom: 0; left: 0; right: 0;border:1px solid">
     <tr style="height: 5%;">
        <td><b>ESTA PÁGINA É UM BACKUP. VERIFICAR O LINK PARA A PÁGINA ORIGINAL <a href="https://yuriribeiro.github.io/DissertacaoMestrado/"> AQUI</a>.</b></td>
     </tr>
</table>
```

# YoloV5

- **Repositório Original**
  - [YoloV5 Git Repository Main Page](https://github.com/ultralytics/yolov5)
  - [YoloV5 Training Tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
  - [YoloV5 Training Multi GPU Tutorial](https://github.com/ultralytics/yolov5/issues/475)
- **Documentations Backups**
  - Commit: 4e2b9ecc7e03e3ec1a6149ede88f316a298294b2
  - Data: 30 jun. 2020 15h33m.
  - [YoloV5 Training](Train Custom Data · ultralytics_yolov5 Wiki)
  - [YoloV5 Training Multi GPU](Multi-GPU Training · Issue 475 · ultralytics_yolov5)
  - [YoloV5 Git Repo Main Page](ultralytics_yolov5 YOLOv5 in PyTorch > ONNX > CoreML > iOS)
- **Snippets de Códigos**
  - Comando básico para treinamento:
```bash
python train.py --img input_img_size --batch batch_size --epochs num_epochs --data data.yaml_file --cfg model/yolov5_model.yaml --weights ''
```

