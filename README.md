# YOLOv11 Next-Gen Object Detection using PyTorch

### _Achieve SOTA results with just 1 line of code!_
# 🚀 Demo

https://github.com/user-attachments/assets/b41621ae-ebdf-453c-b2bc-9ffe01a14d08


https://github.com/user-attachments/assets/1b5d4323-64ee-4514-95f0-6a1080672753


### ⚡ Installation (30 Seconds Setup)

```
conda create -n YOLO python=3.9
conda activate YOLO
pip install thop
pip install tqdm
pip install PyYAML
pip install opencv-python
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```

### 🏋 Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### 🧪 Test/Validate

* Configure your dataset path in `main.py` for testing
* Run `python main.py --Validate` for validation

### 🔍 Inference (Webcam or Video)

* Run `python main.py --inference` for inference

### 📊 Performance Metrics & Pretrained Checkpoints

| Model                                                                                | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|--------------------------------------------------------------------------------------|----------------------|-------------------|--------------------|------------------------|
| [YOLOv11n](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_n.pt) | 39.5                 | 54.8              | **2.6**            | **6.5**                |
| [YOLOv11s](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_s.pt) | 47.0                 | 63.5              | 9.4                | 21.5                   |
| [YOLOv11m](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_m.pt) | 51.5                 | 68.1              | 20.1               | 68.0                   |
| [YOLOv11l](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_l.pt) | 53.4                 | 69.7              | 25.3               | 86.9                   | 50.7                 | 68.9              | 86.7               | 205.7                  |
| [YOLOv11x](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_x.pt) | 54.9                 | 71.3              | 56.9               | 194.9                  | 50.7                 | 68.9              | 86.7               | 205.7                  |

### 📈 Additional Metrics
### 📂 Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

⭐ Star the Repo!

If you find this project helpful, give us a star ⭐ 

#### 🔗 Reference

* https://github.com/ultralytics/ultralytics
# YOLOv11


├── coco-labels
│   ├── LICENSE
│   ├── README.md
│   └── scripts
│       └── python
│           └── dump_coco_labels.py
├── Dataset
│   ├── coco2017
│   │   ├── annotations
│   │   │   ├── captions_train2017.json
│   │   │   ├── captions_val2017.json
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── person_keypoints_train2017.json
│   │   │   └── person_keypoints_val2017.json
│   │   ├── images
│   │   │   ├── test2017
│   │   │   ├── train2017
│   │   │   └── val2017
│   │   └── labels
│   │       ├── val2017
│   │       └── val2017.cache
│   ├── convert.py
│   └── download.py
├── main.py
├── main.sh
├── nets
│   ├── nn.py
│   └── __pycache__
│       └── nn.cpython-39.pyc
├── README.md
├── utils
│   ├── args.yaml
│   ├── augment.py
│   ├── dataset.py
│   ├── __pycache__
│   │   ├── augment.cpython-39.pyc
│   │   ├── dataset.cpython-39.pyc
│   │   └── util.cpython-39.pyc
│   └── util.py
└── weights
    ├── F1_curve.png
    ├── P_curve.png
    ├── PR_curve.png
    ├── R_curve.png
    └── v11_n.pt

按照如上的方式放好下载的COCO2017(执行download.py)，然后执行：
find /media/8T3/ykqiu/rn_xu/Projects/yolo_on_FPGA/YOLOv11/Dataset/coco2017/images/val2017 -type f \( -iname "*.jpg" \) > /media/8T3/ykqiu/rn_xu/Projects/yolo_on_FPGA/YOLOv11/Dataset/coco2017/val2017.txt

分别将train、val、test的图片路径放到对应的txt中，用于dataset.py读取

然后执行convert.py，将val2017的label从json中提取出来，放到labels/val2017中，用于validate

validate的命令：python main --validate，data_dir已经提前修改好了