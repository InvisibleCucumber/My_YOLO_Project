# 🚀 YOLO Object Detection

## 📌 1. Project Description
This project demonstrates **object detection** using a pre-trained **YOLOv5** model.  
We detect objects (such as cars or fruits) in images and save both the **annotated images** and **detection results (CSV)**.

---

## 📊 2. Datasets

### 🚗 Car Object Detection  
Dataset: [Kaggle - Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data)

```python
import kagglehub
path = kagglehub.dataset_download("sshikamaru/car-object-detection")
print("Path to dataset files:", path)
```

### 🍎 Fruits Object Detection  
 Dataset: [Kaggle - Fruits Object Detection](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection)

```python
import kagglehub
path = kagglehub.dataset_download("mbkinaci/fruit-images-for-object-detection")
print("Path to dataset files:", path)
```

## 🛠️ 3. Environment Setup
python 3.13
YOLOv5
yolo_env --> Activate in bash -->
```bash
source yolov5-env/Scripts/activate
source E:/VSCode02/YOLO5/yolov5-env/Scripts/activate
```

## ▶️ 4. Run the Project

### 🐍 Run using YOLOv5 directly (Bash):
```bash
python detect.py --source /e/VSCode02/YOLO5/CarObjectDetection/input_data --weights yolov5s.pt --conf 0.25 --project E:/VSCode02/YOLO5/CarObjectDetection --name outputs
```
📦 Run using Ultralytics (Python):
We used the Ultralytics library to perform the same object detection task directly in Python
### Activate environment
```bash
source yolov5-env/Scripts/activate
```
### Run detection script
```bash
python e:/VSCode02/YOLO5/src/detection.py
```

## ⚙️ 5. Using Function from utils.py :
    To avoid repetitive code, we created a reusable function inside utils.py that performs object detection with a single function call.

    You can also export results as a CSV file by setting **csv=True**.

    If you are using video, webcam, large number of images or streaming, be sure to add **stream=True** so that the outputs are processed as a generator (stream) and not stored in memory.

# 🚀 YOLO Classification

## Description
    Inside the utill, we wrote a function that uses the kwargs and is used for classification!
    In this style of function, we don't need to set new parameters every time, but it intelligently takes and uses any number of new parameters.
