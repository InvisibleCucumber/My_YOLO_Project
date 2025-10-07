import os
from utils import img_classification

images_path = "E:/VSCode02/YOLO5/CF"
output_path = "E:/VSCode02/YOLO5/outputs/classification"

if __name__ == "__main__" :

    results = img_classification(images_path , csv = True , save = True , project = output_path , name = "CF")

