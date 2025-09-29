from ultralytics import YOLO
import os

# model = YOLO("yolov5s.pt")

images_path = "E:/VSCode02/YOLO5/FruitImagesObjectDetection/train_zip/train"
output_path = "E:/VSCode02/YOLO5/outputs"

# results = model.predict(source = os.path.join(images_path , "*.jpg") , 
#                         project = output_path ,
#                         name = "ultralytics" ,
#                         save = True
#                         )
from utils import obj_detection

if __name__ == "__main__" :

    # result = obj_detection(input_path = os.path.join(images_path , "*.jpg") ,
    #                        output_path = output_path , name = "CarObjectDetection" , csv=True)
    
    result = obj_detection(input_path = os.path.join(images_path , "*.jpg") ,
                           output_path = output_path , name = "FruitImagesObjectDetection" , csv=True , stream = True)

    # With the following commands we can get the specifications of each detection.
    # for r in result:
    #     print("Image:", r.path)
    #     print("Boxes:", r.boxes.xyxy)  # Box coordinates
    #     print("Confidence:", r.boxes.conf)  # Confidence
    #     print("Class:", r.boxes.cls)  # Classes


