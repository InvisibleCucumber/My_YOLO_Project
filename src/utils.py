from ultralytics import YOLO
import os
import pandas as pd

def obj_detection(input_path : str , weights : str = "yolov5s.pt" , output_path : str = "outputs" ,
                   name : str = "predict" , save = True , show = False , conf : float = 0.25 , classes = None ,
                   existing = False ,  csv = False , csv_name = "csv" , stream : bool = False):
    model = YOLO(weights) # defualt weights = yolov5s.pt
    results = model.predict(source = input_path , project = output_path , name = name , save = save ,
                            show = show , conf = conf , classes = classes , exist_ok = existing )


    # Print Output Directory
    final_path = os.path.join(output_path , name)
    print(f"✅ Results save in : {final_path}")

    # Save results as CSV file
    if csv :
        detections = []

        for r in results :
            img_path = r.path
            boxes = r.boxes.xyxy.tolist()
            clsses = r.boxes.cls.tolist()
            confs = r.boxes.conf.tolist()

            for box , cls , conf in zip(boxes , clsses , confs) :
                xmin , ymin , xmax , ymax = box
                class_name = r.names[int(cls)]
                detections.append({
                    "image": img_path,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": conf,
                    "class_id": int(cls),
                    "class_name": class_name
                })
        # Create DataFrame from detection list
        df = pd.DataFrame(detections)

        # Save df as CSV file
        csv_folder = os.path.join(final_path, csv_name)
        os.makedirs(csv_folder , exist_ok=True)
        csv_path = os.path.join(csv_folder , "detection.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ CSV saved at: {csv_path}")


    return results