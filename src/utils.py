from ultralytics import YOLO
import os
import pandas as pd
from pathlib import Path
import cv2

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

def  img_classification(input_path : str , weights : str = "yolov8n-cls.pt" , **kwargs):
    
    output_path = kwargs.pop("project", "outputs")
    name = kwargs.pop("name", "classification")
    csv = kwargs.pop("csv", False)
    topk = kwargs.get("topk", 3)
    device = kwargs.pop("device", None)
    save = kwargs.pop("save", True)

    final_dir = Path(output_path) / name # outputs/classification
    final_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)

    kwargs.setdefault("source", input_path)

    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)

    rows = []
    for r in results:
        img = r.orig_img.copy()
        # probs کلاس‌ها
        probs = getattr(r, "probs", None)
        if probs is None:
            continue

        # انتقال به CPU و تبدیل به لیست
        probs_list = probs.cpu().data.numpy().tolist() if hasattr(probs, "cpu") else probs.data.numpy().tolist()

        # مرتب کردن top-k
        indexed = sorted(enumerate(probs_list), key=lambda x: x[1], reverse=True)[:topk]

        for cls_idx, prob in indexed:
            class_name = r.names[cls_idx]

            # نوشتن روی تصویر فقط کلاس اول
            if cls_idx == indexed[0][0] and save:
                cv2.putText(img, f"{class_name} {prob:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # جمع‌آوری اطلاعات CSV
            if csv:
                rows.append({
                    "image": r.path,
                    "class_id": cls_idx,
                    "class_name": class_name,
                    "probability": prob
                })

        # ذخیره تصویر annotated
        if save:
            out_path = final_dir / Path(r.path).name
            cv2.imwrite(str(out_path), img)

    # ذخیره CSV
    if csv and rows:
        df = pd.DataFrame(rows)
        df.to_csv(final_dir / "classification_results.csv", index=False)
        print(f"[✅] CSV saved to {final_dir/'classification_results.csv'}")

    print(f"[✅] Annotated images saved to {final_dir}")
    return results