from ultralytics import YOLO
from PIL import Image

model = YOLO("model\\best.pt")
def run_yolo_crop(image_path, save_path="cropped.jpg", conf_threshold=0.5):
    results = model(image_path, conf=conf_threshold)
    if len(results[0].boxes.xyxy) == 0:
        return None
    box = results[0].boxes.xyxy[0].tolist()
    img = Image.open(image_path)
    crop = img.crop((box[0], box[1], box[2], box[3]))
    crop.save(save_path)
    return save_path