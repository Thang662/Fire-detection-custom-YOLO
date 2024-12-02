import gradio as gr
import smtplib
from email.mime.text import MIMEText
import PIL.Image as Image
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# def predict_image(img, conf_threshold, iou_threshold):
#     results = model.predict(
#         source = img,
#         conf = conf_threshold,
#         iou = iou_threshold,
#         show_labels = True,
#         show_conf = True,
#         imgsz = 640,
#     )
    
#     detected_objects = {}
#     for r in results:
#         for obj in r.boxes:
#             category = model.names[int(obj.cls)]
#             detected_objects[category] = detected_objects.get(category, 0) + 1
#         im_array = r.plot()
#         im = Image.fromarray(im_array[..., ::-1])
    
#     return im, detected_objects

def predict_video(video_path, conf_threshold, iou_threshold):
    cap = cv2.VideoCapture(video_path)
    detected_objects = {}
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(
            source = frame,
            conf = conf_threshold,
            iou = iou_threshold,
            show_labels = True,
            show_conf = True,
            imgsz = 640,
        )
        for r in results:
            for obj in r.boxes:
                category = model.names[int(obj.cls)]
                detected_objects[category] = detected_objects.get(category, 0) + 1
            im_array = r.plot()
            frames.append(im_array)
    
    cap.release()
    return frames

iface = gr.Interface(
    fn = predict_video,
    # fn = lambda img, video, conf_threshold, iou_threshold: predict_image(img, conf_threshold, iou_threshold) if img else predict_video(video, conf_threshold, iou_threshold),
    inputs = [
        # gr.Image(type = "pil", label = "Upload Image", ),
        gr.Video(label = "Upload Video", ),
        gr.Slider(minimum = 0, maximum = 1, value = 0.25, label = "Confidence threshold"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.45, label = "IoU threshold"),
    ],
    outputs = [
        # gr.Image(type = "pil", label = "Result", ),
        gr.Video(label = "Result", ),
        # gr.JSON(label = "Detected Objects")
    ],
    title = "Ultralytics Gradio Security System",
    description = "Upload images or videos for inference. The Ultralytics YOLOv8n model is used by default.",
)

if __name__ == "__main__":
    iface.launch(share = True)