from ultralytics import YOLO
from models.yolo_custom.model import CustomYOLO
import os
custom_model = YOLO(f"yolov8n.pt")
# custom_model = CustomYOLO(f"yolov8n.pt", loss_func = '')
# print(type(model))
# print(type(custom_model))
# print(type(model.trainer))
# print(type(custom_model.trainer))
# print(type(custom_model._smart_load('model')))
# print(custom_model._smart_load('model'))
# print(type(custom_model.model))
# print(*model.__dict__.items())


custom_model.train(data = f"{os.getcwd()}/new_data/data.yaml", epochs = 1, batch = 32)