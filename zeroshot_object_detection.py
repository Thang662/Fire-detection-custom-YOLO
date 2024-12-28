import time
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from tqdm import tqdm_notebook as tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument("--checkpoint", type=str, default="google/owlv2-base-patch16-ensemble")
args = parser.parse_args()


checkpoint = args.checkpoint
model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to("cuda")
processor = AutoProcessor.from_pretrained(checkpoint)
label2id = {'car_fire': 0, 'inhouse_fire': 1, 'wild_fire': 2, 'smoke': 3}
id2label = {v: k for k, v in label2id.items()}

dataset = load_dataset("data/data.py", trust_remote_code=True)
metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

valid_dataset = dataset["validation"]

with torch.no_grad():
    for example in tqdm(valid_dataset):
        image = example["image"]
        text_queries = ["car_fire", "inhouse_fire", "wild_fire", "smoke"]
        inputs = processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        target_sizes = torch.tensor([image.size[::-1]])
        outputs = model(**inputs)
        predictions = processor.post_process_object_detection(outputs, threshold = args.threshold, target_sizes=target_sizes)[0]
        predictions = {k: v.cpu() for k, v in predictions.items()}
        postprocessed_target = []
        postprocessed_predictions = []
        postprocessed_target.append({
            'boxes': example['objects']['bbox'],
            'labels': example['objects']['category'],
            # 'scores': [1.0] * len(example['objects']['category'])
        })
        postprocessed_predictions.append(predictions)
        for k in postprocessed_target[0]:
            postprocessed_target[0][k] = torch.tensor(postprocessed_target[0][k])
        metric.update(postprocessed_predictions, postprocessed_target)

metrics = metric.compute()

# Replace list of per class metrics with separate metric for each class
classes = metrics.pop("classes")
map_per_class = metrics.pop("map_per_class")
mar_100_per_class = metrics.pop("mar_100_per_class")
for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
    class_name = id2label[class_id.item()]
    metrics[f"map_{class_name}"] = class_map
    metrics[f"mar_100_{class_name}"] = class_mar

# Convert metrics to float
metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
print(metrics)

start = time.time()
with torch.no_grad():
    for example in tqdm(valid_dataset):
        image = example["image"]
        text_queries = ["car_fire", "inhouse_fire", "wild_fire", "smoke"]
        inputs = processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        target_sizes = torch.tensor([image.size[::-1]])
        outputs = model(**inputs)
        predictions = processor.post_process_object_detection(outputs, threshold = args.threshold, target_sizes=target_sizes)[0]
elapsed = time.time() - start
imgs_per_sec = len(valid_dataset) / elapsed
metrics = {"imgs_per_sec": imgs_per_sec, "elapsed": elapsed, "total_imgs": len(valid_dataset)}
print(metrics)