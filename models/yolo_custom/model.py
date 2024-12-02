# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from nn.task import CustomDetectionModel
# from models.yolo.detect.train import CustomDetectionTrainer
from models import yolo_custom
from ultralytics.utils import ROOT, yaml_load
from ultralytics import YOLO
from functools import partial


class CustomYOLO(YOLO):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model = "yolo11n.pt", task = None, verbose = False, loss_func = ''):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        self.loss_func = loss_func
        super().__init__(model = model, task = task, verbose = verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": CustomDetectionModel,
                "trainer": partial(yolo_custom.detect.CustomDetectionTrainer, loss_func = self.loss_func),
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }