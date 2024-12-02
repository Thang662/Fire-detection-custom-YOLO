from ultralytics.nn.tasks import DetectionModel
from utils.loss import (
    E2EDetectLoss,
    Customv8DetectionLoss,
)

class CustomDetectionModel(DetectionModel):
    """YOLO detection model."""
    def __init__(self, cfg = "yolov8n.yaml", ch = 3, nc = None, verbose = True, loss_func = ''):  # model, input channels, number of classes
        """Initialize the YOLO detection model with the given config and parameters."""
        super().__init__(cfg = cfg, ch = ch, nc = nc, verbose = verbose)
        self.loss_func = loss_func

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else Customv8DetectionLoss(self, loss_func = self.loss_func)