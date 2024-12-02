from ultralytics.models.yolo.detect import DetectionTrainer
from nn.task import CustomDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

class CustomDetectionTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a detection model.

    Example:
        ```python
        from models.yolo_custom.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = CustomDetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg = DEFAULT_CFG, overrides = None, _callbacks = None, loss_func = ''):
        super().__init__(cfg, overrides, _callbacks)
        self.loss_func = loss_func

    def get_model(self, cfg = None, weights = None, verbose = True):
        model = CustomDetectionModel(cfg, nc = self.data['nc'], verbose = verbose and RANK == -1, loss_func = self.loss_func)
        if weights:
            model.load(weights)
        return model