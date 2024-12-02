from ultralytics.utils.loss import (
    v8DetectionLoss,
    E2EDetectLoss,
    make_anchors
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import autocast

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    https://github.com/ultralytics/ultralytics/issues/1531
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        label = (gt_score > 0).int()
        weight = alpha * (pred_score.sigmoid() - gt_score).abs().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score, gt_score, reduction="none") * weight)
            )
        return loss

loss_mapping = {
    'focal_loss': FocalLoss,
    'varifocal_loss': VarifocalLoss,
}

class Customv8DetectionLoss(v8DetectionLoss):
    """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
    def __init__(self, model, loss_func):
        super().__init__(model = model)
        if loss_func in loss_mapping:
            self.bce = loss_mapping[loss_func]()

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        """
        yolov8n: 4 class
            - pred: 68 x 80 x 80, 68 x 40 x 40, 68 x 20 x 20
            - self.no = m.nc + m.reg_max * 4 --> 68 (4 + 16 * 4)
        """
        feats = preds[1] if isinstance(preds, tuple) else preds
        """
            - torch.cat(n x 68 x 6400, n x 68 x 1600, n x 68 x 400) --> n x 68 x 8400
            - n x 68 x 8400 --> n x 64 x 8400: pred_distri, n x 4 x 8400: pred_scores
        """
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # pred_scores: n x 8400 x 4
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # pred_distri: n x 8400 x 64
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # 640 x 640
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # 8400 x 2, 8400 x 1
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        # cat(n x 1 x l, n x 1 x l, n x l x 4) --> (n x l) x 6
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # n x n_max_boxes x 5
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # n x n_max_boxes x 1, n x n_max_boxes x 4
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # n x n_max_boxes
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        # xyxy, (b, h*w, 4), pred_distri --> top left (tl), right bottom (rb) --> pred_bboxes: anchor - tl, anchor + rb
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
        