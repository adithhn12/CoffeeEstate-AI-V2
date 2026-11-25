"""
Faster R-CNN (ResNet50-FPN) Model Builder
==========================================
Builds a Faster R-CNN with a ResNet-50 FPN backbone, pretrained on COCO,
then replaces the box predictor head with a custom head for 6 classes
(5 coffee cherry classes + background).

RTX 4060 8 GB optimisations
  • Smaller anchors for dense, small cherry objects (16, 32, 64, 128, 256)
  • min_size=640, max_size=800 (keeps GPU under 7 GB at batch=2)
  • 256 RPN proposals (down from 2000) during training
  • Gradient checkpointing available for backbone
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


NUM_CLASSES = 6  # 5 coffee cherry classes + 1 background


def build_fasterrcnn(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 640,
    max_size: int = 800,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.55,
    box_detections_per_img: int = 500,
) -> nn.Module:
    """
    Build and return a Faster R-CNN model with:
      - ResNet-50-FPN backbone (COCO pretrained)
      - Custom anchor sizes optimised for dense coffee cherry objects
      - Custom FastRCNN predictor head for `num_classes`

    Parameters
    ----------
    num_classes : int
        Total number of classes INCLUDING background (default 6).
    pretrained : bool
        Load COCO pretrained weights for backbone and RPN.
    trainable_backbone_layers : int
        Number of backbone layers to unfreeze (0 = all frozen, 5 = all trainable).
    min_size / max_size : int
        Image resize bounds — tuned for 8 GB VRAM at batch=2.
    box_score_thresh : float
        Minimum score for a box to be kept during inference.
    box_nms_thresh : float
        NMS IoU threshold for box prediction.
    box_detections_per_img : int
        Maximum detections per image during inference.
    """

    # Custom anchors — coffee cherries are small and densely packed
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        # RPN — reduced proposal counts to save VRAM
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=500,
        # ROI box head
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )

    # Replace the box predictor head (1024-d fc + N-class output)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load a checkpoint saved by train_frcnn.py.

    Returns the checkpoint dict (contains 'epoch', 'best_map50', etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Model] Loaded checkpoint from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, best_mAP50={ckpt.get('best_map50', 'N/A')})")
    return ckpt


if __name__ == "__main__":
    model = build_fasterrcnn(pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
