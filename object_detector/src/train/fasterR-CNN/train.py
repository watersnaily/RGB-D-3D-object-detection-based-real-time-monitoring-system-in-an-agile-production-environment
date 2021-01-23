import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

train_json_path = "station_mask/annotation/mask_train.json"
val_json_path = "station_mask/annotation/mask_val.json"
train_image_path = "station_mask/images/train"
val_image_path = "station_mask/images/val"
register_coco_instances("station_mask_train", {}, train_json_path, train_image_path)
register_coco_instances("station_mask_val", {}, val_json_path, val_image_path)

station_train_metadata = MetadataCatalog.get("station_mask_train")
station_val_metadata = MetadataCatalog.get("station_mask_val")

train_dataset_dicts = DatasetCatalog.get("station_mask_train")
val_dataset_dicts = DatasetCatalog.get("station_mask_val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("station_mask_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") 
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("station_mask_val", cfg, False, output_dir="./output")
val_loader = build_detection_test_loader(cfg, "station_mask_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
