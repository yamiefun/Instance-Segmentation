import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
import argparse
import json
from test import test


def load_data():
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances(
        "my_dataset_train", {}, "dataset/pascal_train.json",
        "dataset/train_images")
    register_coco_instances(
        "my_dataset_valid", {}, "dataset/pascal_train.json",
        "dataset/train_images")
    register_coco_instances(
        "my_dataset_test", {}, "dataset/test.json",
        "dataset/test_images")


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def get_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--lr", default=0.00025, type=float)
    arg_parser.add_argument("--mode", default="train")
    arg_parser.add_argument("--log_interval", default=10)
    arg_parser.add_argument("--iter", default=300, type=int)
    args = arg_parser.parse_args()
    return args


def main():
    args = get_arguments()
    load_data()
    if args.mode == "train":
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("my_dataset_train",)
        cfg.DATASETS.TEST = ("my_dataset_valid",)
        cfg.DATALOADER.NUM_WORKERS = 2

        # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl"

        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = args.lr
        cfg.SOLVER.MAX_ITER = args.iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

        train(cfg)
    elif args.mode == "test":
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ()
        cfg.DATASETS.TEST = ("my_dataset_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # mAP with thresh 0.5

        ret = test(cfg)
        with open("submission.json", "w") as f:
            json.dump(ret, f)

    else:
        print("mode not support")
        exit(0)


if __name__ == "__main__":
    main()
