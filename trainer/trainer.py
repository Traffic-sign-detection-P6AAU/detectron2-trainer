import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from roboflow import Roboflow
import os, json, random
from detectron2.data.datasets import register_coco_instances
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

dataset_path = "C:/Users/kaspe/Documents/GitHub/detectron2-trainer/speed-signs-for-dk-4/"

def prepare_dataset():
    if not os.path.isdir('speed-signs-for-dk-4'):
        rf = Roboflow(api_key="ay1GyW7t9J2m3JLcZIul")
        project = rf.workspace("university-wtl9y").project("speed-signs-for-dk")
        dataset = project.version(4).download("coco")


    register_coco_instances("my_dataset_train1", {}, dataset_path+"train/_annotations.coco.json", dataset_path+"/train")
    register_coco_instances("my_dataset_val1", {}, dataset_path+"valid/_annotations.coco.json", dataset_path+"/valid")

def train(cfg):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
    cfg.DATASETS.TRAIN = ('my_dataset_train1')
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
    cfg.SOLVER.MAX_ITER = 43000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()

def evaluate(cfg):
    register_coco_instances("my_dataset_val1", {}, dataset_path+"valid/_annotations.coco.json", dataset_path+"/valid")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val1", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val1")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
