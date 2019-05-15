import torch
import os

def prepare_metadata(config):
    img_dir=config["img_dir"]
    tar_obj=config["tar_obj"]
    tar_class=config["tar_class"]
    train_img=config["train"]
    validation=config["val"]
    test_img=config["test"]
