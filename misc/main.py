import numpy as np
from unet import UNet
import matplotlib.pyplot as plt
import data_loader
import os
import torch.nn.functional as F
from dl_model import dl_model
import argparse
import random
import torch

def train_settings():

    # Training settings
    global parser
    global args

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 15)')
    parser.add_argument('--prev-max', type=float, default=0.0, metavar='N',
                        help='prev_max accuracy on val(default: 0.0)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 15)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1000, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--first-time', type=bool, default=False, metavar='N',
                        help='Resume trained session')
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                        help='Start epoch number')
    args = parser.parse_args()





train_settings()
seed()

train_dir = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/FaceDataset/WIDER/Unordered/WIDER_train/images/All'
test_dir = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/FaceDataset/WIDER/Unordered/WIDER_val/images/All'
train_anno = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/FaceDataset/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
test_anno = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/FaceDataset/WIDER/wider_face_split/wider_face_val_bbx_gt.txt'


dict_hp = {
	'lr': 1e-4,
	'train_batch_size': args.batch_size,
	'test_batch_size': args.test_batch_size,
	'optimiser': 'Adam',
	'loss': 'CEL',
	'epoch': 8,
	'cummulative_batch_steps': 1,
	'log_interval_steps': args.log_interval*50,
	'print_log_steps': args.log_interval,
	'test_now': args.log_interval*100
}

model = dl_model(, UNet, train_dir = train_dir, test_dir = test_dir, target_train_dir = train_anno, 
    target_test_dir = test_anno, dict_hp = dict_hp, PreTrained=False)

# print(model.train_model())

model.model.load('0_0_checkpoint.pth.tar', '0_0_info_checkpoint.pth.tar')

# model.test_model()

model.test_one_model('/home/mayank/Desktop/Start_Up/allai/Algorithms/Supervised/Segmentation/Personal_Test_Images/vinit.jpeg')
