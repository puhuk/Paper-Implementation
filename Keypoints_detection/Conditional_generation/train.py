# reconNet = aaa
# poseNet = aaa

import os, argparse, gc, glob, time, pickle
from os import path

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from visdom import Visdom
from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomSaver

import dataset
from model import landmark_detection_network

PARSER = argparse.ArgumentParser(description='Option for Conditional Image Generating')
#------------------------------------------------------------------- data-option
PARSER.add_argument('--data_root', type=str,
                    default='../../../dataset/',
                    help='location of root dir')
PARSER.add_argument('--dataset', type=str,
                    default='celeba',
                    help='location of dataset')
PARSER.add_argument('--testset', type=str,
                    default='../data/',
                    help='location of test data')
PARSER.add_argument('--nthreads', type=int, default=8,
                    help='number of threads for data loader')
PARSER.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='train batch size')
PARSER.add_argument('--val_batch_size', type=int, default=64, metavar='N',
                    help='val batch size')
#------------------------------------------------------------------ model-option
PARSER.add_argument('--pretrained_model', type=str, default='',
                    help='pretrain model location')
PARSER.add_argument('--loss_type', type=str, default='perceptual',
                    help='loss type for criterion: perceptual | l2')
#--------------------------------------------------------------- training-option
PARSER.add_argument('--seed', type=int, default=1234,
                    help='random seed')
PARSER.add_argument('--gpus', type=list, default=[3],
                    help='list of GPUs in use')
#optimizer-option
PARSER.add_argument('--optim_algor', type=str, default='Adam',
                    help='optimization algorithm')
PARSER.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
PARSER.add_argument('--weight_decay', type=float, default=1e-8,
                    help='weight_decay rate')
#saving-option
PARSER.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs')
PARSER.add_argument('--checkpoint_interval', type=int, default=1,
                    help='epoch interval of saving checkpoint')
PARSER.add_argument('--save_path', type=str, default='checkpoint',
                    help='directory for saving checkpoint')
PARSER.add_argument('--resume_checkpoint', type=str, default='',
                    help='location of saved checkpoint')
#only prediction-option
PARSER.add_argument('--trained_model', type=str, default='',
                    help='location of trained checkpoint')

ARGS = PARSER.parse_args()

if __name__ == "__main__":
    print(ARGS)

    model = landmark_detection_network(in_channels=3, n_filters=32)

    image = torch.randn([32, 3, 256, 256])
    print(image.shape)

    x = model(image)
    print(len(x))
    for item in x:
        print(item.shape)

