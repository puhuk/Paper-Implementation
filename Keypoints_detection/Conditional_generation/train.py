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
from model import Encoder, psai_encoder, PoseEncoder, Decoder
from loss import Perceptual_loss

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

def _create_render_sizes(max_size, min_size, renderer_stride):
    render_sizes = []
    size = max_size
    while size >= min_size:
        render_sizes.append(size)
        size = max_size // renderer_stride
        max_size = size
    return render_sizes

class train:

    def __init__(self, opt):
        self.n_filters = 32
        self.n_maps=10
        self.max_size =[128, 128]
        self.min_size = [16, 16]
        self.renderer_stride = 2
        self.map_filters = n_filters*8 + n_maps
        self.n_render_filters = 32
        self.n_final_out=3

        self.opt = opt
        self.map_sizes=_create_render_sizes(self.max_size[0], self.min_size[0], self.renderer_stride)

        self.psai_encoder = psai_encoder(in_channels=3, n_filters=32)
        self.pi_encoder =  PoseEncoder(in_channels=3, n_filters=32, n_maps=10, map_sizes=self.map_sizes)
        self.renderer = Decoder(self.min_size, self.map_filters, self.n_render_filters, self.n_final_out, n_final_res=self.max_size[0])
        
        self.loss = Perceptual_loss()

        
    def train():

        start_epoch = 1
        best_result = 1
        best_flag = False

        if self.opt.resume_checkpoint:
            print('Resuming checkpoint at {}'.format(self.opt.resume_checkpoint))
            checkpoint = torch.load(
                self.opt.resume_checkpoint,
                map_location=lambda storage, loc: storage, pickle_module=pickle)

            model_state = checkpoint['modelstate']
            self.neuralnet.load_state_dict(model_state)

            optim_state = checkpoint['optimstate']
            self.optimizer = _make_optimizer(
                self.opt, self.neuralnet, param_groups=optim_state['param_groups'])

            start_epoch = checkpoint['epoch']+1
            best_result = checkpoint['best_result']


if __name__ == "__main__":
    print(ARGS)

    n_filters = 32
    n_maps=10
    max_size =[128, 128]
    min_size = [16, 16]
    renderer_stride = 2
    map_filters = n_filters*8 + n_maps
    n_render_filters = 32
    n_final_out=3

    model = Encoder(in_channels=3, n_filters=32)

    image = torch.randn([32, 3, 128, 128])
    # print(image.shape)

    # x = model(image)
    # print(len(x))
    # for item in x:
    #     print(item.shape)

    model1 = psai_encoder(in_channels=3, n_filters=32)

    embeddings  = model1(image)
    print(len(embeddings))
    for item in embeddings:
        print(item.shape)

    map_sizes=_create_render_sizes(max_size[0], min_size[0], renderer_stride)

    model2 = PoseEncoder(in_channels=3, n_filters=32, n_maps=10, map_sizes=map_sizes)
    gauss_pt, pose_embeddings = model2(image)
    
    print(len(gauss_pt), len(pose_embeddings))

    renderer = Decoder(min_size, map_filters, n_render_filters, n_final_out, n_final_res=max_size[0])


    print(embeddings[-1].shape, pose_embeddings[-1].shape)

    joint = torch.cat((embeddings[-1], pose_embeddings[-1]), dim=1)
    print(joint.shape)

    pred = renderer(joint)
    print(pred.shape)

    print(renderer)


