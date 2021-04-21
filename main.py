import torch
import torch.nn as nn
from utils import get_data
import numpy as np
from lpcnet import LPCNet
from train import train
import sys
import argparse


parser = argparse.ArgumentParser(description='Train LPCNet from scratch')
parser.add_argument('--feat', default='../features.f32', type=str, help='input feature')
parser.add_argument('--data', default='../data.u8', type=str, help='output wav file')

args = parser.parse_args()

feature_file = args.feat
pcm_file = args.data

torch.cuda._lazy_init()
torch.backends.cudnn.benchmark = True
    

print('Initialize Model....')
model = nn.DataParallel(LPCNet()).cuda()
print('Read Training Data....')
dataloader = get_data(pcm_file, feature_file)
loss = nn.CrossEntropyLoss().cuda()
if __name__ == '__main__':
    print('Start Training!!')
    train(model, dataloader, loss)