# ------------------------------
# Residual Dense Network
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim

from model import model
from data import DIV2K
from utils import *

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

parser.add_argument('--dataDir', default='./data', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='datasave directory')
parser.add_argument('--load', default= 'model_name', help='save result')

parser.add_argument('--model_name', default= 'RDN', help='model to select')
parser.add_argument('--finetuning', default=False, help='finetuning the training')
parser.add_argument('--need_patch', default=False, help='get patch form image')

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=96,  help='patch size')

parser.add_argument('--nThreads', type=int, default=3, help='number of threads for data loading')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=500, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='L1', help='output SR video')

parser.add_argument('--scale', type=int, default= 1, help='scale output size /input size')


args = parser.parse_args()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)

def get_dataset(args):
	data_train = DIV2K(args)
	dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
		drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
	return dataloader

def set_loss(args):
	lossType = args.lossType
	if lossType == 'MSE':
		lossfunction = nn.MSELoss()
	elif lossType == 'L1':
		lossfunction = nn.L1Loss()
	return lossfunction

def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args):

	#  select network
	if args.model_name == 'RDN':
		my_model = model.RDN(args)
	my_model.apply(weights_init_kaiming)
	my_model.cuda()
	save = saveData(args)
	# fine-tuning or retrain
	if args.finetuning:
		my_model = save.load_model(my_model)
	# load data
	dataloader = get_dataset(args)
  
	L1_lossfunction = set_loss(args)
	total_loss = 0
	for epoch in range(args.epochs):
		optimizer = optim.Adam(my_model.parameters())
		learning_rate = set_lr(args, epoch, optimizer)
		total_loss_ = 0
		L1_loss_ = 0
		for batch, (im_lr, im_hr) in enumerate(dataloader):
			im_lr = Variable(im_lr.cuda(), volatile=False)
			im_hr = Variable(im_hr.cuda())
			my_model.zero_grad()
			output = my_model(im_lr)
			L1_loss = L1_lossfunction(output, im_hr)
			total_loss = L1_loss 
			total_loss.backward()
			optimizer.step()
			L1_loss_ += L1_loss.data.cpu().numpy()[0] 
			total_loss_ += L1_loss.data.cpu().numpy()[0] 
		L1_loss_ = L1_loss_ / (batch + 1)
		total_loss_ = total_loss_ / (batch + 1)

		if (epoch+1) % 10 == 0:
			log = "[{} / {}] \tLearning_rate: {}\t total_loss: {:.4f}\t L2_loss: {:.4f}".format(epoch+1, 
							args.epochs, learning_rate, total_loss_, L1_loss_)
			print(log)
			save.save_log(log)
			save.save_model(my_model)
		
if __name__ == '__main__':
    train(args)
