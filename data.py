import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

def RGB_np2Tensor(imgIn, imgTar):
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)  
    return imgIn, imgTar

def augment(imgIn, imgTar):
    if random.random() < 0.3:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]
    if random.random() < 0.3:
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]
    return imgIn, imgTar

def getPatch(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = args.patchSize
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
    return imgIn, imgTar

class DIV2K(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        apath = args.dataDir
        dirHR = 'HR'
        dirLR = 'LR'
        self.dirIn = os.path.join(apath, dirLR)
        self.dirTar = os.path.join(apath, dirHR)
        self.fileList= os.listdir(self.dirHR)
        self.nTrain = len(self.fileList)
        
    def __getitem__(self, idx):
        scale = self.scale
        nameIn, nameTar = self.getFileName(idx)
        imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar)
        if self.args.need_patch:
            imgIn, imgTar = getPatch(imgIn, imgTar, self.args, scale)
        imgIn, imgTar = augment(imgIn, imgTar) 
        return RGB_np2Tensor(imgIn, imgTar)

    def __len__(self):
        return self.nTrain   
        
    def getFileName(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        name = name[0:-4] + 'x3' + '.png'
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar
