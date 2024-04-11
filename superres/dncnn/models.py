import torch
import torch.nn as nn
import lightning as L
import numpy as np
from utils import *

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.dncnn(x)
        return out

class dncnn_lightning(L.LightningModule):
  def __init__(self, model, optimizer, loss_fn):
    super().__init__()
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def training_step(self, batch, batch_idx):  
    noise, noisy = self.__shared_step(batch, batch_idx)

    self.optimizer.zero_grad()
    preds = self.forward(noisy)

    loss, psnr = self.__eval_step(preds, noise, noisy)
    self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True)
    self.log("train_psnr", psnr, on_step=True, on_epoch=False, logger=True)
  
    return loss

  def validation_step(self, batch, batch_idx):
    noise, noisy = self.__shared_step(batch, batch_idx)

    preds = self.forward(noisy)
    loss, psnr = self.__eval_step(preds, noise, noisy)
    
    self.log('valid_loss', loss, logger=True)
    self.log('valid_psnr', psnr, logger=True)

    return loss

  def test_step(self, batch, batch_idx):
    noise, noisy = self.__shared_step(batch, batch_idx)

    preds = self.forward(noisy)
    loss, psnr = self.__eval_step(preds, noise, noisy)
    
    self.log('test_loss', loss, logger=True)
    self.log('test_psnr', psnr, logger=True)
    
    return loss

  def configure_optimizers(self):
    return self.optimizer

  def forward(self, input):
    return self.model(input)

  def __shared_step(self, batch, batch_idx):
    x, y = batch
    noise = x - y
    noise = noise.reshape(noise.shape[0], 1, noise.shape[1], noise.shape[2])
    y = y.reshape(y.shape[0], 1, y.shape[1], y.shape[2])
    noise, y = noise.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)
    return  noise, y

  def __eval_step(self, preds, noise, noisy):
    loss = self.loss_fn(preds, noise) / (noisy.size()[0]*2)
    psnr = batch_PSNR(preds, noisy-noise, 1.)
    return loss, psnr