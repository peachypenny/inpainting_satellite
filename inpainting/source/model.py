import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
import lightning as L
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import seaborn as sns
from torchvision import transforms
from lightning.pytorch.loggers import CSVLogger


class ModisNet(L.LightningModule):
  def __init__(self, model, optimizer, loss_fn):
    super().__init__()
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def training_step(self, batch, batch_idx):
    # training_step defines the train loop.
    # it is independent of forward
    x, y = self.__shared_step(batch, batch_idx)

    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()
    assert all(p.requires_grad for p in self.model.parameters())

    self.optimizer.zero_grad()
    preds = self.forward(x)

    # Compute the loss and its gradients
    loss = self.loss_fn(preds, y).mean()
    self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = self.__shared_step(batch, batch_idx)
    preds = self.forward(x)
    loss = self.__eval_step(preds, y)
    self.log('valid_loss', loss, on_epoch=True, logger=True)

    return loss

  def test_step(self, batch, batch_idx):
    x, y = self.__shared_step(batch, batch_idx)
    preds = self.forward(x)
    loss = self.__eval_step(preds, y)
    self.log('test_loss', loss, logger=True)

    return loss

  def configure_optimizers(self):
    return self.optimizer

  def forward(self, input):
    return self.model(input)

  def __shared_step(self, batch, batch_idx):
    x, y =  batch
    x, y = x.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    y = y.reshape(y.shape[0], 1, y.shape[1], y.shape[2])

    return x, y

  def __eval_step(self, preds, y):
    return self.loss_fn(preds, y).sum()