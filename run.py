
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

NAME = "resnet18"
VERSION = "__" + datetime.now().strftime("%d/%m/%Y %H:%M:%S")

class LitAutoEncoder(pl.LightningModule):
	def __init__(self, model):
		super().__init__()
		self.model = model
		self.loss = nn.CrossEntropyLoss()

	# used for iference only (separate from training)
	def forward(self, x):
		y = self.model(x)
		return y

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		y_hat = self.model(x)  
		loss = self.loss(y_hat, y)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		y_hat = self.model(x)  
		# print(y)
		# print(y_hat)
		loss = self.loss(y_hat, y)
		self.log('val_loss', loss)
	
	def train_dataloader(self):
		training_data = CustomImageDataset(img_dir = "/content/images", transform = transform_train)
		train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
		return train_dataloader

	def val_dataloader(self):
		val_data = CustomImageDataset(img_dir = "/content/images", val_transform=transform_val)
		val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
		return val_dataloader

#logger
logger = TensorBoardLogger(save_dir = "lightning_logs/", name=NAME, version = VERSION)

# model
model = LitAutoEncoder(model)

# training
trainer = pl.Trainer(logger=logger,
                     accelerator="cpu",
                     precision=32,
										 limit_train_batches=1.0,
										 check_val_every_n_epoch=5)
# trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model)
    
