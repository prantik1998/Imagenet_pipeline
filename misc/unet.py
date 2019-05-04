import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from .unet_parts import *


class UNet(nn.Module):

	# n_channels, n_classes, lr=1e-4, opt='Adam', lossf='MSE', logger_dir='', PreTrained=True

	def __init__(self, config):

		n_channels = config['n_channels']
		n_classes = config['n_classes']
		lr = config['lr']
		opt = config['optimizer']
		lossf = config['lossf']
		self.PreTrained = config['PreTrained']
		self.config = config
		
		super(UNet, self).__init__()
		self.inc = inconv(n_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		self.outc = outconv(64, n_classes)

		if opt == 'Adam':
			self.opt = optim.Adam(self.parameters(), lr=lr)

		if lossf == 'CEL':
			self.lossf = torch.nn.CrossEntropyLoss()
		elif lossf == 'MSE':
			self.loss = torch.nn.MSELoss()

	def accuracy(self, x, y):

		_, arg = torch.max(x, dim=1)

		eq = torch.eq(arg.squeeze(), y.squeeze())

		return torch.mean(eq.float())

	def print_info(self, info):

		print('The average accuracy is :', np.mean(info['Acc']))
		print('The current accuracy is :', info['Acc'][-1])
		print('The average loss is :', np.mean(info['Loss']))
		print('The current loss is :', info['Loss'][-1])

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)

		return x

	def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar'):

		torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict()},str(no)+'_'+str(epoch_i)+'_'+filename)
		torch.save(info, str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		if is_best:
			shutil.copyfile(str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			shutil.copyfile(str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

	def load(self, path, path_info):

		checkpoint = torch.load(path)

		self.load_state_dict(checkpoint['state_dict'])
		self.opt.load_state_dict(checkpoint['optimizer'])

		return checkpoint['epoch'], torch.load(path_info)

	def loss(self, pred, target, info):

		b, ch, h, w = pred.size()

		pred = pred.contiguous().view(b, ch)
		target = target.contiguous().view(b)

		loss_c = self.lossf(pred, target)

		if info['Keep_log']:
			info['Acc'].append(self.accuracy(pred, target).data.cpu().numpy()[0])
			info['Loss'].append(loss_c.data.cpu().numpy()[0])

		else:
			info['Acc'] = (self.accuracy(pred, target).data.cpu().numpy()[0] + info['Count']*info['Acc'])/(info['Count']+1)
			info['Loss'] = (loss_c.data.cpu().numpy()[0] + info['Count']*info['Loss'])/(info['Count']+1)

		info['Count'] += 1
		return loss_c