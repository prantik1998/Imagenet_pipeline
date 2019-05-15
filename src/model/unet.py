import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import shutil
import os

from unet_parts import *


class UNet(nn.Module):

	def __init__(self, n_channels, n_classes, lr=1e-4, opt='Adam', lossf='MSE', logger_dir='', PreTrained=True):
		
		super(UNet, self).__init__()
		self.inc = inconv(n_channels, 32*2)
		self.down1 = down(32*2, 64*2)
		self.down2 = down(64*2, 128*2)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 512)
		self.up2 = up(512+256, 256)
		self.up3 = up((128+64)*2, 64*2)
		self.up4 = up(96*2, 64*2)
		self.sigma = nn.Sigmoid()
		
		self.outc = outconv(64*2, n_classes)

		self.PreTrained = PreTrained

		if opt == 'Adam':
			self.opt = optim.Adam(self.parameters(), lr=lr)

		if lossf == 'CEL':
			self.lossf = torch.nn.CrossEntropyLoss()
		elif lossf == 'MSE':
			self.lossf = torch.nn.MSELoss()
		# elif lossf == 'CMSE':
		# 	self.lossf = self.cmse

	# def cmse(self, x, y):

	# 	return torch.mean((x - y)**2)
	def accuracy(self, x, y):

		_, arg = torch.max(x, dim=1)

		eq = torch.eq(arg.squeeze(), y.squeeze())

		return torch.mean(eq.float())

	def print_info(self, info):

		# print('The average accuracy is :', np.mean(info['Acc']))
		# print('The current accuracy is :', info['Acc'][-1])
		print('The average loss is :', np.mean(info['Loss']))
		print('The current loss is :', info['Loss'][-1])

	def forward(self, x):

		x1 = self.inc(x)
		# print(x.size(), 'x1')
		x2 = self.down1(x1)
		# print(x2.size(), 'x2')

		x3 = self.down2(x2)
		# print(x3.size(), 'x3')

		x4 = self.down3(x3)
		# print(x4.size(), 'x4')

		x5 = self.down4(x4)
		# print(x5.size(), 'x5')

		x_ = self.up1(x5, x4)
		# print(x.size(), 'up1')

		x_ = self.up2(x_, x3)
		# print(x.size(), 'up2')

		x_ = self.up3(x_, x2)
		# print(x.size(), 'up3')
		x_ = self.up4(x_, x1)
		# print(x.size(), 'up4')

		# x = self.up5(x)
		# # print(x.size(), 'up5')

		# x = self.up6(x)
		# # print(x.size(), 'up6')

		# x = self.up7(x)
		# # print(x.size(), 'up7')

		x = self.outc(x)
		x = self.sigma(x)
		# print(x.size(), 'outc')

		return x


	def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar'):
		# if no == 0:
		# 	all_files = os.listdir()
		# 	for i in all_files:
		# 		if '_'+str(epoch_i)+'_'+filename in i:
		# 			os.remove(i)
		# 		if '_'+str(epoch_i)+'_'+'info_'+filename in i:
		# 			os.remove(i)

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

		pred = pred.transpose(1, 3).contiguous().view(b*w*h, ch)
		target = target.transpose(1, 3).contiguous().view(b*h*w, ch)

		loss_c = self.lossf(pred, target)

		if info['Keep_log']:

			# info['Acc'].append(self.accuracy(pred, target).data.cpu().numpy())
			info['Loss'].append(loss_c.data.cpu().numpy())

		else:

			# info['Acc'] = (self.accuracy(pred, target).data.cpu().numpy() + info['Count']*info['Acc'])/(info['Count']+1)
			info['Loss'] = (loss_c.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)

		info['Count'] += 1


		return loss_c



