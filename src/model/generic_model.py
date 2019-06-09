import torch
import torch.nn as nn
import numpy as np
import shutil

from ..logger import Logger
import configs.config as config

log = Logger()


class Model(nn.Module):

	@staticmethod
	def accuracy(x, y):

		_, arg = torch.max(x, dim=1)

		eq = torch.eq(arg.squeeze(), y.squeeze())

		return torch.mean(eq.float())

	@staticmethod
	def print_info(info):

		log.info('The average accuracy is :', np.mean(info['Acc']))
		log.info('The current accuracy is :', info['Acc'][-1])
		log.info('The average loss is :', np.mean(info['Loss']))
		log.info('The current loss is :', info['Loss'][-1])

	def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar'):

		torch.save({
			'epoch': epoch_i,
			'state_dict': self.state_dict(),
			'optimizer': self.opt.state_dict()}, config.dir['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
		torch.save(info, config.dir['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		if is_best:
			
			shutil.copyfile(
				config.dir['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			shutil.copyfile(
				config.dir['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

	def load(self, path, path_info):

		checkpoint = torch.load(config.dir['Exp']+'/'+path)

		self.load_state_dict(checkpoint['state_dict'])
		self.opt.load_state_dict(checkpoint['optimizer'])

		return checkpoint['epoch'], torch.load(config.dir['Exp']+'/'+path_info)