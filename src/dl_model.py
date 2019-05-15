import sys
from torchvision import transforms
import torch.utils.data as data
import torch
from torch.autograd import Variable
import os
import numpy as np
import random
from .read_yaml import read_yaml
from .data_loader import DataLoader
from .model.resnet import resnet152
from .model.alexnet import alexnet
from .model.unet import unet
from .logger import Logger
from .model import generic_model

log = Logger()

class dl_model():

	def __init__(self, model,work="classification", use_trained = False):
		
		self.config = self.get_config()
		self.cuda = self.config['cuda'] and torch.cuda.is_available()
		self.seed()
		
		self.get_transforms()

		self.work = work;
		
		self.train_data = DataLoader(self.config['train'][self.work], transform=self.train_transform, target_transform = self.target_transform,work=self.work)
		self.test_data = DataLoader(self.config['test'][self.work], transform=self.test_transform, target_transform = self.target_transform,work=self.work)

		self.train_data_loader = data.DataLoader(self.train_data, batch_size=self.config['train'][self.work]['batch_size'], shuffle=True, num_workers=self.config['train'][self.work]['cpu_alloc'])
		self.test_data_loader = data.DataLoader(self.test_data, batch_size=self.config['test'][self.work]['batch_size'], shuffle=False, num_workers=self.config['test'][self.work]['cpu_alloc'])

		self.model = self.get_model(model, use_trained)

		self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
		self.testing_info = {'Loss': 0, 'Acc': 0, 'Keep_log': False, 'Count':0}
		
		self.model_best = {'Loss': sys.float_info.max, 'Acc': 0.0}
		

		
		if self.cuda:
			self.model.cuda()

		self.epoch_start = 0
		self.start_no = 0


		if self.config['PreTrained'] == True:
			self.epoch_start, self.training_info = self.model.load(self.config['pretrained_model']['checkpoint'], self.config['pretrained_model']['checkpoint_info'])
			self.start_no = int(self.config['pretrained_model']['checkpoint'].split('_')[0])
			self.epoch_start = int(self.config['pretrained_model']['checkpoint'].split('_')[1])
			log.info('Loaded the model')
	
	def get_model(self, model, predefined):

		if predefined:
			if self.config['pretrained_model']['checkpoint']:
				toreturn = resnet152(pretrained=False, config=self.config)
				if self.cuda:
					toreturn.cuda()

				toreturn.load('0_0_checkpoint.pth.tar','0_0_info_checkpoint.pth.tar')

				return toreturn
			else:
				print("No trained model")


		elif model == 'ResNet':
			return resnet152(pretrained=True, config=self.config)
		elif model == 'AlexNet':
			return alexnet(pretrained=True, config=self.config)
		elif model == 'UNet':
			return unet()
		
		else:
			print("Can't find model")

	def get_transforms(self):

		if self.config['train'][self.work]['transform'] == False:
			self.train_transform = transforms.Compose([
											transforms.ColorJitter(brightness=self.config['augmentation']['brightness'], contrast=self.config['augmentation']['contrast'], saturation=self.config['augmentation']['saturation'], hue=self.config['augmentation']['hue']),
											transforms.ToTensor(),
											])
		else:
			self.train_transform = train_transform

		if self.config['test'][self.work]['transform'] == False:
			self.test_transform = transforms.Compose([
											 transforms.ToTensor(),
											 ])
		else:
			self.test_transform = test_transform
		
		if self.config['target_transform'] == False:

			self.target_transform = transforms.Compose([
											 transforms.ToTensor(),
											 ])
		else:
			self.target_transform = target_transform

	def get_config(self):

		return read_yaml()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])

	def __str__(self):

		return str(self.config)


	def train_model(self):

		try:
			if self.work=="classification":
				self.train_data.get_all_names_refresh()
				self.test_data.get_all_names_refresh()
	
				self.model.requires_grad = True
	
				self.model.train()
	
				self.model.opt.zero_grad()
	
				for epoch_i in range(self.epoch_start, self.config['epoch']+1):
	
					for no, (file_name, data, target) in enumerate(self.train_data_loader):
	
						data, target = Variable(data), Variable(target)
	
						if self.cuda:
							data, target = data.cuda(), target.cuda()
	
						data = self.model(data)
	
						loss = self.model.loss(data, target, self.training_info)
	
						loss.backward()
	
						if (self.start_no + no)%self.config['cummulative_batch_steps']==0 and (self.start_no + no)!=0:
	
							self.model.opt.step()
	
							self.model.opt.zero_grad()
	
						if (self.start_no + no)%self.config['log_interval_steps'] == 0 and (self.start_no + no) != 0:
	
							self.model.save(no=(self.start_no + no), epoch_i=epoch_i, info = self.training_info)
	
						if (self.start_no + no)%self.config['print_log_steps'] == 0 and (self.start_no + no) != 0:
	
							self.model.print_info(self.training_info)
							log.info()
	
						if (self.start_no + no)%self.config['test_now'] == 0 and (self.start_no + no)!=0:
	
							self.test_model()
	
							self.model.train()
							self.model.requires_grad = True
							self.model.opt.zero_grad()
						if (self.start_no + no) == len(self.train_data_loader) - 1:
							break
	
					self.model.save(no=0, epoch_i=epoch_i, info = self.training_info)
	
					self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}

			return True

		except KeyboardInterrupt:

			return False

	def test_model(self):

		log.info('Testing Mode')

		try:

			if self.work == "classification":
	
				self.model.requires_grad = False
	
				self.model.eval()
	
				for no, (file_name, data, target) in enumerate(self.test_data_loader):
	
					data, target = Variable(data), Variable(target)
	
					if self.cuda:
						data, target = data.cuda(), target.cuda()
	
					data_out = self.model(data)
	
					if not os.path.exists(self.config['dir']['Exp']+'/Temporary'):
						os.mkdir(self.config['dir']['Exp']+'/Temporary')
	
					loss = self.model.loss(data_out, target, self.testing_info)
	
				log.info('Testing Completed successfully')
				log.info('Test Results\n\n', self.testing_info)
	
				if self.testing_info['Acc'] > self.model_best['Acc']:
	
					print("New best model found")
	
					self.model_best['Acc'] = self.testing_info['Acc']
					
					self.model.save(no=0, epoch_i=0, info = self.testing_info)
	
				self.testing_info = {'Loss': 0, 'Acc': 0, 'Keep_log': False, 'Count':0}
	
				log.info('Testing Completed successfully')

			return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False