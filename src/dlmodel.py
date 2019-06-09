import sys
import torch.utils.data as data
import torch
from tqdm import tqdm
import os
import numpy as np
import random
import importlib

from .data_loader import DataLoader
from .logger import Logger
import configs.config as config


log = Logger()


class DlModel:

	def __init__(self):

		self.cuda = config.cuda and torch.cuda.is_available()
		self.seed()

		self.train_data = DataLoader(config.train, transform=config.train_transform)
		self.test_data = DataLoader(config.test, transform=config.test_transform)

		self.train_data_loader = data.DataLoader(
			self.train_data, batch_size=config.train['batch_size'], shuffle=True,
			num_workers=config.train['cpu_alloc'])
		self.test_data_loader = data.DataLoader(
			self.test_data, batch_size=config.train['batch_size'], shuffle=False,
			num_workers=config.test['cpu_alloc'])

		if config.model.lower() == 'resnet':
			from .model.resnet import resnet152
			self.model = resnet152(pretrained=config.pre_trained_net)
		elif config.model.lower() == 'alexnet':
			from .model.alexnet import alexnet
			self.model = alexnet(pretrained=config.pre_trained_net)

		if self.cuda:
			self.model = self.model.cuda()

		self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count': 0}
		self.testing_info = {'Loss': 0, 'Acc': 0, 'Keep_log': False, 'Count': 0}

		self.model_best = {'Loss': sys.float_info.max, 'Acc': 0.0}

		if self.cuda:
			self.model.cuda()

		self.epoch_start = 0
		self.start_no = 0

		if config.PreTrained:
			self.epoch_start, self.training_info = self.model.load(
				config.pretrained_model['checkpoint'],
				config.pretrained_model['checkpoint_info'])
			self.start_no = int(config.pretrained_model['checkpoint'].split('_')[0])
			self.epoch_start = int(config.pretrained_model['checkpoint'].split('_')[1])
			log.info('Loaded the model')

	def get_config(self):

		importlib.reload(config)

	def seed(self):

		np.random.seed(config.seed)
		random.seed(config.seed)
		torch.manual_seed(config.seed)
		torch.cuda.manual_seed(config.seed)

	def __str__(self):

		return str(config)

	def start_training(self):

		self.model.requires_grad = True

		self.model.train()

		self.model.opt.zero_grad()

	def train_model(self):

		try:

			self.start_training()

			for epoch_i in range(self.epoch_start, config.epoch + 1):

				for no, (file_name, data, target) in enumerate(tqdm(self.train_data_loader)):

					if self.cuda:
						data, target = data.cuda(), target.cuda()

					data = self.model(data)

					loss = self.model.loss(data, target, self.training_info)

					loss.backward()

					if (self.start_no + no) % config.cummulative_batch_steps == 0 and (self.start_no + no) != 0:

						self.model.opt.step()
						self.model.opt.zero_grad()

					if (self.start_no + no) % config.log_interval_steps == 0 and (self.start_no + no) != 0:
						self.model.save(no=(self.start_no + no), epoch_i=epoch_i, info=self.training_info)

					if (self.start_no + no) % config.print_log_steps == 0 and (self.start_no + no) != 0:
						self.model.print_info(self.training_info)
						log.info()

					if (self.start_no + no) % config.test_now == 0 and (self.start_no + no) != 0:

						self.test_model()
						self.start_training()

					if (self.start_no + no) == len(self.train_data_loader) - 1:
						break

				self.model.save(no=0, epoch_i=epoch_i, info=self.training_info)

				self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count': 0}

			return True

		except KeyboardInterrupt:

			return False

	def test_model(self):

		log.info('Testing Mode')

		try:

			self.model.requires_grad = False

			self.model.eval()

			for no, (file_name, data, target) in enumerate(tqdm(self.test_data_loader)):

				if self.cuda:
					data, target = data.cuda(), target.cuda()

				data_out = self.model(data)

				if not os.path.exists(config.dir['Exp'] + '/Temporary'):
					os.mkdir(config.dir['Exp'] + '/Temporary')

				self.model.loss(data_out, target, self.testing_info)

			log.info('Testing Completed successfully')
			log.info('Test Results\n\n', self.testing_info)

			if self.testing_info['Acc'] > self.model_best['Acc']:
				print("New best model found")

				self.model_best['Acc'] = self.testing_info['Acc']

				self.model.save(no=0, epoch_i=0, info=self.testing_info)

			self.testing_info = {'Loss': 0, 'Acc': 0, 'Keep_log': False, 'Count': 0}

			log.info('Testing Completed successfully')

			return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False
