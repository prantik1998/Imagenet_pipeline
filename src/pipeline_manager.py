import os
from .dlmodel import DlModel
from .logger import Logger
import configs.config as config
log = Logger()


class PipelineManager:

	def __init__(self):
		for i in config.dir:
			if not os.path.exists(config.dir[i]):
				os.mkdir(config.dir[i])

	def train(self):
		train()

	def test(self):
		test()


def train():

	driver = DlModel()
	driver.train_model()


def test():

	driver = DlModel()
	driver.test_model()
