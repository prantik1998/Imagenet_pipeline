import os
from .dlmodel import DlModel
from .logger import Logger
import configs.config as config
log = Logger()


class PipelineManager:
	"""
	This Class should be used as a control center to run different functions
	"""

	def __init__(self):

		for i in config.dir:
			if not os.path.exists(config.dir[i]):
				os.mkdir(config.dir[i])

	@staticmethod
	def train():
		train()

	@staticmethod
	def test():
		test()


def train():

	driver = DlModel()
	driver.train_model()


def test():

	driver = DlModel()
	driver.test_model()
