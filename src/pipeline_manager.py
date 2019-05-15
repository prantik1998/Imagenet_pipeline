import yaml#Inbuild function for reading files
import os#Inbuild function 
from .dl_model import dl_model#Created by the coder
from .read_yaml import read_yaml#Created by the coder
from .logger import Logger#Created by the coder
from .prepare_metadata import prepare_metadata
log = Logger()#Calling the function

class PipelineManager():#Creating the PipelineManger

	def __init__(self):#Constructor
		self.config_file = read_yaml()#Calling the function read_yaml
		for i in self.config_file['dir']:#
			if not os.path.exists(self.config_file['dir'][i]):
				os.mkdir(self.config_file['dir'][i])		
		
	def prepare_metadata(self, pipeline_name, model_name):
		if pipeline_name == 'classification':
			implemented = ['ResNet', 'AlexNet', 'InceptionNet']
			if model_name in implemented:
				prepare_metadata(self.config_file['prepare_metadata'][model_name])
			else:
				log.info(model_name, ' Model not yet implemented')
		elif pipeline_name == 'segmentation':
			implemented = ['UNet']
			if model_name in implemented:
				prepare_metadata(self.config_file['prepare_metadata'][model_name])
			else:
				log.info(model_name, ' Model not yet implemented')
		else:
			log.info(pipeline_name, ' Pipeline not yet implemented')

	def train(self, pipeline_name, model_name):
		if pipeline_name == 'classification':
			implemented = ['ResNet', 'AlexNet', 'InceptionNet']
			if model_name in implemented:
				driver = dl_model(model_name)
				driver.train_model()
			else:
				log.info(model_name, ' Model not yet implemented')
		elif pipeline_name == 'segmentation':
			implemented = ['UNet']
			if model_name in implemented:
				driver = dl_model(model_name, work="segmentation")
				driver.train_model()
			else:
				log.info(model_name, ' Model not yet implemented')
		else:
			log.info(pipeline_name, ' Pipeline not yet implemented')

	def test(self, pipeline_name, model_name):
		test(pipeline_name, model_name)







