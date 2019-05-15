import click#Inbuild class
from src.pipeline_manager import PipelineManager#Created by All-AI

from src.logger import Logger#Created by All-AI

@click.group()#Decorators
def main():#main=click.group(main)
	pass

@main.command()#Decorators
@click.option('-p', '--pipeline_name', help='classification,segmentation', required=True)#Decorators	specifing the work performed for example Classification
@click.option('-m', '--model', help='ResNet, IncepNet, AlexNet,UNet' , required=True)#Decorators	specifing the model that will be used to solve this problem
def prepare_metadata(pipeline_name, model):#prepare_metadata=main.command(click.option(click.option(prepare_metadata)))
	pipeline_manager.prepare_metadata(pipeline_name, model)

@main.command()#Decorators
@click.option('-p', '--pipeline_name', help='classification,segmentation', required=True)#Decorators	specifing the work performed for example Classification
@click.option('-m', '--model', help='ResNet, IncepNet, AlexNet,UNet' , required=True)#Decorators	specifing the model that will be used for training the data
def train(pipeline_name, model):#train=main.command(click.option(click.option(train)))
	pipeline_manager.train(pipeline_name, model)  # Training the Model

@main.command()#Decorators
@click.option('-p', '--pipeline_name', help='classification,segmentation', required=True)#Decorators specifing the work performed for example Classification
@click.option('-m', '--model', help='ResNet, IncepNet, AlexNet,UNet' , required=True)#Decorators specfing the model that will be used for training
def test(pipeline_name, model):#test=main.command(click.option(click.option(test)))
	pipeline_manager.test(pipeline_name, model)  # Training the Model

if __name__ == "__main__":#Running the code

	pipeline_manager = PipelineManager()#Calling the PipelineManager
	log = Logger()#Calling the Logger function
	log.first()

	main()#Calling the main function
