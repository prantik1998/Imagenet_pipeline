# Author - Mayank Kumar Singh

import configs.config as config
import click
from src.pipeline_manager import PipelineManager
from src.logger import Logger


@click.group()
def main():
	pass


@main.command()
def train():
	"""
		Call this to train the model. To run - python main.py train
	"""
	pipeline_manager.train()


@main.command()
def test():
	"""
		Call this to test the model. To run - python main.py test
	"""
	pipeline_manager.test()


if __name__ == "__main__":

	pipeline_manager = PipelineManager()
	log = Logger()
	log.first()
	main()
