from datetime import datetime
import sys

import configs.config as config


class Logger:
	"""
	This Class is used to log the information printed on the terminal with a time stamp.
	Usage - call log.info(string1, string2 ... ) instead of print(string1, string2 ... )
	"""

	def __init__(self):

		self.write_path = config.dir['Exp']+'/log.txt'
		self.f = open(self.write_path, 'a')

	def first(self):

		self.f.write('\n--------- Starting new session: ' + str(datetime.now().time()) + ' ---------\n\n')
		print('\n--------- Starting new session: ' + str(datetime.now().time()) + ' ---------\n\n', file=sys.stderr)
	
	def info(self, *args):

		temp = ' '.join([str(i) for i in args])
		if "".join(temp.split()) == '\n':
			string = '\n'
		elif "".join(temp.split()) == '':
			string = ''
		else:
			string = str(datetime.now().time())+': '+temp
		print(string)
		self.f.write(string+'\n')
