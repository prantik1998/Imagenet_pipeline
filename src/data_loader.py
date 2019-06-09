import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .logger import Logger

log = Logger()


class DataLoader(data.Dataset):

	def __init__(self, config, **kwargs):
		
		self.root = config['dir']
		self.IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
		self.imgs = []
		self.classes = []
		self.class_to_idx = []

		self.get_all_names_refresh()

		self.Type = config['Type']
		self.transform = kwargs['transform']
		self.img_size = config['image_size']

	@staticmethod
	def aspect_resize(img, n_height):

		"""

		:param img: PIL image
		:param n_height: height to which we have to aspect resize (Only supports square images)
		:return:
		"""

		width, height = img.size
		
		length = max(width, height)

		blank = np.zeros([length, length, 3]).astype(np.uint8)

		blank[(length - height)//2:(length + height)//2, (length - width)//2:(length + width)//2, :] = np.array(img)

		img = Image.fromarray(blank, 'RGB')

		return img.resize((n_height, n_height), Image.ANTIALIAS)

	def get_all_names_refresh(self):

		"""
		Refreshes the dataset file list
		:return:
		"""

		self.classes, self.class_to_idx = self.find_classes(self.root)
		self.imgs = self.make_data_set(self.root, self.class_to_idx)

		if len(self.imgs) == 0:
			raise(RuntimeError(
				"Found 0 images in sub folders of: " + self.root + "\n"
				"Supported image extensions are: " + ",".join(self.IMG_EXTENSIONS)))

	def is_image_file(self, filename):

		"""
		Function to check if the filename has image type extension
		:param filename:
		:return:
		"""
		
		filename_lower = filename.lower()
		return any(filename_lower.endswith(ext) for ext in self.IMG_EXTENSIONS)

	@staticmethod
	def find_classes(dir_):

		"""
		This will get the path of the dataset and create a dictionary having class name with corresponding idx
		*Remember that the class is sorted according to string lexical order.
		So string lexical order is 0, 1, 10, 11, 12, 13, 14, ...
		:param dir_: Path of the dataset
		:return:
		"""
		
		classes = [d for d in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

	def make_data_set(self, dir_, class_to_idx):

		"""
		Generates the file_names from base_dir and class_to_idx
		:param dir_: Base directory of the dataset
		:param class_to_idx: Created in find_classes
		:return:
		"""

		images = []
		dir_ = os.path.expanduser(dir_)
		for target in sorted(os.listdir(dir_)):
			d = os.path.join(dir_, target)
			if not os.path.isdir(d):
				continue

			for root, _, fnames in sorted(os.walk(d)):
				for f_name in sorted(fnames):
					if self.is_image_file(f_name):
						path = os.path.join(root, f_name)
						item = (path, class_to_idx[target])
						images.append(item)

		return images

	@staticmethod
	def loader(path):

		"""
		Used for opening an image in PIL format and automatically converting it to RGB
		:param path: path of the image file we are trying to open
		:return:
		"""

		# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	@staticmethod
	def show_img(img):

		"""
		Used for showing the image using matplotlib
		:param img: numpy array having dimension like images in RGB format
		:return:
		"""

		plt.imshow(img)
		plt.show()

	def __getitem__(self, index):
		
		path, target = self.imgs[index]
		img_p = self.loader(path)
		img_p = self.aspect_resize(img_p, self.img_size)
		img_p = self.transform(img_p)

		return path, img_p, target
		
	def __len__(self):
		return len(self.imgs)
