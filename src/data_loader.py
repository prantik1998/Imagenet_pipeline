import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.utils.data as data

import os
import numpy as np

from scipy.misc import imresize
from PIL import Image

import cv2
import matplotlib.pyplot as plt
from .logger import Logger

log = Logger()

class DataLoader(data.Dataset):

	def __init__(self, config, **kwargs):
		
		self.root = config['dir']
		self.IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

		self.get_all_names_refresh()

		self.Type = config['Type']
		
		self.transform = kwargs['transform']
		self.target_transform = kwargs['target_transform']

		if not config['loader']['flag']:
			self.loader = self.default_loader
		else:
			self.loader = kwargs['loader']

		self.img_size = config['image_size']

	def aspect_resize(self, img, n_height, center_aspect=1, target_all=[[0, 0, 0, 0]]):

		# A 5by5 image would have a center of center_aspect. A nbyn would have a center of n//self.factor*center_aspect.
	
		width, height = img.size
		
		length = max(width, height)

		blank = np.zeros([length, length, 3]).astype(np.uint8)

		blank[(length - height)//2:(length + height)//2, (length - width)//2:(length + width)//2, :] = np.array(img)

		for target in target_all:

			new_width, new_height = int(target[2]/length*n_height), int(target[3]/length*n_height)

			center = [(length - height)//2 + target[1] + target[3]//2, (length - width)//2 + target[0] + target[2]//2]

			center = (np.array(center)/length*n_height).astype(np.int32)		

		img = Image.fromarray(blank, 'RGB')

		return img.resize((n_height, n_height), Image.ANTIALIAS)

	def get_all_names_refresh(self):

		self.classes, self.class_to_idx = self.find_classes(self.root)
		self.imgs = self.make_dataset(self.root, self.class_to_idx)

		if len(self.imgs) == 0:
			raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
							   "Supported image extensions are: " + ",".join(self.IMG_EXTENSIONS)))

	def is_image_file(self, filename):
		
		filename_lower = filename.lower()
		return any(filename_lower.endswith(ext) for ext in self.IMG_EXTENSIONS)

	def find_classes(self, dir):
		
		classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

	def make_dataset(self, dir, class_to_idx):
		images = []
		dir = os.path.expanduser(dir)
		for target in sorted(os.listdir(dir)):
			d = os.path.join(dir, target)
			if not os.path.isdir(d):
				continue

			for root, _, fnames in sorted(os.walk(d)):
				for fname in sorted(fnames):
					if self.is_image_file(fname):
						path = os.path.join(root, fname)
						item = (path, class_to_idx[target])
						images.append(item)

		return images

	def pil_loader(self, path):
		# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def accimage_loader(self, path):
		try:
			return accimage.Image(path)
		except IOError:
			# Potentially a decoding problem, fall back to PIL.Image
			return self.pil_loader(path)

	def default_loader(self, path):
		from torchvision import get_image_backend
		if get_image_backend() == 'accimage':
			return self.accimage_loader(path)
		else:
			return self.pil_loader(path)

	def show_img(self, img):

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
