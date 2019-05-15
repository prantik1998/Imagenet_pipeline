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

		self.work = kwargs['work']
		self.Type = config['Type']

		if self.work=="classification":
			self.root=config["dir"]
		if self.work=="segmenation":
			self.img_dir=config["img_dir"]
			self.target=config["tar_class"]
			self.filename=config[self.Type]
		self.IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

		self.get_all_names_refresh()


		self.transform = kwargs['transform']
		self.target_transform = kwargs['target_transform']

		if not config['loader']['flag']:
			self.loader = self.default_loader
		else:
			self.loader = kwargs['loader']

		self.img_size = config['image_size']

	def get_all_names_refresh(self):

		if self.work=="classification":
			self.classes, self.class_to_idx = self.find_classes(self.root)
			self.imgs = self.make_dataset(self.root,self.class_to_idx)

			if len(self.imgs) == 0:
				raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
								   "Supported image extensions are: " + ",".join(self.IMG_EXTENSIONS)))

		elif self.work=="segmentation":
			f=open(self.filename,"r")
			a=f.read().split("\n")
			self.imgs=[i+".jpg" for i in a if i!=""]
			self.target=[i+".png" for i in a if i!=""]


	def find_classes(self, dir):

		classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

	def make_dataset(self, dir, class_to_idx):
		if self.work=="classification":
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

