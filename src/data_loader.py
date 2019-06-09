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

	def aspect_resize(self, img, n_height):

		width, height = img.size
		
		length = max(width, height)

		blank = np.zeros([length, length, 3]).astype(np.uint8)

		blank[(length - height)//2:(length + height)//2, (length - width)//2:(length + width)//2, :] = np.array(img)

		img = Image.fromarray(blank, 'RGB')

		return img.resize((n_height, n_height), Image.ANTIALIAS)

	def get_all_names_refresh(self):

		self.classes, self.class_to_idx = self.find_classes(self.root)
		self.imgs = self.make_dataset(self.root, self.class_to_idx)

		if len(self.imgs) == 0:
			raise(RuntimeError(
				"Found 0 images in subfolders of: " + self.root + "\n"
				"Supported image extensions are: " + ",".join(self.IMG_EXTENSIONS)))

	def is_image_file(self, filename):
		
		filename_lower = filename.lower()
		return any(filename_lower.endswith(ext) for ext in self.IMG_EXTENSIONS)

	def find_classes(self, dir):
		
		classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

	def make_dataset(self, dir_, class_to_idx):

		images = []
		dir_ = os.path.expanduser(dir_)
		for target in sorted(os.listdir(dir_)):
			d = os.path.join(dir_, target)
			if not os.path.isdir(d):
				continue

			for root, _, fnames in sorted(os.walk(d)):
				for fname in sorted(fnames):
					if self.is_image_file(fname):
						path = os.path.join(root, fname)
						item = (path, class_to_idx[target])
						images.append(item)

		return images

	def loader(self, path):
		# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

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
