import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.utils.data as data

import os
import numpy as np

from scipy.misc import imresize
from PIL import Image

import cv2
import matplotlib.pyplot as plt

class image_dataloader(data.Dataset):

	def __init__(self, root, target_file_add, loader=False, transform=None, target_transform=None,Type = 'Train', factor = 1.5):
		
		self.root = root
		self.target_file_add = target_file_add
		self.IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
		self.factor = factor
		# self.classes, self.class_to_idx = self.find_classes(root)
		# self.imgs = self.make_dataset(self.root, self.class_to_idx)
		self.imgs = []
		self.target = self.get_targets()

		if len(self.imgs) == 0:
			raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
							   "Supported image extensions are: " + ",".join(self.IMG_EXTENSIONS)))
		self.Type = Type
		
		self.transform = transform
		self.target_transform = target_transform

		if not loader:
			self.loader = self.default_loader
		else:
			self.loader = loader		

		self.img_size = 512

	def aspect_resize(self, img, n_height, center_aspect=1, target_all=[[0, 0, 0, 0]]):

		# A 5by5 image would have a center of center_aspect. A nbyn would have a center of n//self.factor*center_aspect.
	
		width, height = img.size
		
		length = max(width, height)

		blank = np.zeros([length, length, 3]).astype(np.uint8)

		blank[(length - height)//2:(length + height)//2, (length - width)//2:(length + width)//2, :] = np.array(img)

		blank_target_center = np.zeros([n_height, n_height]).astype(np.uint8)

		for target in target_all:

			new_width, new_height = int(target[2]/length*n_height), int(target[3]/length*n_height)

			center = [(length - height)//2 + target[1] + target[3]//2, (length - width)//2 + target[0] + target[2]//2]

			center = (np.array(center)/length*n_height).astype(np.int32)		

			blank_target_center[center[0] - int((new_height/self.factor*center_aspect)/2) : center[0] + int((new_height/self.factor*center_aspect)/2), center[1] - int((new_width/self.factor*center_aspect)/2) : center[1] + int((new_width/self.factor*center_aspect)/2)] = 255

		target_center = Image.fromarray(blank_target_center, mode='L')

		img = Image.fromarray(blank, 'RGB')

		return img.resize((n_height, n_height), Image.ANTIALIAS), target_center

	def get_all_names_refresh(self):

		# self.classes, self.class_to_idx = self.find_classes(self.root)
		# self.imgs = self.make_dataset(self.root, self.class_to_idx)

		# self.imgs = os.listdir(self.root)
		self.targets = self.get_targets()

		if len(self.imgs) == 0:
			raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
							   "Supported image extensions are: " + ",".join(self.IMG_EXTENSIONS)))

	def get_targets(self):

		self.imgs = []

		with open(self.target_file_add) as f:
			all_targets = f.readlines()

		target = []

		counter = 0

		for i in range(len(os.listdir(self.root))):

			self.imgs.append(all_targets[counter].split('/')[-1][:-1])

			# print(all_targets[counter].split('/')[-1][:-1])

			counter += 1

			number_of_faces = int(all_targets[counter])

			counter += 1

			temp_target = []

			for j in range(number_of_faces):

				x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = all_targets[counter].split()

				x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = int(x1), int(y1), int(w), int(h), int(blur), int(expression), int(illumination), int(invalid), int(occlusion), int(pose)

				temp_target.append([x1, y1, w, h])

				counter += 1

			target.append(temp_target)

		return target

	def is_image_file(self, filename):
		
		filename_lower = filename.lower()
		return any(filename_lower.endswith(ext) for ext in self.IMG_EXTENSIONS)

	# def find_classes(self, dir):
		
	# 	classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
	# 	classes.sort()
	# 	#print(classes)
	# 	class_to_idx = {classes[i]: i for i in range(len(classes))}
	# 	return classes, class_to_idx


	# def make_dataset(self, dir, class_to_idx):
	# 	images = []
	# 	dir = os.path.expanduser(dir)
	# 	for target in sorted(os.listdir(dir)):
	# 		d = os.path.join(dir, target)
	# 		if not os.path.isdir(d):
	# 			continue

	# 		for root, _, fnames in sorted(os.walk(d)):
	# 			for fname in sorted(fnames):
	# 				if self.is_image_file(fname):
	# 					path = os.path.join(root, fname)
	# 					item = (path, class_to_idx[target])
	# 					images.append(item)

	# 	return images


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

	def set_size(self, value):

		self.img_size = value

	def show_img(self, img):

		plt.imshow(img)
		plt.show()

	def __getitem__(self, index):
		
		path, target = self.imgs[index], self.target[index]

		img_p = self.loader(self.root+'/'+path)

		img_p, target_p = self.aspect_resize(img_p, self.img_size, target_all = target)

		img_p = self.transform(img_p)
		target_p = self.target_transform(target_p).long()

		return path, img_p, target_p
		
	def __len__(self):
		return len(self.imgs)
