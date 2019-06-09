import os
from torchvision import transforms

dir = {
	'Datasets': 'Datasets',
	'train_dir': 'Datasets/Train',
	'test_dir': 'Datasets/Test',
	'Exp': 'Exp',
}
if not os.path.exists(dir['Exp']):
	os.mkdir(dir['Exp'])

train = {
		'transform': False,
		'batch_size': 20,
		'cpu_alloc': 6,
		'Type': 'Train',
		'loader': {
			'flag': False
		},
		'image_size': 224,
		'dir': '<Please Put Your Directory Path for Train Images>'
}

test = {
		'transform': False,
		'batch_size': 20,
		'cpu_alloc': 6,
		'Type': 'Test',
		'loader': {
			'flag': False
		},
		'image_size': 224,
		'dir': '<Please Put Your Directory Path for Test Images>'
}

augmentation = {
	'brightness': 0.2,
	'contrast': 0.2,
	'saturation': 0.2,
	'hue': 0.2,
}

train_transform = transforms.Compose(
	[
		transforms.ColorJitter(
			brightness=augmentation['brightness'],
			contrast=augmentation['contrast'],
			saturation=augmentation['saturation'],
			hue=augmentation['hue']), transforms.ToTensor(),
	]
)

test_transform = transforms.Compose(
	[
		transforms.ToTensor(),
	]
)

model = 'ResNet'
pre_trained_net = True
PreTrained = False

pretrained_model = {
	'checkpoint': '0_0_checkpoint.pth.tar',
	'checkpoint_info': '0_0_checkpoint_info.pth.tar',
}

# Logging

log_interval_steps = 50
print_log_steps = 50
test_now = 500

# Parameters

image_size = 224
n_channels = 3
n_classes = 1000
lr = 0.0001
optimizer = 'Adam'
lossf = 'CEL'
cuda = True
seed = 2
rank = 5
epoch = 100
cummulative_batch_steps = 1
