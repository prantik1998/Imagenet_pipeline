import numpy as np
import torch
import matplotlib.pyplot as plt

x = torch.load('0_0_info_checkpoint.pth.tar')

print(np.mean(x['Acc']))