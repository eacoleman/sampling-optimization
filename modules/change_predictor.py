import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import h5py
from tqdm import tqdm
import gc
import json
import pdb

"""
    ### ChangePredictor

    Description:

        A network architecture for abundance-based inferences on 1D or 2D NMR
        data.

"""
class ChangePredictor(nn.Module):
    def __init__(self, n_channels, plot_size):
        super(ChangePredictor, self).__init__()

        self.n_channels = n_channels
        self.plot_size = plot_size

    def forward(self, x):
        return 0


    def forward(self, x):
        # Separate layers
        x1, x2 = torch.split(x, 1, dim=1)

        return 0


if __name__ == '__main__' :
    n_chan = 1
    n_timestep = 3
    plot_size = 128

    model = ChangePredictor(n_timestep, plot_size)

    input_data = torch.randn(10, n_timestep, plot_size, plot_size)
    #output = model(input_data)

    # print model structure to tensorboard
    writer = SummaryWriter('runs/ChangePredictor_test')
    writer.add_graph(model, input_data)
    writer.close()

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(trainable_params)