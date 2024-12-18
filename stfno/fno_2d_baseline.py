import torch
import torch.nn as nn
from stfno.fourier_transform_2d_layer import SpectralConv2d
from stfno.fourier_transform_2d_layer_jit_torchCompile import SpectralConv2d_jit_torchCompile
from stfno.channelwise_conv_mlp import MLP
import numpy as np
import torch.nn.functional as F

class FNO2d_glob_orig(nn.Module):
    def __init__(self, modes1, modes2, width,T_in,sum_vector_a_elements_i_iter,T_out,sum_vector_u_elements_i_iter,number_of_layers,if_model_jit_torchCompile):
        super(FNO2d_glob_orig, self).__init__()
        self.modes1 = modes1 
        self.modes2 = modes2 
        self.width = width * sum_vector_a_elements_i_iter
        self.padding = 8 # pad the domain if input is non-periodic
        self.T_input   = T_in * sum_vector_a_elements_i_iter
        self.parameters_out   = T_out * sum_vector_u_elements_i_iter
        self.p = nn.Linear( (self.T_input)+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        if if_model_jit_torchCompile:
            self.conv0 = SpectralConv2d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2)
            self.conv1 = SpectralConv2d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2)
            self.conv2 = SpectralConv2d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2)
            self.conv3 = SpectralConv2d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2)
        else:
            self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, self.parameters_out , self.width * 4) # output channel is 1: u(x, y)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, steps=size_x, dtype=torch.float,device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, steps=size_y, dtype=torch.float,device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)