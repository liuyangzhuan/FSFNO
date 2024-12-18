import torch
import torch.nn as nn
from stfno.fourier_transform_3d_layer import SpectralConv3d
from stfno.fourier_transform_3d_layer_jit_torchCompile import SpectralConv3d_jit_torchCompile
from stfno.channelwise_conv_mlp import MLP_3D
import numpy as np
import torch.nn.functional as F

class FNO2d_glob_orig_3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width,T_in,total_vector_elements_i, if_model_jit_torchCompile):
        super(FNO2d_glob_orig_3D, self).__init__()
        self.modes1 = modes1 
        self.modes2 = modes2 
        self.modes3 = modes3 
        self.width = width * total_vector_elements_i
        self.padding = 8 # pad the domain if input is non-periodic
        self.T_input   = T_in * total_vector_elements_i
        self.parameters_out   = total_vector_elements_i
        self.p = nn.Linear( (self.T_input)+3, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        if if_model_jit_torchCompile:
            self.conv0 = SpectralConv3d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv1 = SpectralConv3d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv2 = SpectralConv3d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv3 = SpectralConv3d_jit_torchCompile(self.width, self.width, self.modes1, self.modes2, self.modes3)
        else:
            self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP_3D(self.width, self.width, self.width)
        self.mlp1 = MLP_3D(self.width, self.width, self.width)
        self.mlp2 = MLP_3D(self.width, self.width, self.width)
        self.mlp3 = MLP_3D(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm3d(self.width)
        self.q = MLP_3D(self.width, self.parameters_out , self.width * 4) # output channel is 1: u(x, y)
    def forward(self, x):
        grid = self.get_grid_3D(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 2, 3, 1)
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
        x = x.permute(0, 4, 2, 3, 1)
        return x
    def get_grid_3D(self, shape, device):
        batchsize, size_x, size_y,size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, steps=size_x, dtype=torch.float,device=device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, steps=size_y, dtype=torch.float,device=device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, steps=size_z, dtype=torch.float,device=device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy,gridz), dim=-1).to(device)