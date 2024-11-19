import torch
import torch.nn as nn

class SpectralConv2d_jit_torchCompile(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_jit_torchCompile, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    def compl_mul2d(self, input, weights):
        return self.custom_einsum_1(input, weights)
    def custom_einsum_1(self,input,weights):
        r_input = torch.real(input)
        i_input = torch.imag(input)
        r_weights = torch.real(weights)
        i_weights = torch.imag(weights)
        r_out = torch.einsum("bixy,ioxy->boxy", r_input, r_weights)- torch.einsum("bixy,ioxy->boxy", i_input, i_weights)
        i_out = torch.einsum("bixy,ioxy->boxy", r_input, i_weights)+ torch.einsum("bixy,ioxy->boxy", i_input, r_weights)
        return torch.complex(r_out,i_out)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x