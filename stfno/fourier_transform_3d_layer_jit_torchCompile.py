import torch
import torch.nn as nn


class SpectralConv3d_jit_torchCompile(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_jit_torchCompile, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
    def _get_weight(self, index):
        return self.weight[index]
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        fft_norm='forward' #'backward'
        x_ft = torch.fft.rfftn(x.float(), norm=fft_norm, dim=[-3, -2, -1])
        out_fft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)// 2 + 1, dtype=torch.cfloat , device=x.device)
        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.modes1 // 2),  # :half_n_modes[0],
            slice(self.modes2 // 2),  # :half_n_modes[1],
            slice(self.modes3),  # :half_n_modes[2]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(self.modes1 // 2),  # ...... :half_n_modes[0],
            slice(-self.modes2 // 2, None),  # -half_n_modes[1]:,
            slice(self.modes3),  # ......      :half_n_modes[0]]
        )
        slices2 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.modes1 // 2, None),  # -half_n_modes[0]:,
            slice(self.modes2 // 2),  # ...... :half_n_modes[1],
            slice(self.modes3),  # ......      :half_n_modes[2]]
        )
        slices3 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.modes1 // 2, None),  # -half_n_modes[0],
            slice(-self.modes2 // 2, None),  # -half_n_modes[1],
            slice(self.modes3),  # ......      :half_n_modes[2]]
        )
        out_fft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_fft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_fft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights3)
        out_fft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_fft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1], norm=fft_norm)
        return x