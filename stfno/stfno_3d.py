# STFNO Copyright (c) 2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.

#-----------------------------------------------------------------
#     Mustafa Rahman, Yang Liu
#     Copyright 2024
#     Lawrence Berkeley National Laboratory
#     All Rights Reserved
#-----------------------------------------------------------------

#     Revision 1.1  2024/08/20 15:30:00  mustafar
#     Original source.

#     STFNO code: Sparsified Time-dependent PDEs FNO code 
#-----------------------------------------------------------------


import torch
import torch.nn as nn
from stfno.fourier_transform_3d_layer import SpectralConv3d
from stfno.fourier_transform_3d_layer_jit_torchCompile import SpectralConv3d_jit_torchCompile
from stfno.channelwise_conv_mlp import MLP, MLP_3D
import numpy as np
import torch.nn.functional as F


class FNO2d_NIMRODglobal_3D(nn.Module):
    def  __init__(self, modes1, modes2,modes3, width,T_in,
                                                total_vector_a_elements_i,T,
                                                total_vector_u_elements_i,number_of_layers,
                                                input_parameter_order, 
                                                mWidth_input_parameters, 
                                                nWidth_output_parameters,
                                                if_model_jit_torchCompile):
        super(FNO2d_NIMRODglobal_3D, self).__init__()        
        # if total_vector_u_elements_i == 10:
        #     self.input_parameter_order = [[0,1,2,3,4,5,6,7,8,9],[3,4,5,0,1,2],[6,0,1,2,3,4,5,7,8,9],[7,8,9,0,1,2]]
        #     self.mWidth_input_parameters  = [10, 6,10, 6]
        #     self.nWidth_output_parameters = [ 3, 3, 1, 3]
        # elif total_vector_a_elements_i == 4:
        #     self.input_parameter_order = [[0,1,2,3]]
        #     self.mWidth_input_parameters  = [4]
        #     self.nWidth_output_parameters = [ 3]
        # elif total_vector_a_elements_i == 3:
        #     self.input_parameter_order = [[0,1,2]]
        #     self.mWidth_input_parameters  = [3]
        #     self.nWidth_output_parameters = [ 3]
        # elif total_vector_u_elements_i == 3:
        #     self.input_parameter_order = [[0,1,2,3,4,5,6,7,8,9]]
        #     self.mWidth_input_parameters  = [10]
        #     self.nWidth_output_parameters = [ 3]
        # else: 
        #     exit(1)
        self.input_parameter_order = input_parameter_order
        self.mWidth_input_parameters  = mWidth_input_parameters
        self.nWidth_output_parameters = nWidth_output_parameters
        self.modes1 = modes1 
        self.modes2 = modes2 
        self.modes3 = modes3
        self.width_in  = width * total_vector_a_elements_i
        self.width_intermediate  = width * 3
        self.width_out  = width * total_vector_u_elements_i
        self.width = width #* total_vector_a_elements_i
        self.padding   = 8 # pad the domain if input is non-periodic
        self.T_in      = T_in
        self.T_out     = T
        self.total_vector_a_elements_i= total_vector_a_elements_i
        self.total_vector_u_elements_i= total_vector_u_elements_i
        self.n_layers = number_of_layers
        self.p_linears = nn.ModuleList([nn.Linear((self.T_in)+3, self.width) for i in range(self.total_vector_a_elements_i)])
        self.conv_linears = nn.ModuleList()
        if if_model_jit_torchCompile:
            for j in range(self.n_layers):
                    self.conv_linears.append(  nn.ModuleList([SpectralConv3d_jit_torchCompile(self.width * self.mWidth_input_parameters[i], self.width * self.nWidth_output_parameters[i], self.modes1, self.modes2, self.modes3) for i in range(len(self.mWidth_input_parameters))])  )
        else:
            for j in range(self.n_layers):
                    self.conv_linears.append(  nn.ModuleList([SpectralConv3d(self.width * self.mWidth_input_parameters[i], self.width * self.nWidth_output_parameters[i], self.modes1, self.modes2, self.modes3) for i in range(len(self.mWidth_input_parameters))])  )
        self.mlp_linears = nn.ModuleList()
        for j in range(self.n_layers):
                self.mlp_linears.append( nn.ModuleList([MLP_3D(self.width * self.nWidth_output_parameters[i], self.width * self.nWidth_output_parameters[i], self.width * self.nWidth_output_parameters[i]) for i in range(len(self.mWidth_input_parameters))]) )
        self.w_linears = nn.ModuleList()
        for j in range(self.n_layers):
                self.w_linears.append(  nn.ModuleList([nn.Conv3d(self.width* self.mWidth_input_parameters[i], self.width*  self.nWidth_output_parameters[i], 1) for i in range(len(self.mWidth_input_parameters))]) )
        self.norm_linears_1 = nn.ModuleList()
        for j in range(self.n_layers):
                self.norm_linears_1.append(  nn.ModuleList([nn.InstanceNorm3d(self.width * self.mWidth_input_parameters[i]  ) for i in range(len(self.mWidth_input_parameters))]) )
        self.norm_linears_2 = nn.ModuleList()
        for j in range(self.n_layers):
                self.norm_linears_2.append(  nn.ModuleList([nn.InstanceNorm3d(self.width * self.nWidth_output_parameters[i]  ) for i in range(len(self.mWidth_input_parameters))]) )
        self.q_linears = nn.ModuleList([MLP_3D(self.width* self.nWidth_output_parameters[i] , self.T_out  * self.nWidth_output_parameters[i] , self.width * self.nWidth_output_parameters[i]* 4) for i in range(len(self.nWidth_output_parameters))])
    def forward(self, x):
        grid = self.get_grid_3D(x.shape, x.device)
        for i, l in enumerate(self.p_linears):   
            if i == len(self.p_linears)-1:
                x1=(torch.cat((x[:,:,:,:,((i-len(self.p_linears))*self.T_in):], grid),dim=-1))
                x1=self.p_linears[i](x1)
                x = torch.cat(  (x[...,:((i-len(self.p_linears))*self.T_in)],x1 ),dim=-1)
            else:
                x1=(torch.cat((x[:,:,:,:,((i-len(self.p_linears))*self.T_in):((i+1-len(self.p_linears))*self.T_in)], grid),dim=-1))
                x1=self.p_linears[i](x1)
                x = torch.cat(  (x[...,:((i-len(self.p_linears))*self.T_in)],x1,x[...,((i+1-len(self.p_linears))*self.T_in):] ),dim=-1)
        x = x.permute(0, 4, 1, 2, 3)
        for i_n_layers in range(self.n_layers):
            x3 =[]
            for i in range(1):  
                x2_tmp = x[ :, self.width* self.input_parameter_order[i][0] :self.width* (self.input_parameter_order[i][0] + 1),:,:,: ]
                for j in range(len(self.input_parameter_order[i])-1):
                    x2_tmp = torch.cat( (x2_tmp, x[ :, self.width* self.input_parameter_order[i][j] :self.width* (self.input_parameter_order[i][j] + 1),:,:,:]), dim=-4) 
                x3.append(x2_tmp)
            x1=[]
            for i in range(1):  
                x1.append( self.norm_linears_1[i_n_layers][i](x3[i]) )
            for i in range(1): 
                x1[i]=self.conv_linears[i_n_layers][i](x1[i])
            for i in range(1): 
                x1[i]=self.norm_linears_2[i_n_layers][i](x1[i])
            for i in range(1): 
                x1[i]=self.mlp_linears[i_n_layers][i](x1[i])
            x2=[]
            for i in range(1): 
                x2.append(self.w_linears[i_n_layers][i](x3[i]) )
            x [:,self.width *(0): self.width * (0+self.nWidth_output_parameters[0]),:,:,:]= x1[0] + x2[0]
            sum_nWidth_output_parameters_SoFar = 3
            for i in range(1,len(self.mWidth_input_parameters)):  
                x2_tmp = x[ :, self.width* self.input_parameter_order[i][0] :self.width* (self.input_parameter_order[i][0] + 1),:,:,: ]
                for j in range(len(self.input_parameter_order[i])-1):
                    x2_tmp = torch.cat( (x2_tmp, x[ :, self.width* self.input_parameter_order[i][j] :self.width* (self.input_parameter_order[i][j] + 1),:,:,:]), dim=-4) 
                x3.append(x2_tmp)
                x1.append( self.norm_linears_1[i_n_layers][i](x3[i]) )
                x1[i]=self.conv_linears[i_n_layers][i](x1[i])
                x1[i]=self.norm_linears_2[i_n_layers][i](x1[i])
                x1[i]=self.mlp_linears[i_n_layers][i](x1[i])
                x2.append(self.w_linears[i_n_layers][i](x3[i]) )        
                x[:,self.width *(i): self.width * (i+self.nWidth_output_parameters[i]),:,:,:]= x1[i] + x2[i]
            if i_n_layers != (self.n_layers-1):
                x = F.gelu(x)
        Solve_q_linearsWithVariable_x1_AndNotAsSelfModificationOfx = True
        if Solve_q_linearsWithVariable_x1_AndNotAsSelfModificationOfx:
            for i, l in enumerate(self.q_linears):   
                if i == 0 :
                    x1 = self.q_linears[i] ( x[:,self.width * self.nWidth_output_parameters[i]*(i): self.width * self.nWidth_output_parameters[i] * (i+1),:,:])
                else:
                    x1 = torch.cat((x1, self.q_linears[i](x[:,self.T_out * self.nWidth_output_parameters[i]*(i): self.T_out * self.nWidth_output_parameters[i]*(i) + self.width* self.nWidth_output_parameters[i] ,:,:,:]) ),dim=-4)
            x= x1
        else:
            for i, l in enumerate(self.q_linears):   
                if i == 0 and len(self.nWidth_output_parameters)!=1:
                    x = torch.cat((self.q_linears[i] ( x[:,self.width * self.nWidth_output_parameters[i]*(i): self.width * self.nWidth_output_parameters[i] * (i+1),:,:]),x[:,self.width * self.nWidth_output_parameters[i] *(i+1): ,:,:,:] ),dim=-4)
                elif i == 0 and len(self.nWidth_output_parameters)==1:
                    x = self.q_linears[i] ( x[:,self.width * self.nWidth_output_parameters[i]*(i): self.width * self.nWidth_output_parameters[i] * (i+1),:,:]) 
                elif i == len(self.q_linears)-1:
                    x = torch.cat(( x[:,: self.T_out * self.nWidth_output_parameters[i]* (i),:,:],self.q_linears[i] (x[:,self.T_out * self.nWidth_output_parameters[i] *(i):,:,:,:] )),dim=-4)
                else:
                    x = torch.cat((x[:,:self.T_out * self.nWidth_output_parameters[i]*(i) ,:,:], self.q_linears[i](x[:,self.T_out * self.nWidth_output_parameters[i]*(i): self.T_out * self.nWidth_output_parameters[i]*(i) + self.width* self.nWidth_output_parameters[i] ,:,:]),x[:,(self.T_out * self.nWidth_output_parameters[i]*(i) + self.width* self.nWidth_output_parameters[i]): ,:,:,:] ),dim=-4)
        x = x.permute(0, 2, 3,4, 1)
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