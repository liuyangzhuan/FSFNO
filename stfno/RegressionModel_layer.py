# Sparsified Time-dependent PDEs FNO (STFNO) Copyright (c) 2025, The Regents of 
# the University of California, through Lawrence Berkeley National Laboratory 
# (subject to receipt of any required approvals from the U.S.Dept. of Energy).  
# All rights reserved.
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

import torch
import torch.nn as nn
from stfno.utilities3 import *

class RegressionModel(nn.Module):
    def __init__(self,T_in,total_vector_a_elements_i,T_out_sub_time_consecutiveIterator_factor,total_vector_u_elements_i,number_of_layers):
        super(RegressionModel, self).__init__()
        self.parameters_in  = T_in * total_vector_a_elements_i
        self.parameters_out = T_out_sub_time_consecutiveIterator_factor * total_vector_u_elements_i
        self.fc1 = nn.Linear(self.parameters_in, (10 * (self.parameters_in + self.parameters_out ) )//2  ).to(device)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear((10 * (self.parameters_in + self.parameters_out ) )//2, self.parameters_out).to(device)
    def forward(self, x):
        x = self.fc1(x.to(device))
        x = self.gelu(x)
        x = self.fc2(x)
        return x