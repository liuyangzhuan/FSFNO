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

import torch
from stfno.utilities3 import *
from timeit import default_timer

def inferenceTimeTestData_file5(
    S_r,S_theta , T_in,T_out, T_in_steadystate,
    if_IncludeSteadyState, 
    sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
    epochs,
    strn_epochs_dump_path_file5,
    T_out_sub_time_consecutiveIterator_factor, step,
    batch_size,
    model
    ):

    if if_IncludeSteadyState:
        test_a_ntest1size  = torch.zeros(1 ,S_r,S_theta,(T_in+T_in_steadystate) *( sum_vector_a_elements_i_iter ) )
        test_u_ntest1size  = torch.zeros(1 ,S_r,S_theta,T_out   *( sum_vector_u_elements_i_iter ) )
    else:
        test_a_ntest1size  = torch.zeros(1 ,S_r,S_theta,T_in*( sum_vector_a_elements_i_iter ) )
        test_u_ntest1size  = torch.zeros(1 ,S_r,S_theta,T_out   *( sum_vector_u_elements_i_iter ) )
    test_loader_ntest1size  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset( test_a_ntest1size,  test_u_ntest1size), batch_size=batch_size, shuffle=False)
    for ep in range(epochs,epochs+1):
        file5 = open(strn_epochs_dump_path_file5, "a")  
        with torch.no_grad():
            count = -1
            for xx, yy in test_loader_ntest1size:
                xx = xx.to(device)
                yy = yy.to(device)
                count= count +1 
                t0_start = default_timer()
                t1_intermediate = t0_start
                t2_now = t0_start
                for t in range(0, T_out*sum_vector_u_elements_i_iter  , T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter ):
                    im = model(xx)
                    t2_now = default_timer()
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    str_file5= ( str( count) +',' #+ str(t2 - t1)   
                    + str(  t2_now - t1_intermediate ) +','
                    + str(  t2_now - t0_start ) #+','
                    + '\n' )
                    file5.write(str_file5) 
                    t1_intermediate = t2_now
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_tmp = xx[...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                    xx = torch.cat((xx[...,T_out_sub_time_consecutiveIterator_factor * step*sum_vector_u_elements_i_iter:], im), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx = torch.cat((xx, xx_tmp[:]), dim=-1)
                        exit(1)
        file5.close()
    exit(1)