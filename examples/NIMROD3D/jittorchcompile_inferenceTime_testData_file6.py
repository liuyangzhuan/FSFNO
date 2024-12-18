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

import numpy as np
from stfno.utilities3 import *
from timeit import default_timer
from relativeError_eachTestDataSample_file4 import relativeErrorEachTestDataSample_file4
from inferenceTime_testData_file5 import inferenceTimeTestData_file5
import torch._dynamo

    
def jittorchcompile_inferenceTimeTestData_file6(
            ntest,
            S,
            T_out, 
            sum_vector_u_elements_i_iter,
            epochs,
            strn_epochs_dump_path_file6,
            T_out_sub_time_consecutiveIterator_factor, step,
            if_model_jit_torchCompile,
            model,
            test_loader,            
            count_params_model
            ):
    torch._dynamo.reset()
    gridx = np.arange(S)
    gridy = np.arange(S)
    xi, yi = np.meshgrid(gridx, gridy)

    
    # if if_IncludeSteadyState:
    #     if if_model_Nimrod_STFNO_global:
    #         model = FNO2d_global            (modes, modes, width,(T_in+T_in_steadystate),sum_vector_a_elements_i_iter,T_out,sum_vector_u_elements_i_iter,number_of_layers).cuda()
    #     else:
    #         model = FNO2d_glob_orig         (modes, modes, width,(T_in+T_in_steadystate),sum_vector_a_elements_i_iter).cuda()
    # else:
    #     if if_model_Nimrod_STFNO_global:
    #         model = FNO2d_global            (modes, modes, width,(T_in),
    #                                             sum_vector_a_elements_i_iter,T_out,
    #                                             sum_vector_u_elements_i_iter,number_of_layers,
    #                                             input_parameter_order, 
    #                                             mWidth_input_parameters, 
    #                                             nWidth_output_parameters,
    #                                             if_model_jit_torchCompile).cuda()
    #         if if_model_jit_torchCompile:
    #             model = torch.compile(model, backend="cudagraphs") 
    #     else:
    #         model = FNO2d_glob_orig         (modes, modes, width,(T_in)  ,sum_vector_a_elements_i_iter,T_out,sum_vector_u_elements_i_iter,number_of_layers,
    #                                          if_model_jit_torchCompile).cuda()
    #         if if_model_jit_torchCompile:
    #             model = torch.compile(model, backend="cudagraphs")     
    # model = torch.compile(model, backend="cudagraphs")
    # print(count_params(model))
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    # if if_model_jit_torchCompile:
    model = torch.compile(model, backend="cudagraphs")    
    # with torch.no_grad():
    if if_model_jit_torchCompile:
        print(" Testing the data inference")
        myloss = LpLoss_fieldElements(size_average=True)
        myloss_MaxNormRel =LpLoss_MaxNormRel_fieldElements(size_average=True)
        for ep in range(epochs):
            t12mid = default_timer()
            test_l2_step = 0
            test_l2_full = 0
            test_l2_step_MaxNormRel = 0
            test_l2_full_MaxNormRel = 0
            with torch.no_grad():
                model = torch.compile(model, backend="cudagraphs")
                count = -1
                for i_testloader,(xx, yy) in enumerate(test_loader):
                    loss = 0
                    loss_MaxNormRel = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    count= count +1 
                    for t in range(0, T_out *sum_vector_u_elements_i_iter  , T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter ):
                        y = yy[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter) ]
                        torch.compiler.cudagraph_mark_step_begin()
                        im = model(xx)
                        loss += myloss(im, y)                
                        loss_MaxNormRel += myloss_MaxNormRel (im, y)                
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        if step*sum_vector_u_elements_i_iter >1:
                            xx_tmp = xx[...,:step*sum_vector_u_elements_i_iter]
                        xx = torch.cat((xx[..., step*sum_vector_u_elements_i_iter:], im), dim=-1)
                        if step*sum_vector_u_elements_i_iter >1:
                            xx = torch.cat((xx, xx_tmp), dim=-1)
                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred, yy).item()
                    test_l2_step_MaxNormRel += loss_MaxNormRel.item()
                    test_l2_full_MaxNormRel += myloss_MaxNormRel(pred, yy).item()
            t2 = default_timer()
            print('ep=', ep, ', t2 - t12mid (testTime)=',t2 - t12mid, 
                ', test_l2_step / ntest / (T_out / step)=',test_l2_step / ntest / (T_out / step),
                ', test_l2_full / ntest=',test_l2_full / ntest, 
                ", count_params(model)=",count_params_model,
                ', t2 - t12mid (testTime)=',t2 - t12mid )
            file6 = open(strn_epochs_dump_path_file6, "a")  
            str_file6= ( str( ep) +','+ str(t2 - t12mid)   
                +','+ str(test_l2_step / ntest / (T_out / step))
                +','+str(test_l2_full / ntest)
                +','+str(count_params_model)  
                +','+ str(t2 - t12mid)  
                +'\n' )
            file6.write(str_file6)
            file6.close()