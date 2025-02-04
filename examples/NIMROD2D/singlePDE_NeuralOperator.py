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

import numpy as np
from stfno.utilities3 import *
from timeit import default_timer
from contour_plotting import contourplotting
from relativeError_eachTestDataSample_file4 import relativeErrorEachTestDataSample_file4
from inferenceTime_testData_file5 import inferenceTimeTestData_file5
from jittorchcompile_inferenceTime_testData_file6 import  jittorchcompile_inferenceTimeTestData_file6

def singlePDENeuralOperator(data_read_global,
        data_read_global_mean,data_read_global_std,
        data_read_global_eachTimeStep_mean,
        data_read_global_eachTimeStep_std,
        ntrain,ntest,
        S,S_r,S_theta,
        r_theta_phi, 
        T_in,T_out, T_in_steadystate,
        if_IncludeSteadyState, 
        startofpatternlist_i_file_no_in_SelectData,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        epochs,
        strn_epochs_dump_path_file6,
        strn_epochs_dump_path_file5,
        T_out_sub_time_consecutiveIterator_factor, step,
        batch_size,
        i_file_no_in_SelectData, 
        strn_epochs_dump_path_file4,
        strn_epochs_dump_path_file3,
        strn_epochs_dump_path_file2,
        strn_epochs_dump_path_file1,
        if_model_parameters_load,
        if_model_jit_torchCompile,
        if_postTraingAndTesting_ContourPlotsOfTestingData,
        if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
        OneByPowerTransformationFactorOfData,
        log_param,
        nlvls,
        epsilon_inPlottingErrorNormalization,
        model,
        train_loader,test_loader,
        optimizer,scheduler,
        count_params_model,
        if_intermediate_parameter_update,
        model_save_path
        ):
    gridx = np.arange(S)
    gridy = np.arange(S)
    xi, yi = np.meshgrid(gridx, gridy)
    if not if_model_parameters_load:
        print(" Training and testing the data")
        myloss = LpLoss_fieldElements(size_average=True)
        myloss_MaxNormRel =LpLoss_MaxNormRel_fieldElements(size_average=True)
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0
            train_l2_full = 0
            train_l2_step_MaxNormRel = 0
            train_l2_full_MaxNormRel = 0
            for xx, yy in train_loader:
                loss = 0
                loss_MaxNormRel = 0
                xx = xx.to(device)
                yy = yy.to(device)
                for t in range(0, T_out*sum_vector_u_elements_i_iter , T_out_sub_time_consecutiveIterator_factor*sum_vector_u_elements_i_iter ):                                    
                    y = yy[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter) ]
                    im = model(xx)
                    loss += myloss(im, y)
                    loss_MaxNormRel += myloss_MaxNormRel(im, y)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    if step*sum_vector_u_elements_i_iter >1:
                        xx_tmp = xx[...,:step*sum_vector_u_elements_i_iter]
                    xx = torch.cat((xx[..., step*sum_vector_u_elements_i_iter:], im), dim=-1)
                    if step*sum_vector_u_elements_i_iter >1:
                        xx = torch.cat((xx, xx_tmp[:]), dim=-1)
                    xx_tmp = xx[...,:step*sum_vector_a_elements_i_iter]
                    xx = torch.cat((xx[..., step*sum_vector_a_elements_i_iter:], im), dim=-1)
                    xx = torch.cat((xx, xx_tmp), dim=-1)
                train_l2_step += loss.item()
                l2_full = myloss(pred, yy)
                train_l2_full += l2_full.item()
                train_l2_step_MaxNormRel += loss_MaxNormRel.item()
                train_l2_full_MaxNormRel += myloss_MaxNormRel(pred, yy).item()
                optimizer.zero_grad()                                
                loss.backward()
                optimizer.step()
                scheduler.step()
            t12mid = default_timer()
            test_l2_step = 0
            test_l2_full = 0
            test_l2_step_MaxNormRel = 0
            test_l2_full_MaxNormRel = 0
            with torch.no_grad():
                count = -1
                for i_testloader,(xx, yy) in enumerate(test_loader):
                    loss = 0
                    loss_MaxNormRel = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    count= count +1 
                    for t in range(0, T_out *sum_vector_u_elements_i_iter  , T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter ):
                        y = yy[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter) ]
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
            print('ep=', ep, ', t2 - t1 (trainTime+testTime)=',t2 - t1, 
                ', train_l2_step / ntrain / (T_out / step)=', train_l2_step / ntrain / (T_out / step), 
                ', train_l2_full / ntrain=', train_l2_full / ntrain,
                ', test_l2_step / ntest / (T_out / step)=',test_l2_step / ntest / (T_out / step),
                ', test_l2_full / ntest=',test_l2_full / ntest, 
                ", count_params(model)=",count_params_model,
                ', t12mid - t1 (trainTime)=',t12mid - t1, 
                ', t2 - t12mid (testTime)=',t2 - t12mid )
            file1 = open(strn_epochs_dump_path_file1, "a")  # append mode
            str_file1= ( 'ep=' +str( ep) + ', t2 - t1 (trainTime+testTime) ='+str(t2 - t1) +  
                ', train_l2_step / ntrain / (T_out / step)='+str( train_l2_step / ntrain / (T_out / step))+
                ', train_l2_full / ntrain='+str( train_l2_full / ntrain)+
                ', test_l2_step / ntest / (T_out / step)='+str(test_l2_step / ntest / (T_out / step))+
                ', test_l2_full / ntest='+str(test_l2_full / ntest) + 
                ', t12mid - t1(train)='+str(t12mid - t1) +  
                ', t2 - t12mid(test)='+str(t2 - t12mid) +  
                '\n' )
            file1.write(str_file1)
            file1.close()
            file2 = open(strn_epochs_dump_path_file2, "a")  
            str_file2= ( str( ep) +','+ str(t2 - t1)   
                +','+ str( train_l2_step / ntrain / (T_out / step))
                +','+ str( train_l2_full / ntrain)
                +','+ str(test_l2_step / ntest / (T_out / step))
                +','+str(test_l2_full / ntest)
                +','+str(count_params_model)  
                +','+ str(t12mid - t1)   
                +','+ str(t2 - t12mid)  
                +'\n' )
            file2.write(str_file2)
            file2.close()
            file3 = open(strn_epochs_dump_path_file3, "a")  
            str_file3= ( str( ep) +','+ str(t2 - t1)   
                +','+ str( train_l2_step_MaxNormRel / ntrain / (T_out / step))
                +','+ str( train_l2_full_MaxNormRel / ntrain)
                +','+ str(test_l2_step_MaxNormRel / ntest / (T_out / step))
                +','+str(test_l2_full_MaxNormRel / ntest)
                +','+str(count_params_model)
                +','+ str(t12mid - t1)   
                +','+ str(t2 - t12mid)  
                + '\n' )
            file3.write(str_file3)
            file3.close()
        file1 = open(strn_epochs_dump_path_file1, "a")  # append mode
        str_file1= ( '\n\n\n\n\n\n\n\n' )
        file1.write(str_file1)
        file1.close()
        if if_intermediate_parameter_update:
            pass  # Not considering this way of modifying the model
        torch.save(model.state_dict(), model_save_path)
    if if_model_parameters_load:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
    print('Finished training & testing the model with epochs')
    print('Calculating relative error norm lists of test sample and writing at ',strn_epochs_dump_path_file4)
    relativeErrorEachTestDataSample_file4(
        ntrain,ntest,
        T_out, 
        startofpatternlist_i_file_no_in_SelectData,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        epochs,
        T_out_sub_time_consecutiveIterator_factor, step,
        batch_size,
        i_file_no_in_SelectData, 
        strn_epochs_dump_path_file4,
        model,
        test_loader
        )
    if if_postTraingAndTesting_ContourPlotsOfTestingData:
        print('Generating the contour plots')
        contourplotting(data_read_global,
            data_read_global_mean,data_read_global_std,
            data_read_global_eachTimeStep_mean,
            data_read_global_eachTimeStep_std,
            ntrain,
            r_theta_phi, 
            T_out,
            startofpatternlist_i_file_no_in_SelectData,
            i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
            sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
            epochs,
            T_out_sub_time_consecutiveIterator_factor, step,
            batch_size,
            i_file_no_in_SelectData, 
            if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
            OneByPowerTransformationFactorOfData,
            log_param,
            nlvls,
            epsilon_inPlottingErrorNormalization,
            model,
            test_loader
            )
    print('Measuring the inference time of test data and writing at ',strn_epochs_dump_path_file5)
    inferenceTimeTestData_file5(
    S_r,S_theta , T_in,T_out, T_in_steadystate,
    if_IncludeSteadyState, 
    sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
    epochs,
    strn_epochs_dump_path_file5,
    T_out_sub_time_consecutiveIterator_factor, step,
    batch_size,
    model
    )

    if if_model_jit_torchCompile:
        print('Measuring JIT torchDOTcompile test time')
        jittorchcompile_inferenceTimeTestData_file6(            
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
            )