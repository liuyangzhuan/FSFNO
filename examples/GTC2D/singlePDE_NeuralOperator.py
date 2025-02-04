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

def singlePDENeuralOperator(data_read_global,
        data_read_global_eachTimeStep_std,
        ntrain,ntest,
        S_r,S_theta , T_in,T_out, T_in_steadystate,
        if_IncludeSteadyState, 
        startofpatternlist_i_file_no_in_SelectData,
        if_model_Nimrod_STFNO_global,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        epochs,
        epochs_ofWeigthModificationFactor,
        strn_epochs_dump_path_file5,
        T_out_sub_time_consecutiveIterator_factor, step,
        batch_size,
        i_file_no_in_SelectData, 
        strn_epochs_dump_path_file4,
        strn_epochs_dump_path_file3,
        strn_epochs_dump_path_file2,
        strn_epochs_dump_path_file1,
        if_model_parameters_load,
        if_load_best_model_parameters_test_l2_step__if_model_parameters_load_True,
        model,
        train_loader,test_loader,
        optimizer,scheduler,
        count_params_model,
        if_intermediate_parameter_update,
        model_save_path,
        model_2_logRMS_RegressionModel,
        model_2_logRMS_RegressionModel_save_path,
        train_loader_2_logRMS_RegressionModel,
        test_loader_2_logRMS_RegressionModel,
        optimizer_2_logRMS_RegressionModel, scheduler_2_logRMS_RegressionModel,
        count_params_model_2_logRMS_RegressionModel,
        strn_epochs_dump_path_file22,
        manual_seed_value_set,
        number_of_layers,
        best_loss_test_l2_step,
        best_model_state_dict_test_l2_step,
        best_model_save_path_test_l2_step,
        if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
        OneByPowerTransformationFactorOfData,
        log_param,
        nlvls,
        epsilon_inPlottingErrorNormalization
        ):
        
    if not if_model_parameters_load:
        print(" Training and testing the log(RMS(fieldData)) magnitude")
        myloss_criterion_2_logRMS_RegressionModelMSELoss = LpLoss_fieldElements_2_logRMS_RegressionModel(size_average=False)
        for ep in range(epochs):
            model_2_logRMS_RegressionModel.train()
            t1_2_logRMS_RegressionModel = default_timer()
            train_l2_step_2_logRMS_RegressionModel = 0
            train_l2_full_2_logRMS_RegressionModel = 0
            for xx_2_logRMS_RegressionModel, yy_2_logRMS_RegressionModel in train_loader_2_logRMS_RegressionModel:
                loss_2_logRMS_RegressionModelMSELoss = 0
                xx_2_logRMS_RegressionModel = xx_2_logRMS_RegressionModel.to(device)
                yy_2_logRMS_RegressionModel = yy_2_logRMS_RegressionModel.to(device)
                for t in range(0, T_out*sum_vector_u_elements_i_iter , T_out_sub_time_consecutiveIterator_factor*sum_vector_u_elements_i_iter ):                                    
                    y_2_logRMS_RegressionModel = yy_2_logRMS_RegressionModel[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter) ]
                    im_2_logRMS_RegressionModel = model_2_logRMS_RegressionModel(xx_2_logRMS_RegressionModel.to(device))
                    loss_2_logRMS_RegressionModelMSELoss += myloss_criterion_2_logRMS_RegressionModelMSELoss(im_2_logRMS_RegressionModel, y_2_logRMS_RegressionModel)  # Squeeze to match dimensions
                    if t == 0:
                        pred_2_logRMS_RegressionModel = im_2_logRMS_RegressionModel
                    else:
                        pred_2_logRMS_RegressionModel = torch.cat((pred_2_logRMS_RegressionModel, im_2_logRMS_RegressionModel), -1)
                    xx_2_logRMS_RegressionModel = torch.cat((xx_2_logRMS_RegressionModel[..., step:], im_2_logRMS_RegressionModel), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_tmp_2_logRMS_RegressionModel = xx_2_logRMS_RegressionModel[...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                    xx_2_logRMS_RegressionModel = torch.cat((xx_2_logRMS_RegressionModel[...,T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter:], im_2_logRMS_RegressionModel), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_2_logRMS_RegressionModel = torch.cat((xx_2_logRMS_RegressionModel, xx_tmp_2_logRMS_RegressionModel[:]), dim=-1)
                        exit(1)
                train_l2_step_2_logRMS_RegressionModel += loss_2_logRMS_RegressionModelMSELoss.item()
                l2_full_2_logRMS_RegressionModel = myloss_criterion_2_logRMS_RegressionModelMSELoss(pred_2_logRMS_RegressionModel, yy_2_logRMS_RegressionModel)
                train_l2_full_2_logRMS_RegressionModel += l2_full_2_logRMS_RegressionModel.item()
                optimizer_2_logRMS_RegressionModel.zero_grad()
                loss_2_logRMS_RegressionModelMSELoss.backward()
                optimizer_2_logRMS_RegressionModel.step()
                scheduler_2_logRMS_RegressionModel.step()
            t12_mid_2_logRMS_RegressionModel = default_timer()
            test_l2_step_2_logRMS_RegressionModel = 0
            test_l2_full_2_logRMS_RegressionModel = 0
            with torch.no_grad():
                for xx_2_logRMS_RegressionModel, yy_2_logRMS_RegressionModel in test_loader_2_logRMS_RegressionModel:
                    loss_2_logRMS_RegressionModel = 0
                    xx_2_logRMS_RegressionModel = xx_2_logRMS_RegressionModel.to(device)
                    yy_2_logRMS_RegressionModel = yy_2_logRMS_RegressionModel.to(device)
                    for t in range(0, T_out *sum_vector_u_elements_i_iter  , T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter ):
                        y_2_logRMS_RegressionModel = yy_2_logRMS_RegressionModel[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter )]
                        im_2_logRMS_RegressionModel = model_2_logRMS_RegressionModel(xx_2_logRMS_RegressionModel)
                        loss_2_logRMS_RegressionModel += myloss_criterion_2_logRMS_RegressionModelMSELoss(im_2_logRMS_RegressionModel, y_2_logRMS_RegressionModel)
                        if t == 0:
                            pred_2_logRMS_RegressionModel = im_2_logRMS_RegressionModel
                        else:
                            pred_2_logRMS_RegressionModel = torch.cat((pred_2_logRMS_RegressionModel, im_2_logRMS_RegressionModel), -1)
                        xx_2_logRMS_RegressionModel = torch.cat((xx_2_logRMS_RegressionModel[..., step:], im_2_logRMS_RegressionModel), dim=-1)
                    test_l2_step_2_logRMS_RegressionModel += loss_2_logRMS_RegressionModel.item()
                    test_l2_full_2_logRMS_RegressionModel += myloss_criterion_2_logRMS_RegressionModelMSELoss(pred_2_logRMS_RegressionModel, yy_2_logRMS_RegressionModel).item()
            t2_2_logRMS_RegressionModel = default_timer()
            print(ep, t2_2_logRMS_RegressionModel - t1_2_logRMS_RegressionModel, train_l2_step_2_logRMS_RegressionModel / ntrain / (T_out / step), train_l2_full_2_logRMS_RegressionModel / ntrain, test_l2_step_2_logRMS_RegressionModel / ntest / (T_out / step), test_l2_full_2_logRMS_RegressionModel / ntest)
            file22 = open(strn_epochs_dump_path_file22, "a")  
            str_file22= ( str( ep) +','+ str(t2_2_logRMS_RegressionModel - t1_2_logRMS_RegressionModel)   
                +','+ str( train_l2_step_2_logRMS_RegressionModel / ntrain / (T_out / step))
                +','+ str( train_l2_full_2_logRMS_RegressionModel / ntrain)
                +','+ str(test_l2_step_2_logRMS_RegressionModel / ntest / (T_out / step))
                +','+str(test_l2_full_2_logRMS_RegressionModel / ntest)
                +','+str(count_params_model_2_logRMS_RegressionModel)  
                +','+ str(t12_mid_2_logRMS_RegressionModel - t1_2_logRMS_RegressionModel)   
                +','+ str(t2_2_logRMS_RegressionModel - t12_mid_2_logRMS_RegressionModel)  
                +'\n' )
            file22.write(str_file22)
            file22.close()
        print(" Finished training and testing the log(RMS(fieldData)) magnitude")
        print(" Training and testing the normalized data")
        myloss = LpLoss_fieldElements(size_average=True)
        myloss_MaxNormRel = LpLoss_L1NormRel_fieldElements(size_average=True)
        for ep in range(epochs):
            if ep % epochs_ofWeigthModificationFactor == 0 : #An optional condition not being utilized now. If used, will randomly modify the weights of the model. 
                if if_IncludeSteadyState:
                    if if_model_Nimrod_STFNO_global:
                        pass
                    else:
                        pass
                else:
                    if if_model_Nimrod_STFNO_global:
                        for model_conv_linears_layer_idx, model_conv_linears_layer in enumerate(model.conv_linears):
                            for model_SpectralConv2d_idx, model_SpectralConv2d in enumerate(model_conv_linears_layer):
                                torch.manual_seed( manual_seed_value_set + (number_of_layers * 100000) + 10000 + ep)
                                model_SpectralConv2d.weights1 = nn.Parameter( model_SpectralConv2d.weights1 * (torch.ones(model_SpectralConv2d.weights1.size(),device=model_SpectralConv2d.weights1.device ) +torch.rand(model_SpectralConv2d.weights1.size(),device=model_SpectralConv2d.weights1.device ) ) )
                                torch.manual_seed( manual_seed_value_set + (number_of_layers * 100000) + 20000 + ep)
                                model_SpectralConv2d.weights2 = nn.Parameter( model_SpectralConv2d.weights2 * (torch.ones(model_SpectralConv2d.weights2.size(),device=model_SpectralConv2d.weights2.device ) +torch.rand(model_SpectralConv2d.weights2.size(),device=model_SpectralConv2d.weights2.device ) ) )                                
                    else:
                        pass
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
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_tmp = xx[...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                    xx = torch.cat((xx[...,T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter:], im), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx = torch.cat((xx, xx_tmp[:]), dim=-1)
                        exit(1)
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
                        y = yy[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter )]
                        im = model(xx)
                        loss += myloss(im, y)                
                        loss_MaxNormRel += myloss_MaxNormRel (im, y)                
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                            xx_tmp = xx[...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                        xx = torch.cat((xx[...,T_out_sub_time_consecutiveIterator_factor * step*sum_vector_u_elements_i_iter:], im), dim=-1)
                        if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                            xx = torch.cat((xx, xx_tmp[:]), dim=-1)
                            exit(1)
                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred, yy).item()
                    test_l2_step_MaxNormRel += loss_MaxNormRel.item()
                    test_l2_full_MaxNormRel += myloss_MaxNormRel(pred, yy).item()
            if test_l2_step < best_loss_test_l2_step:
                best_loss_test_l2_step = test_l2_step
                best_model_state_dict_test_l2_step = model.state_dict()
            t2 = default_timer()
            print('ep=', ep, ', t2 - t1=',t2 - t1,                                 ', train_l2_step / ntrain / (T_out / step)=', train_l2_step / ntrain / (T_out / step),                                 ', train_l2_full / ntrain=', train_l2_full / ntrain,                                ', test_l2_step / ntest / (T_out / step)=',test_l2_step / ntest / (T_out / step),                                ', test_l2_full / ntest=',test_l2_full / ntest,                                 ", count_params(model)=",count_params_model,                                ', t12mid - t1 (trainTime)=',t12mid - t1,                                 ', t2 - t12mid (testTime)=',t2 - t12mid )
            file1 = open(strn_epochs_dump_path_file1, "a")  # append mode
            str_file1= ( 'ep=' +str( ep) + ', t2 - t1(trainTime+testTime)='+str(t2 - t1) +  
                ', train_l2_step / ntrain / (T_out / step)='+str( train_l2_step / ntrain / (T_out / step))+
                ', train_l2_full / ntrain='+str( train_l2_full / ntrain)+
                ', test_l2_step / ntest / (T_out / step)='+str(test_l2_step / ntest / (T_out / step))+
                ', test_l2_full / ntest='+str(test_l2_full / ntest) + 
                ', t12mid - t1(trainTime)='+str(t12mid - t1) +  
                ', t2 - t12mid(testTime)='+str(t2 - t12mid) +  
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
        print(" Finished training and testing the normalized data")
        file1 = open(strn_epochs_dump_path_file1, "a")  # append mode
        str_file1= ( '\n\n\n\n\n\n\n\n' )
        file1.write(str_file1)
        file1.close()
        if if_intermediate_parameter_update:
            pass # Not considering this way of modifying the model
        torch.save(model_2_logRMS_RegressionModel.state_dict(), model_2_logRMS_RegressionModel_save_path)
        torch.save(model.state_dict(), model_save_path)
        torch.save(best_model_state_dict_test_l2_step, best_model_save_path_test_l2_step)
    if if_model_parameters_load:
        if if_load_best_model_parameters_test_l2_step__if_model_parameters_load_True:
            model.load_state_dict(torch.load(best_model_save_path_test_l2_step))
            model.eval()
        else:
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
        model_2_logRMS_RegressionModel.load_state_dict(torch.load(model_2_logRMS_RegressionModel_save_path))
        model_2_logRMS_RegressionModel.eval()
    print('Finished training & testing the models with epochs')
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
        test_loader,
        test_loader_2_logRMS_RegressionModel
        )
    print('Generating the contour plots')
    contourplotting(data_read_global,
        data_read_global_eachTimeStep_std,
        ntrain,
        T_out,
        startofpatternlist_i_file_no_in_SelectData,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        epochs,
        T_out_sub_time_consecutiveIterator_factor, step,
        batch_size,
        i_file_no_in_SelectData, 
        model,
        test_loader,
        test_loader_2_logRMS_RegressionModel,
        model_2_logRMS_RegressionModel,
        if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
        OneByPowerTransformationFactorOfData,
        log_param,
        nlvls,
        epsilon_inPlottingErrorNormalization
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