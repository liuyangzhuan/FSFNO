
# FSFNO Copyright (c) 2024, The Regents of the University of California,
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
import os
# from fsfno.fsfno_2d_nimrod import FNO2d_global
# from fsfno.fsfno_2d_GTC import FNO2d_GTCglobal
from fsfno.utilities3 import *
from fsfno.RegressionModel_layer import RegressionModel
from fsfno.fsfno_2d import FNO2d_global

def initializationTrainTestParametersFile( 
        sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
        data_read_global,
        data_read_global_eachTimeStep_std_logRMS,
        train_a_range,train_u_range,test_a_range,test_u_range,
        ntrain,ntest,
        S,S_r,S_theta , T_in,T_out, T_in_steadystate,
        epochs,
        epochs_ofWeigthModification,
        epochs_ofWeigthModificationFactor,                        
        T_out_sub_time_consecutiveIterator_factor,
        IncludeSteadyState, 
        n_beg, startofpatternlist_i_file_no_in_SelectData,
        model_Nimrod_FNO2d_global  ,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j ,
        ii_sub_fieldlist_parm_eq_vector_train_global_lst_i,
        modes, width,
        batch_size,number_of_layers, learning_rate,iterations,
        input_parameter_order, 
        mWidth_input_parameters, 
        nWidth_output_parameters,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        strn_epochs_dump_path_file5,
        strn_epochs_dump_path_file4,
        strn_epochs_dump_path_file3,
        strn_epochs_dump_path_file2,
        strn_epochs_dump_path_file1,
        train_loader ,test_loader,
        model,optimizer,scheduler,
        count_params_model,model_save_path,
        model_2_logRMS_RegressionModel,
        model_2_logRMS_RegressionModel_save_path,
        train_loader_2_logRMS_RegressionModel,
        test_loader_2_logRMS_RegressionModel,
        optimizer_2_logRMS_RegressionModel, scheduler_2_logRMS_RegressionModel,
        count_params_model_2_logRMS_RegressionModel,
        strn_epochs_dump_path_file22,
        best_loss_test_l2_step,
        best_model_state_dict_test_l2_step,
        best_model_save_path_test_l2_step
        ) :
    
    sum_vector_a_elements_i_iter =  len(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii[j_fieldlist_parm_eq_vector_train_global_lst_i][0])
    train_a_global = torch.zeros(ntrain,sum_vector_a_elements_i_iter,S_r ,S_theta,T_in)
    test_a_global  = torch.zeros(ntest, sum_vector_a_elements_i_iter,S_r ,S_theta,T_in)
    train_a_global_2_logRMS_RegressionModel = torch.zeros(ntrain,sum_vector_a_elements_i_iter,T_in)
    test_a_global_2_logRMS_RegressionModel  = torch.zeros(ntest, sum_vector_a_elements_i_iter,T_in)
    if IncludeSteadyState:
        train_a_global_steadystate = torch.zeros(ntrain,sum_vector_a_elements_i_iter,S_r ,S_theta,T_in_steadystate )
        test_a_global_steadystate  = torch.zeros(ntest, sum_vector_a_elements_i_iter,S_r ,S_theta,T_in_steadystate)
    sum_vector_u_elements_i_iter = len(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii[j_fieldlist_parm_eq_vector_train_global_lst_i][1])
    train_u_global = torch.zeros(ntrain,sum_vector_u_elements_i_iter,S_r,S_theta ,T_out)
    test_u_global  = torch.zeros(ntest, sum_vector_u_elements_i_iter,S_r,S_theta,T_out)
    train_u_global_2_logRMS_RegressionModel = torch.zeros(ntrain,sum_vector_u_elements_i_iter,T_out)
    test_u_global_2_logRMS_RegressionModel  = torch.zeros(ntest, sum_vector_u_elements_i_iter,T_out)
    data_read_global_i = range(data_read_global.shape[0])
    for i_ntrain in range(0,ntrain):
        for intermediate_i_file_no in train_a_range:
            item_of_sum_vector_train_a_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
                        item_of_sum_vector_train_a_elements_i += 1                    
                        if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'sfluidne':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureiperp':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureipara':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'flowi':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'apara_2d':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'phi_2d':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 5
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'densityi':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 6
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'x':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 7
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'y':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 8
                        else:
                            exit(1)
                        train_a_global[i_ntrain,item_of_sum_vector_train_a_elements_i,:,:,(intermediate_i_file_no-(n_beg))] = data_read_global[ startofpatternlist_i_file_no_in_SelectData[(i_ntrain)]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
                        train_a_global_2_logRMS_RegressionModel[i_ntrain,item_of_sum_vector_train_a_elements_i,(intermediate_i_file_no-(n_beg))] = data_read_global_eachTimeStep_std_logRMS[ startofpatternlist_i_file_no_in_SelectData[(i_ntrain)]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]]
                        if IncludeSteadyState:
                            for T_in_steadystate_iteration in range(T_in_steadystate):
                                train_a_global_steadystate [i_ntrain,item_of_sum_vector_train_a_elements_i,:,:,T_in_steadystate_iteration]                = data_read_global[ startofpatternlist_i_file_no_in_SelectData[(i_ntrain) ]+T_in_steadystate_iteration,fieldlist_parm_eq_vector_train_global_lst_i_j_0,1,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
    if torch.any(torch.isnan(train_a_global[:,:,:,:])): #data_dump[:,:,:,:])):
        nan_mask = torch.isnan(train_a_global[:,:,:,:])
        print("3292 NaN mask:", nan_mask)
        nan_indices = torch.nonzero(nan_mask, as_tuple=True)
        print("3298 Indices of NaN values:", nan_indices)
        exit(1)
    for i_ntrain in range(0,ntrain):
        for intermediate_i_file_no in train_u_range:
            item_of_sum_vector_train_u_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                    item_of_sum_vector_train_u_elements_i += 1                    
                    if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'sfluidne':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureiperp':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureipara':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'flowi':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'apara_2d':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'phi_2d':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 5
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'densityi':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 6
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'x':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 7
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'y':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 8
                    else:
                        exit(1)
                    train_u_global [i_ntrain,item_of_sum_vector_train_u_elements_i,:,:,(intermediate_i_file_no-(n_beg+(T_in*1)))//1] = data_read_global[startofpatternlist_i_file_no_in_SelectData[(i_ntrain)]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
                    train_u_global_2_logRMS_RegressionModel [i_ntrain,item_of_sum_vector_train_u_elements_i,(intermediate_i_file_no-(n_beg+(T_in*1)))//1] =  data_read_global_eachTimeStep_std_logRMS[startofpatternlist_i_file_no_in_SelectData[(i_ntrain)]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]]
    if torch.any(torch.isnan(train_u_global[:,:,:,:])): #data_dump[:,:,:,:])):
        exit(1)
    for i_ntest in range(0,ntest):
        for intermediate_i_file_no in test_a_range:    
            item_of_sum_vector_test_a_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                    item_of_sum_vector_test_a_elements_i += 1                    
                    if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'sfluidne':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureiperp':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureipara':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'flowi':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'apara_2d':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'phi_2d':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 5
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'densityi':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 6
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'x':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 7
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'y':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 8
                    else:
                        exit(1)
                    test_a_global [i_ntest ,item_of_sum_vector_test_a_elements_i,:,:,(intermediate_i_file_no)//1] = data_read_global[ startofpatternlist_i_file_no_in_SelectData[i_ntest + ntrain]     +intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
                    test_a_global_2_logRMS_RegressionModel [i_ntest ,item_of_sum_vector_test_a_elements_i,(intermediate_i_file_no)//1] = data_read_global_eachTimeStep_std_logRMS[ startofpatternlist_i_file_no_in_SelectData[i_ntest + ntrain]     +intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]]
                    if IncludeSteadyState:
                        for T_in_steadystate_iteration in range(T_in_steadystate):
                            test_a_global_steadystate [ i_ntest,item_of_sum_vector_test_a_elements_i,:,:,T_in_steadystate_iteration] = data_read_global[ startofpatternlist_i_file_no_in_SelectData[i_ntest + ntrain]+T_in_steadystate_iteration,fieldlist_parm_eq_vector_train_global_lst_i_j_0,1,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
    if torch.any(torch.isnan(test_a_global[:,:,:,:])): #data_dump[:,:,:,:])):
        exit(1)
    for i_ntest in range(0,(ntest)):
        for intermediate_i_file_no in test_u_range:
            item_of_sum_vector_test_u_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                    item_of_sum_vector_test_u_elements_i += 1                    
                    if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'sfluidne':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureiperp':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'pressureipara':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'flowi':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'apara_2d':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'phi_2d':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 5
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'densityi':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 6
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'x':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 7
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'y':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 8
                    else:
                        exit(1)
                    test_u_global[i_ntest,item_of_sum_vector_test_u_elements_i,:,:,(intermediate_i_file_no-(n_beg+(T_in*1)))//1] = data_read_global[startofpatternlist_i_file_no_in_SelectData[(i_ntest)+ntrain]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
                    test_u_global_2_logRMS_RegressionModel [i_ntest,item_of_sum_vector_test_u_elements_i,(intermediate_i_file_no-(n_beg+(T_in*1)))//1] = data_read_global_eachTimeStep_std_logRMS[startofpatternlist_i_file_no_in_SelectData[(i_ntest)+ntrain]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]]
    if torch.any(torch.isnan(train_u_global[:,:,:,:])): #data_dump[:,:,:,:])):
        exit(1)
    if IncludeSteadyState:
        train_a = torch.zeros(ntrain,S_r,S_theta,(T_in+T_in_steadystate ) *( sum_vector_a_elements_i_iter ) )
        train_u = torch.zeros(ntrain,S_r,S_theta,T_out   *( sum_vector_u_elements_i_iter ) )
        test_a  = torch.zeros(ntest ,S_r,S_theta,(T_in+T_in_steadystate) *( sum_vector_a_elements_i_iter ) )
        test_u  = torch.zeros(ntest ,S_r,S_theta,T_out   *( sum_vector_u_elements_i_iter ) )
    else:
        train_a = torch.zeros(ntrain,S_r,S_theta,T_in*( sum_vector_a_elements_i_iter ) )
        train_u = torch.zeros(ntrain,S_r,S_theta,T_out   *( sum_vector_u_elements_i_iter ) )
        test_a  = torch.zeros(ntest ,S_r,S_theta,T_in*( sum_vector_a_elements_i_iter ) )
        test_u  = torch.zeros(ntest ,S_r,S_theta,T_out   *( sum_vector_u_elements_i_iter ) )
    if IncludeSteadyState:
        train_a_2_logRMS_RegressionModel = torch.zeros(ntrain,(T_in+T_in_steadystate ) *( sum_vector_a_elements_i_iter ) )
        train_u_2_logRMS_RegressionModel = torch.zeros(ntrain,T_out   *( sum_vector_u_elements_i_iter ) )
        test_a_2_logRMS_RegressionModel  = torch.zeros(ntest ,(T_in+T_in_steadystate) *( sum_vector_a_elements_i_iter ) )
        test_u_2_logRMS_RegressionModel  = torch.zeros(ntest ,T_out   *( sum_vector_u_elements_i_iter ) )
    else:
        train_a_2_logRMS_RegressionModel = torch.zeros(ntrain,T_in*( sum_vector_a_elements_i_iter ) )
        train_u_2_logRMS_RegressionModel = torch.zeros(ntrain,T_out   *( sum_vector_u_elements_i_iter ) )
        test_a_2_logRMS_RegressionModel  = torch.zeros(ntest ,T_in*( sum_vector_a_elements_i_iter ) )
        test_u_2_logRMS_RegressionModel  = torch.zeros(ntest ,T_out   *( sum_vector_u_elements_i_iter ) )
    for item_of_sum_vector_a_elements_i_iter in range(sum_vector_a_elements_i_iter): 
        train_a[:,:,:,(T_in*item_of_sum_vector_a_elements_i_iter):(T_in*(item_of_sum_vector_a_elements_i_iter+1))] = train_a_global[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
        train_a_2_logRMS_RegressionModel[:,(T_in*item_of_sum_vector_a_elements_i_iter):(T_in*(item_of_sum_vector_a_elements_i_iter+1))] = train_a_global_2_logRMS_RegressionModel[:,item_of_sum_vector_a_elements_i_iter,:]
        test_a [:,:,:,(T_in*item_of_sum_vector_a_elements_i_iter):(T_in*(item_of_sum_vector_a_elements_i_iter+1))] = test_a_global [:,item_of_sum_vector_a_elements_i_iter,:,:,:]
        test_a_2_logRMS_RegressionModel [:,(T_in*item_of_sum_vector_a_elements_i_iter):(T_in*(item_of_sum_vector_a_elements_i_iter+1))] = test_a_global_2_logRMS_RegressionModel [:,item_of_sum_vector_a_elements_i_iter,:]
    if IncludeSteadyState:
        for item_of_sum_vector_a_elements_i_iter in range(sum_vector_a_elements_i_iter): 
            train_a[:,:,:,((T_in+T_in_steadystate )*item_of_sum_vector_a_elements_i_iter):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0)] = train_a_global[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
            train_a[:,:,:,(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in+T_in_steadystate-0)] = train_a_global_steadystate[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
            test_a[:,:,:,((T_in+T_in_steadystate)*item_of_sum_vector_a_elements_i_iter):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0)] = test_a_global [:,item_of_sum_vector_a_elements_i_iter,:,:,:]
            test_a [:,:,:,(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in+T_in_steadystate-0)] = test_a_global_steadystate[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
    for item_of_sum_vector_u_elements_i_iter in range(sum_vector_u_elements_i_iter): 
        train_u[:,:,:,(T_out   *item_of_sum_vector_u_elements_i_iter):(T_out   *(item_of_sum_vector_u_elements_i_iter+1))] = train_u_global[:,item_of_sum_vector_u_elements_i_iter,:,:,:]
        train_u_2_logRMS_RegressionModel[:,(T_out   *item_of_sum_vector_u_elements_i_iter):(T_out   *(item_of_sum_vector_u_elements_i_iter+1))] = train_u_global_2_logRMS_RegressionModel[:,item_of_sum_vector_u_elements_i_iter,:]
        test_u [:,:,:,(T_out   *item_of_sum_vector_u_elements_i_iter):(T_out   *(item_of_sum_vector_u_elements_i_iter+1))] = test_u_global [:,item_of_sum_vector_u_elements_i_iter,:,:,:]
        test_u_2_logRMS_RegressionModel [:,(T_out   *item_of_sum_vector_u_elements_i_iter):(T_out   *(item_of_sum_vector_u_elements_i_iter+1))] = test_u_global_2_logRMS_RegressionModel [:,item_of_sum_vector_u_elements_i_iter,:] # .unsqueeze(dim=-1)
    assert (S_r == train_u.shape[-3])
    assert (S_theta == train_u.shape[-2])
    if IncludeSteadyState:
        assert ((T_in+T_in_steadystate ) * sum_vector_a_elements_i_iter == train_a.shape[-1])
        assert ((T_in+T_in_steadystate) * sum_vector_a_elements_i_iter == test_a.shape[-1] )
    else:
        assert (T_in * sum_vector_a_elements_i_iter == train_a.shape[-1])
        assert (T_in * sum_vector_a_elements_i_iter == test_a.shape[-1])
    assert (T_out * sum_vector_u_elements_i_iter == train_u.shape[-1])
    assert (S_theta == train_a.shape[-2])
    assert (S_r == train_a.shape[-3])
    assert (S_r == test_u.shape[-3])
    assert (S_theta == test_u.shape[-2])
    assert (T_out * sum_vector_u_elements_i_iter == test_u.shape[-1])
    assert (T_out * sum_vector_u_elements_i_iter == test_u_2_logRMS_RegressionModel.shape[-1])
    assert (T_out * sum_vector_u_elements_i_iter == train_u_2_logRMS_RegressionModel.shape[-1])
    assert (S_theta == test_a.shape[-2])
    assert (S_r == test_a.shape[-3])                
    if IncludeSteadyState:
        train_a = train_a.reshape(ntrain,S_r,S_theta,(T_in+T_in_steadystate)*sum_vector_a_elements_i_iter)
        test_a  =  test_a.reshape( ntest,S_r,S_theta,(T_in+T_in_steadystate)*sum_vector_a_elements_i_iter)
    else:
        train_a = train_a.reshape(ntrain,S_r,S_theta,T_in*sum_vector_a_elements_i_iter)
        test_a  =  test_a.reshape( ntest,S_r,S_theta,T_in*sum_vector_a_elements_i_iter)
        train_a_2_logRMS_RegressionModel = train_a_2_logRMS_RegressionModel.reshape(ntrain,T_in*sum_vector_a_elements_i_iter)
        test_a_2_logRMS_RegressionModel  =  test_a_2_logRMS_RegressionModel.reshape( ntest,T_in*sum_vector_a_elements_i_iter)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=False )
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset( test_a,  test_u), batch_size=batch_size, shuffle=False)
    train_loader_2_logRMS_RegressionModel = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a_2_logRMS_RegressionModel , train_u_2_logRMS_RegressionModel ), batch_size=batch_size, shuffle=False )
    test_loader_2_logRMS_RegressionModel  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset( test_a_2_logRMS_RegressionModel ,  test_u_2_logRMS_RegressionModel ), batch_size=batch_size, shuffle=False )
    print('Done initializing the training and testing data')
    if IncludeSteadyState:
        if model_Nimrod_FNO2d_global:
            model = FNO2d_global            (modes, modes, width,(T_in+T_in_steadystate),sum_vector_a_elements_i_iter,T_out_sub_time_consecutiveIterator_factor,sum_vector_u_elements_i_iter,number_of_layers).cuda()
        else:
            model = FNO2d_glob_orig         (modes, modes, width,(T_in+T_in_steadystate),sum_vector_a_elements_i_iter).cuda()
    else:
        if model_Nimrod_FNO2d_global:
            total_vector_a_elements_i = 7
            total_vector_u_elements_i = 7
            model = FNO2d_global  (modes, modes, width,
                                        T_in,total_vector_a_elements_i,
                                        T_out_sub_time_consecutiveIterator_factor,
                                        total_vector_u_elements_i,number_of_layers,
                                        input_parameter_order, 
                                        mWidth_input_parameters, 
                                        nWidth_output_parameters).cuda()
            # model = FNO2d_GTCglobal   (modes, modes, width,
            #                             T_in,total_vector_a_elements_i,
            #                             T_out_sub_time_consecutiveIterator_factor,
            #                             total_vector_u_elements_i,number_of_layers,
            #                             input_parameter_order, 
            #                             mWidth_input_parameters, 
            #                             nWidth_output_parameters).cuda()
        else:
            model = FNO2d_glob_orig         (modes, modes, width,(T_in)  ,sum_vector_a_elements_i_iter,T_out_sub_time_consecutiveIterator_factor,sum_vector_u_elements_i_iter,number_of_layers).cuda()
    count_params_model=count_params(model)
    model_2_logRMS_RegressionModel =  RegressionModel(T_in,total_vector_a_elements_i,T_out_sub_time_consecutiveIterator_factor,total_vector_u_elements_i,number_of_layers)
    print("count_params(RegressionModel)=",count_params(model_2_logRMS_RegressionModel))
    print("count_params(FNO2d_glob_orig)=",count_params(model))
    count_params_model_2_logRMS_RegressionModel=count_params(model_2_logRMS_RegressionModel)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations*epochs_ofWeigthModification/epochs)
    optimizer_2_logRMS_RegressionModel = torch.optim.Adam(model_2_logRMS_RegressionModel.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_2_logRMS_RegressionModel = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2_logRMS_RegressionModel , T_max=iterations*epochs_ofWeigthModification/epochs)
    model_save_path_filename= 'i_fieldlist_parm_eq_vector_train_global_lst'+str(i_fieldlist_parm_eq_vector_train_global_lst
                            ) + '__ii_sub_fieldlist_parm_eq_vector_train_global_lst_i' + str(ii_sub_fieldlist_parm_eq_vector_train_global_lst_i
                            )+'__j_fieldlist_parm_eq_vector_train_global_lst_i'+str(j_fieldlist_parm_eq_vector_train_global_lst_i
                            )+ '__state_dict_model.pt'
    if not os.path.exists('./model_parameter/'):
        os.makedirs('./model_parameter/')
    model_save_path = "./model_parameter/"
    model_save_path = model_save_path + model_save_path_filename
    best_model_state_dict_test_l2_step = None
    best_loss_test_l2_step = float('inf')  # Initialize with a high value
    best_model_save_path_test_l2_step = "./model_parameter/" + 'best_' +model_save_path_filename
    model_2_logRMS_RegressionModel_save_path = "./model_parameter/" + 'logRMS_' +model_save_path_filename
    print('Initialized the models, optimizers and schedulers')    
    if not os.path.exists('./plots_Colrs/'):
        os.makedirs('./plots_Colrs/')
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    if not os.path.exists('./plots_pmesh/'):
        os.makedirs('./plots_pmesh/')
    if not os.path.exists('./plots_pmesh_RdBu/'):
        os.makedirs('./plots_pmesh_RdBu/')
    strn_epochs_dump_path_file1 = 'data/'+'epochs_output_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file1 = strn_epochs_dump_path_file1+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file1 =strn_epochs_dump_path_file1 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file1 = strn_epochs_dump_path_file1+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file1 =strn_epochs_dump_path_file1 + ".txt"
    strn_epochs_dump_path_file2 = 'data/'+'epochs_diags_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file2 = strn_epochs_dump_path_file2+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file2 =strn_epochs_dump_path_file2 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file2 = strn_epochs_dump_path_file2+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file2 =strn_epochs_dump_path_file2 + ".txt"
    file2 = open(strn_epochs_dump_path_file2, "w") 
    str_file2= ( 'ep' + ', t2 - t1' +  
            ', train_l2_step / ntrain / (T_out / step)'+
            ', train_l2_full / ntrain' +
            ', test_l2_step / ntest / (T_out / step)' +
            ', test_l2_full / ntest' + 
            ', count_params(model)='+str(count_params(model))+
            ', t12mid - t1'+
            ', t2 - t12mid'+'\n' ) 
    file2.write(str_file2)
    file2.close()
    strn_epochs_dump_path_file3 = 'data/'+'epochs_diagsMaxNormRel_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file3 = strn_epochs_dump_path_file3+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file3 =strn_epochs_dump_path_file3 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file3 = strn_epochs_dump_path_file3+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file3 =strn_epochs_dump_path_file3 + ".txt"
    file3 = open(strn_epochs_dump_path_file3, "w") 
    str_file3= ( 'ep' + ', t2 - t1' +  
            ', train_l1MaxNormRel_step / ntrain / (T_out / step)'+
            ', train_l1MaxNormRel_full / ntrain' +
            ', test_l1MaxNormRel_step / ntest / (T_out / step)' +
            ', test_l1MaxNormRel_full / ntest' + 
            ', t12mid - t1'+
            ', t2 - t12mid'+
            '\n' ) 
    file3.write(str_file3)
    file3.close()
    strn_epochs_dump_path_file4 = 'data/'+'epochs_diagsMaxNormRelList_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file4 = strn_epochs_dump_path_file4+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file4 =strn_epochs_dump_path_file4 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file4 = strn_epochs_dump_path_file4+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file4 =strn_epochs_dump_path_file4 + ".txt"
    file4 = open(strn_epochs_dump_path_file4, "w") 
    str_file4= ( 'Sample' + ', Time-step, ' +  
            ', test_l1MaxNormRelList_step / ntest / (T_out / step)'+
            ', T_out'  +
            ', test_l1MaxNormRelList_full / ntest' +
            ', T__Of__T_out'  +
            ', test_Substep_l1MaxNormRelList_full / ntest' +
            ', T_substep__Of__T_out_sub_time_consecutiveIterator_factor' +
            ', ntst' + 
            '\n' ) 
    file4.write(str_file4)
    file4.close()
    strn_epochs_dump_path_file5 = 'data/'+'epochs_diagsTime_ntestZerosSize1only_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file5 = strn_epochs_dump_path_file5+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file5 =strn_epochs_dump_path_file5 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file5 = strn_epochs_dump_path_file5+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file5 =strn_epochs_dump_path_file5 + ".txt"
    file5 = open(strn_epochs_dump_path_file5, "w") 
    str_file5= ( 'SubTimeStep' + ', sub-step time , ' +  
            ',total time (till this sub-step) '+
            '\n' ) 
    file5.write(str_file5)
    file5.close()
    strn_epochs_dump_path_file22 = 'data/'+'epochs_diagslogRMS_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file22 = strn_epochs_dump_path_file22+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file22 =strn_epochs_dump_path_file22 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file22 = strn_epochs_dump_path_file22+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file22 =strn_epochs_dump_path_file22 + ".txt"
    file22 = open(strn_epochs_dump_path_file22, "w") 
    str_file22= ( 'ep' + ', t2 - t1' +  
            ', train_l2_step / ntrain / (T_out / step)'+
            ', train_l2_full / ntrain' +
            ', test_l2_step / ntest / (T_out / step)' +
            ', test_l2_full / ntest' + 
            ', count_params(model)='+str(count_params(model))+
            ', t12mid - t1'+
            ', t2 - t12mid'+'\n' ) 
    file22.write(str_file2)
    file22.close()
    print('Created the output dump files at ./data/')
    return ( 
        sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
        data_read_global,
        data_read_global_eachTimeStep_std_logRMS,
        train_a_range,train_u_range,test_a_range,test_u_range,
        ntrain,ntest,
        S,S_r,S_theta , T_in,T_out, T_in_steadystate,
        epochs,
        epochs_ofWeigthModification,
        epochs_ofWeigthModificationFactor,                        
        T_out_sub_time_consecutiveIterator_factor,
        IncludeSteadyState, 
        n_beg, startofpatternlist_i_file_no_in_SelectData,
        model_Nimrod_FNO2d_global  ,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j ,
        ii_sub_fieldlist_parm_eq_vector_train_global_lst_i,
        modes, width,
        batch_size,number_of_layers, learning_rate,iterations,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        strn_epochs_dump_path_file5,
        strn_epochs_dump_path_file4,
        strn_epochs_dump_path_file3,
        strn_epochs_dump_path_file2,
        strn_epochs_dump_path_file1,
        train_loader, test_loader,
        model, optimizer, scheduler,
        count_params_model,model_save_path,
        model_2_logRMS_RegressionModel,
        model_2_logRMS_RegressionModel_save_path,
        train_loader_2_logRMS_RegressionModel,
        test_loader_2_logRMS_RegressionModel,
        optimizer_2_logRMS_RegressionModel, scheduler_2_logRMS_RegressionModel,
        count_params_model_2_logRMS_RegressionModel,        
        strn_epochs_dump_path_file22,
        best_loss_test_l2_step,
        best_model_state_dict_test_l2_step,
        best_model_save_path_test_l2_step   
        )