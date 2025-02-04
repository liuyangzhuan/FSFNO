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
import os
# from stfno.stfno_2d_nimrod import FNO2d_global
from stfno.utilities3 import *
from stfno.stfno_2d import FNO2d_global
from stfno.fno_2d_baseline import FNO2d_glob_orig
from stfno.stfno_3d import FNO2d_NIMRODglobal_3D
from stfno.fno_3d_baseline import FNO2d_glob_orig_3D
def initializationTrainTestParametersFile( 
        sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
        data_read_global,
        train_a_range,train_u_range,test_a_range,test_u_range,
        ntrain,ntest,
        S,S_r,S_theta , T_in,T_out, T_in_steadystate,
        if_IncludeSteadyState, 
        n_beg, startofpatternlist_i_file_no_in_SelectData,
        if_model_Nimrod_STFNO_global,
        if_model_jit_torchCompile, 
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j ,
        ii_sub_fieldlist_parm_eq_vector_train_global_lst_i,
        modes, width,
        batch_size,number_of_layers, learning_rate,iterations,
        input_parameter_order, 
        mWidth_input_parameters, 
        nWidth_output_parameters,
        if_3D,S_n_phi,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        strn_epochs_dump_path_file6,
        strn_epochs_dump_path_file5,
        strn_epochs_dump_path_file4,
        strn_epochs_dump_path_file3,
        strn_epochs_dump_path_file2,
        strn_epochs_dump_path_file1,
        train_loader ,test_loader,
        model,optimizer,scheduler,
        count_params_model,model_save_path
        ):
    sum_vector_a_elements_i_iter =  len(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii[j_fieldlist_parm_eq_vector_train_global_lst_i][0])
    train_a_global = torch.zeros(ntrain,sum_vector_a_elements_i_iter,S_r,S_theta,S_n_phi,T_in)
    test_a_global  = torch.zeros(ntest, sum_vector_a_elements_i_iter,S_r,S_theta,S_n_phi,T_in)
    if if_IncludeSteadyState:
        train_a_global_steadystate = torch.zeros(ntrain,sum_vector_a_elements_i_iter,S_r,S_theta,S_n_phi,T_in_steadystate )
        test_a_global_steadystate  = torch.zeros(ntest, sum_vector_a_elements_i_iter,S_r,S_theta,S_n_phi,T_in_steadystate)
    sum_vector_u_elements_i_iter = len(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii[j_fieldlist_parm_eq_vector_train_global_lst_i][1])
    train_u_global = torch.zeros(ntrain,sum_vector_u_elements_i_iter,S_r,S_theta,S_n_phi,T_out)
    test_u_global  = torch.zeros(ntest, sum_vector_u_elements_i_iter,S_r,S_theta,S_n_phi,T_out)
    data_read_global_i = range(data_read_global.shape[0])
    for i_ntrain in range(0,ntrain):
        for intermediate_i_file_no in train_a_range:
            item_of_sum_vector_train_a_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
                        item_of_sum_vector_train_a_elements_i += 1                    
                        if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'n':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'p':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'v':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'b':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                        elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'j':
                            fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                        else:
                            exit(1)
                        train_a_global[i_ntrain,item_of_sum_vector_train_a_elements_i,:,:,:,(intermediate_i_file_no-(n_beg))] = data_read_global[ startofpatternlist_i_file_no_in_SelectData[(i_ntrain)]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:,:]
                        if if_IncludeSteadyState:
                            for T_in_steadystate_iteration in range(T_in_steadystate):
                                train_a_global_steadystate [i_ntrain,item_of_sum_vector_train_a_elements_i,:,:,T_in_steadystate_iteration]                = data_read_global[ startofpatternlist_i_file_no_in_SelectData[(i_ntrain) ]+T_in_steadystate_iteration,fieldlist_parm_eq_vector_train_global_lst_i_j_0,1,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
    if torch.any(torch.isnan(train_a_global[:,:,:,:])): #data_dump[:,:,:,:])):
        nan_mask = torch.isnan(train_a_global[:,:,:,:])
        print("initialization NaN mask:", nan_mask)
        nan_indices = torch.nonzero(nan_mask, as_tuple=True)
        print("initialization Indices of NaN values:", nan_indices)
        exit(1)
    for i_ntrain in range(0,ntrain):
        for intermediate_i_file_no in train_u_range:
            item_of_sum_vector_train_u_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                    item_of_sum_vector_train_u_elements_i += 1                    
                    if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'n':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'p':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'v':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'b':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'j':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                    else:
                        exit(1)
                    train_u_global [i_ntrain,item_of_sum_vector_train_u_elements_i,:,:,:,(intermediate_i_file_no-(n_beg+(T_in*1)))//1] = data_read_global[startofpatternlist_i_file_no_in_SelectData[(i_ntrain)]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:,:]
    if torch.any(torch.isnan(train_u_global[:,:,:,:])): #data_dump[:,:,:,:])):
        exit(1)
    for i_ntest in range(0,ntest):
        for intermediate_i_file_no in test_a_range:    
            item_of_sum_vector_test_a_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                    item_of_sum_vector_test_a_elements_i += 1                    
                    if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'n':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'p':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'v':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'b':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'j':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                    else:
                        exit(1)
                    test_a_global [i_ntest ,item_of_sum_vector_test_a_elements_i,:,:,:,(intermediate_i_file_no)//1] = data_read_global[ startofpatternlist_i_file_no_in_SelectData[i_ntest + ntrain]     +intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:,:]
                    if if_IncludeSteadyState:
                        for T_in_steadystate_iteration in range(T_in_steadystate):
                            test_a_global_steadystate [ i_ntest,item_of_sum_vector_test_a_elements_i,:,:,T_in_steadystate_iteration] = data_read_global[ startofpatternlist_i_file_no_in_SelectData[i_ntest + ntrain]+T_in_steadystate_iteration,fieldlist_parm_eq_vector_train_global_lst_i_j_0,1,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
    if torch.any(torch.isnan(test_a_global[:,:,:,:])): #data_dump[:,:,:,:])):
        exit(1)
    for i_ntest in range(0,(ntest)):
        for intermediate_i_file_no in test_u_range:
            item_of_sum_vector_test_u_elements_i = -1
            for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                    item_of_sum_vector_test_u_elements_i += 1                    
                    if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'n':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'p':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'v':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'b':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                    elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'j':
                        fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                    else:
                        exit(1)
                    test_u_global[i_ntest,item_of_sum_vector_test_u_elements_i,:,:,:,(intermediate_i_file_no-(n_beg+(T_in*1)))//1] = data_read_global[startofpatternlist_i_file_no_in_SelectData[(i_ntest)+ntrain]+intermediate_i_file_no,fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:,:]
    if torch.any(torch.isnan(train_u_global[:,:,:,:])): #data_dump[:,:,:,:])):
        exit(1)    
    if if_IncludeSteadyState:
        train_a = torch.zeros(ntrain,S_r,S_theta,S_n_phi,(T_in+T_in_steadystate ) *( sum_vector_a_elements_i_iter ) )
        train_u = torch.zeros(ntrain,S_r,S_theta,S_n_phi,T_out   *( sum_vector_u_elements_i_iter ) )
        test_a  = torch.zeros(ntest ,S_r,S_theta,S_n_phi,(T_in+T_in_steadystate) *( sum_vector_a_elements_i_iter ) )
        test_u  = torch.zeros(ntest ,S_r,S_theta,S_n_phi,T_out   *( sum_vector_u_elements_i_iter ) )
    else:
        train_a = torch.zeros(ntrain,S_r,S_theta,S_n_phi,T_in*( sum_vector_a_elements_i_iter ) )
        train_u = torch.zeros(ntrain,S_r,S_theta,S_n_phi,T_out   *( sum_vector_u_elements_i_iter ) )
        test_a  = torch.zeros(ntest ,S_r,S_theta,S_n_phi,T_in*( sum_vector_a_elements_i_iter ) )
        test_u  = torch.zeros(ntest ,S_r,S_theta,S_n_phi,T_out   *( sum_vector_u_elements_i_iter ) )
    for item_of_sum_vector_a_elements_i_iter in range(sum_vector_a_elements_i_iter): 
        train_a[:,:,:,:,(T_in*item_of_sum_vector_a_elements_i_iter):(T_in*(item_of_sum_vector_a_elements_i_iter+1))] = train_a_global[:,item_of_sum_vector_a_elements_i_iter,:,:,:,:]
        test_a [:,:,:,:,(T_in*item_of_sum_vector_a_elements_i_iter):(T_in*(item_of_sum_vector_a_elements_i_iter+1))] = test_a_global [:,item_of_sum_vector_a_elements_i_iter,:,:,:,:]
    if if_IncludeSteadyState:
        for item_of_sum_vector_a_elements_i_iter in range(sum_vector_a_elements_i_iter): 
            train_a[:,:,:,((T_in+T_in_steadystate )*item_of_sum_vector_a_elements_i_iter):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0)] = train_a_global[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
            train_a[:,:,:,(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in+T_in_steadystate-0)] = train_a_global_steadystate[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
            test_a[:,:,:,((T_in+T_in_steadystate)*item_of_sum_vector_a_elements_i_iter):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0)] = test_a_global [:,item_of_sum_vector_a_elements_i_iter,:,:,:]
            test_a [:,:,:,(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in-0):(((T_in+T_in_steadystate)*(item_of_sum_vector_a_elements_i_iter))+T_in+T_in_steadystate-0)] = test_a_global_steadystate[:,item_of_sum_vector_a_elements_i_iter,:,:,:]
    for item_of_sum_vector_u_elements_i_iter in range(sum_vector_u_elements_i_iter): 
        train_u[:,:,:,:,(T_out   *item_of_sum_vector_u_elements_i_iter):(T_out   *(item_of_sum_vector_u_elements_i_iter+1))] = train_u_global[:,item_of_sum_vector_u_elements_i_iter,:,:,:,:]
        test_u [:,:,:,:,(T_out   *item_of_sum_vector_u_elements_i_iter):(T_out   *(item_of_sum_vector_u_elements_i_iter+1))] = test_u_global [:,item_of_sum_vector_u_elements_i_iter,:,:,:,:]
    assert (S_theta == train_u.shape[-3])
    assert (S_n_phi == train_u.shape[-2])
    if if_IncludeSteadyState:
        assert ((T_in+T_in_steadystate ) * sum_vector_a_elements_i_iter == train_a.shape[-1])
        assert ((T_in+T_in_steadystate) * sum_vector_a_elements_i_iter == test_a.shape[-1] )
    else:
        assert (T_in * sum_vector_a_elements_i_iter == train_a.shape[-1])
        assert (T_in * sum_vector_a_elements_i_iter == test_a.shape[-1])
    assert (T_out * sum_vector_u_elements_i_iter == train_u.shape[-1])
    assert (S_n_phi == train_a.shape[-2])
    assert (S_theta == train_a.shape[-3])
    assert (S_theta == test_u.shape[-3])
    assert (S_n_phi == test_u.shape[-2])
    assert (T_out * sum_vector_u_elements_i_iter == test_u.shape[-1])
    assert (S_n_phi == test_a.shape[-2])
    assert (S_theta == test_a.shape[-3])                
    if if_IncludeSteadyState:
        train_a = train_a.reshape(ntrain,S_r,S_theta,S_n_phi,(T_in+T_in_steadystate)*sum_vector_a_elements_i_iter)
        test_a  =  test_a.reshape( ntest,S_r,S_theta,S_n_phi,(T_in+T_in_steadystate)*sum_vector_a_elements_i_iter)
    else:
        train_a = train_a.reshape(ntrain,S_r,S_theta,S_n_phi,T_in*sum_vector_a_elements_i_iter)
        test_a  =  test_a.reshape( ntest,S_r,S_theta,S_n_phi,T_in*sum_vector_a_elements_i_iter)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=False )
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset( test_a,  test_u), batch_size=batch_size, shuffle=False)
    print('Done initializing the training and testing data')
    if if_model_jit_torchCompile:
        torch._dynamo.reset()

    if if_IncludeSteadyState:
        if if_model_Nimrod_STFNO_global:
            model = FNO2d_global            (modes, modes,modes, width,(T_in+T_in_steadystate),sum_vector_a_elements_i_iter,T_out,sum_vector_u_elements_i_iter,number_of_layers).cuda()
        else:
            model = FNO2d_glob_orig         (modes, modes, width,(T_in+T_in_steadystate),sum_vector_a_elements_i_iter).cuda()
    else:
        if if_model_Nimrod_STFNO_global:
            if if_3D:
                model = FNO2d_NIMRODglobal_3D   (modes, modes,modes, width,(T_in),
                                                sum_vector_a_elements_i_iter,T_out,
                                                sum_vector_u_elements_i_iter,number_of_layers,
                                                input_parameter_order, 
                                                mWidth_input_parameters, 
                                                nWidth_output_parameters,
                                                if_model_jit_torchCompile).cuda()
            else:
                model = FNO2d_global            (modes, modes, width,(T_in),
                                                sum_vector_a_elements_i_iter,T_out,
                                                sum_vector_u_elements_i_iter,number_of_layers,
                                                input_parameter_order, 
                                                mWidth_input_parameters, 
                                                nWidth_output_parameters,
                                                if_model_jit_torchCompile).cuda()
        else:
            if if_3D:
                model = FNO2d_glob_orig_3D   (      modes, modes,modes, width,(T_in),
                                                sum_vector_a_elements_i_iter,
                                                if_model_jit_torchCompile)
            else:
                model = FNO2d_glob_orig         (modes, modes, width,(T_in)  ,sum_vector_a_elements_i_iter).cuda()
    count_params_model=count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    model_save_path_filename= 'i_fieldlist_parm_eq_vector_train_global_lst'+str(i_fieldlist_parm_eq_vector_train_global_lst
                            ) + '__ii_sub_fieldlist_parm_eq_vector_train_global_lst_i' + str(ii_sub_fieldlist_parm_eq_vector_train_global_lst_i
                            )+'__j_fieldlist_parm_eq_vector_train_global_lst_i'+str(j_fieldlist_parm_eq_vector_train_global_lst_i
                            )+ '__state_dict_model.pt'
    if not os.path.exists('./model_parameter/'):
        os.makedirs('./model_parameter/')
    model_save_path = "./model_parameter/"
    model_save_path = model_save_path + model_save_path_filename
    print('Initialized the model, optimizer and scheduler')
    
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
            ', train_l1MaxNormRelList_step / ntrain / (T_out / step)'+
            ', train_l1MaxNormRelList_full / ntrain' +
            ', test_l1MaxNormRelList_step / ntest / (T_out / step)' +
            ', test_l1MaxNormRelList_full / ntest' + 
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
    strn_epochs_dump_path_file6 = 'data/'+'epochs_diagsTime_jit_torchcompile'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'__'
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[0]):
        strn_epochs_dump_path_file6 = strn_epochs_dump_path_file6+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file6 =strn_epochs_dump_path_file6 + "_"
    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
        strn_epochs_dump_path_file6 = strn_epochs_dump_path_file6+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])
    strn_epochs_dump_path_file6 =strn_epochs_dump_path_file6 + ".txt"
    file6 = open(strn_epochs_dump_path_file6, "w") 
    str_file6= ( 'ep' + ', t2 - t1' +  
            ', train_l2_step / ntrain / (T_out / step)'+
            ', train_l2_full / ntrain' +
            ', test_l2_step / ntest / (T_out / step)' +
            ', test_l2_full / ntest' + 
            ', count_params(model)='+str(count_params(model))+
            ', t12mid - t1'+
            ', t2 - t12mid'+'\n' ) 
    file6.write(str_file6)
    file6.close()
    print('Created the output dump files at ./data/')
    return ( 
        sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
        data_read_global,
        train_a_range,train_u_range,test_a_range,test_u_range,
        ntrain,ntest,
        S,S_r,S_theta , T_in,T_out, T_in_steadystate,
        if_IncludeSteadyState, 
        n_beg, startofpatternlist_i_file_no_in_SelectData,
        if_model_Nimrod_STFNO_global,
        if_model_jit_torchCompile, 
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j ,
        ii_sub_fieldlist_parm_eq_vector_train_global_lst_i,
        modes, width,
        batch_size,number_of_layers, learning_rate,iterations,
        input_parameter_order, 
        mWidth_input_parameters, 
        nWidth_output_parameters,
        if_3D,S_n_phi,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        strn_epochs_dump_path_file6,
        strn_epochs_dump_path_file5,
        strn_epochs_dump_path_file4,
        strn_epochs_dump_path_file3,
        strn_epochs_dump_path_file2,
        strn_epochs_dump_path_file1,
        train_loader, test_loader,
        model, optimizer, scheduler,
        count_params_model,model_save_path
        )
# exit(1)