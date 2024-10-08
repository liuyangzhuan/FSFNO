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

from singlePDE_NeuralOperator import singlePDENeuralOperator
from initializationTrainTestParametersAndFile import initializationTrainTestParametersFile
import torch

def multiPDEs_overallsetup(
            data_read_global,
            data_read_global_mean,data_read_global_std,
            data_read_global_eachTimeStep_mean,
            data_read_global_eachTimeStep_std,
            ntrain,ntest,
            S,S_r,S_theta,
            r_theta_phi, 
            T_in,T_out, T_in_steadystate,
            IncludeSteadyState, 
            n_beg, startofpatternlist_i_file_no_in_SelectData,
            model_Nimrod_FNO2d_global  ,
            epochs,
            T_out_sub_time_consecutiveIterator_factor, step,
            batch_size,number_of_layers,learning_rate,iterations,
            i_file_no_in_SelectData, 
            modes, width,
            fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,
            fieldlist_parm_eq_vector_train_global_lst,
            if_model_parameters_load,
            if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
            OneByPowerTransformationFactorOfData,
            log_param,
            nlvls,
            epsilon_inPlottingErrorNormalization,
            input_parameter_order, 
            mWidth_input_parameters, 
            nWidth_output_parameters
            ):

    train_a_range = range(n_beg,n_beg+(T_in*1),1)
    train_u_range = range(n_beg+(T_in*1),n_beg+((T_out+T_in)*1),1)
    test_a_range = range(n_beg,n_beg+(T_in*1),1)
    test_u_range = range(n_beg+(T_in*1),n_beg+((T_out+T_in)*1),1)
    intermediate_parameter_update = False
    for i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i in enumerate(fieldlist_parm_eq_vector_train_global_lst): 
        print('Now going to initialize the training and testing data')
        for ii_sub_fieldlist_parm_eq_vector_train_global_lst_i, sub_fieldlist_parm_eq_vector_train_global_lst_i_ii in enumerate(fieldlist_parm_eq_vector_train_global_lst_i):
                for j_fieldlist_parm_eq_vector_train_global_lst_i, fieldlist_parm_eq_vector_train_global_lst_i_j in enumerate(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii):
                    (sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
                    data_read_global,
                    train_a_range,train_u_range,test_a_range,test_u_range,
                    ntrain,ntest,
                    S,S_r,S_theta , T_in,T_out, T_in_steadystate,
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
                    train_loader ,test_loader,
                    model,optimizer, scheduler,
                    count_params_model,
                    model_save_path) = initializationTrainTestParametersFile( 
                        sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
                        data_read_global,
                        train_a_range,train_u_range,test_a_range,test_u_range,
                        ntrain,ntest,
                        S,S_r,S_theta , T_in,T_out, T_in_steadystate,
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
                        sum_vector_a_elements_i_iter=None, sum_vector_u_elements_i_iter=None,
                        strn_epochs_dump_path_file5=None,
                        strn_epochs_dump_path_file4=None,
                        strn_epochs_dump_path_file3=None,
                        strn_epochs_dump_path_file2=None,
                        strn_epochs_dump_path_file1=None,
                        train_loader = None,test_loader = None,
                        model = None,optimizer=None, scheduler=None,
                        count_params_model = None, model_save_path=None
                    )
                    
                    singlePDENeuralOperator(data_read_global,
                        data_read_global_mean,data_read_global_std,
                        data_read_global_eachTimeStep_mean,
                        data_read_global_eachTimeStep_std,
                        ntrain,ntest,
                        S,S_r,S_theta,
                        r_theta_phi, 
                        T_in,T_out, T_in_steadystate,
                        IncludeSteadyState, 
                        startofpatternlist_i_file_no_in_SelectData,
                        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
                        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
                        epochs,
                        strn_epochs_dump_path_file5,
                        T_out_sub_time_consecutiveIterator_factor, step,
                        batch_size,
                        i_file_no_in_SelectData, 
                        strn_epochs_dump_path_file4,
                        strn_epochs_dump_path_file3,
                        strn_epochs_dump_path_file2,
                        strn_epochs_dump_path_file1,
                        if_model_parameters_load,
                        if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
                        OneByPowerTransformationFactorOfData,
                        log_param,
                        nlvls,
                        epsilon_inPlottingErrorNormalization,
                        model,
                        train_loader,test_loader,
                        optimizer,scheduler,
                        count_params_model,
                        intermediate_parameter_update,
                        model_save_path
                        )
                    