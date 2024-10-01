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

from singlePDE_NeuralOperator import singlePDENeuralOperator
from initializationTrainTestParametersAndFile import initializationTrainTestParametersFile
import torch

def multiPDEs_overallsetup(
            data_read_global,
            data_read_global_eachTimeStep_std,
            data_read_global_eachTimeStep_std_logRMS,
            ntrain,ntest,
            S,S_r,S_theta , T_in,T_out, T_in_steadystate,
            IncludeSteadyState, 
            n_beg, startofpatternlist_i_file_no_in_SelectData,
            model_Nimrod_FNO2d_global  ,
            epochs,
            epochs_ofWeigthModification,
            epochs_ofWeigthModificationFactor,
            T_out_sub_time_consecutiveIterator_factor, step,
            batch_size,number_of_layers,learning_rate,iterations,
            i_file_no_in_SelectData, 
            modes, width,
            fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,
            fieldlist_parm_eq_vector_train_global_lst,
            if_model_parameters_load,
            if_load_best_model_parameters_test_l2_step__if_model_parameters_load_True,
            manual_seed_value_set,
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
    train_a_global = torch.zeros(ntrain,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),S_r ,S_theta,T_in)
    train_u_global = torch.zeros(ntrain,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),S_r ,S_theta,T_out)
    test_a_global = torch.zeros(ntest,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),S_r ,S_theta,T_in)
    test_u_global = torch.zeros(ntest,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),S_r ,S_theta,T_out)
    train_a_global_2_logRMS_RegressionModel = torch.zeros(ntrain,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),T_in)
    train_u_global_2_logRMS_RegressionModel = torch.zeros(ntrain,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),T_out)
    test_a_global_2_logRMS_RegressionModel = torch.zeros(ntest,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),T_in)
    test_u_global_2_logRMS_RegressionModel = torch.zeros(ntest,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),T_out)
    if IncludeSteadyState:
        train_a_global_steadystate = torch.zeros(ntrain,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),S_r ,S_theta,T_in_steadystate)
        test_a_global_steadystate  = torch.zeros(ntest ,len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst),S_r ,S_theta,T_in_steadystate)
    for i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i in enumerate(fieldlist_parm_eq_vector_train_global_lst): 
        sum_vector_train_elements_i_all = 0
        sum_vector_test_elements_i_all = 0 
        for ii_sub_fieldlist_parm_eq_vector_train_global_lst_i, sub_fieldlist_parm_eq_vector_train_global_lst_i_ii in enumerate(fieldlist_parm_eq_vector_train_global_lst_i):
                for j_fieldlist_parm_eq_vector_train_global_lst_i, fieldlist_parm_eq_vector_train_global_lst_i_j in enumerate(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii):
                    sum_vector_train_elements_i_all = sum_vector_train_elements_i_all + len(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii[j_fieldlist_parm_eq_vector_train_global_lst_i][0])
                    sum_vector_test_elements_i_all = sum_vector_test_elements_i_all + len(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii[j_fieldlist_parm_eq_vector_train_global_lst_i][1])
                    for jj_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_jj in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j):
                        for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j_jj):
                            pass
        train_a_global_glob_all = torch.zeros(ntrain,sum_vector_train_elements_i_all,S_r ,S_theta,T_in)
        train_u_global_glob_all = torch.zeros(ntrain,sum_vector_train_elements_i_all,S_r ,S_theta,T_out)
        test_a_global_glob_all  = torch.zeros(ntest, sum_vector_test_elements_i_all,S_r ,S_theta,T_in)
        test_u_global_glob_all  = torch.zeros(ntest, sum_vector_test_elements_i_all,S_r ,S_theta,T_out)
        train_a_global_glob_all_2_logRMS_RegressionModel = torch.zeros(ntrain,sum_vector_train_elements_i_all,T_in)
        train_u_global_glob_all_2_logRMS_RegressionModel = torch.zeros(ntrain,sum_vector_train_elements_i_all,T_out)
        test_a_global_glob_all_2_logRMS_RegressionModel  = torch.zeros(ntest, sum_vector_test_elements_i_all,T_in)
        test_u_global_glob_all_2_logRMS_RegressionModel  = torch.zeros(ntest, sum_vector_test_elements_i_all,T_out)
        intermediate_parameters_dict = {'pressureiperp':[],'pressureipara':[],'flowi':[],'apara_2d':[],'phi_2d':[],'densityi':[],'x':[],'y':[]}
        intermediate_parameter_update = False
        intermediate_parameters_dict_list = []
        if intermediate_parameter_update: 
            if  fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] in intermediate_parameters_dict:
                list0 = []
        if intermediate_parameter_update:
            if  not intermediate_parameters_dict[fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] ]:
                list0 = []
        if intermediate_parameter_update:
            intermediate_parameters_dict[fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] ].append( intermediate_parameters_dict )
            list0 = []
        for ii_sub_fieldlist_parm_eq_vector_train_global_lst_i, sub_fieldlist_parm_eq_vector_train_global_lst_i_ii in enumerate(fieldlist_parm_eq_vector_train_global_lst_i):
                for j_fieldlist_parm_eq_vector_train_global_lst_i, fieldlist_parm_eq_vector_train_global_lst_i_j in enumerate(sub_fieldlist_parm_eq_vector_train_global_lst_i_ii):
                    











                    model = None
                    (sub_fieldlist_parm_eq_vector_train_global_lst_i_ii, j_fieldlist_parm_eq_vector_train_global_lst_i,
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
                    train_loader ,test_loader,
                    model,optimizer, scheduler,
                    count_params_model,
                    model_save_path,
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
                    ) = initializationTrainTestParametersFile( 
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
                        sum_vector_a_elements_i_iter=None, sum_vector_u_elements_i_iter=None,
                        strn_epochs_dump_path_file5=None,
                        strn_epochs_dump_path_file4=None,
                        strn_epochs_dump_path_file3=None,
                        strn_epochs_dump_path_file2=None,
                        strn_epochs_dump_path_file1=None,
                        train_loader = None,test_loader = None,
                        model = None,optimizer=None, scheduler=None,
                        count_params_model = None, model_save_path=None,
                        model_2_logRMS_RegressionModel = None,
                        model_2_logRMS_RegressionModel_save_path = None,
                        train_loader_2_logRMS_RegressionModel = None,
                        test_loader_2_logRMS_RegressionModel = None,
                        optimizer_2_logRMS_RegressionModel = None, scheduler_2_logRMS_RegressionModel = None,
                        count_params_model_2_logRMS_RegressionModel = None,
                        strn_epochs_dump_path_file22 = None,
                        best_loss_test_l2_step = None,
                        best_model_state_dict_test_l2_step = None,
                        best_model_save_path_test_l2_step = None                      
                    )
                    
                    singlePDENeuralOperator(
                        data_read_global,
                        data_read_global_eachTimeStep_std,
                        ntrain,ntest,
                        S_r,S_theta , T_in,T_out, T_in_steadystate,
                        IncludeSteadyState, 
                        startofpatternlist_i_file_no_in_SelectData,
                        model_Nimrod_FNO2d_global,
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
                        intermediate_parameter_update,
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
                        )