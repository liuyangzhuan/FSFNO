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
import numpy as np
import random
from rBegn_rEnd_estimate import globl_r_begn_r_end_estmt_theta
from i_file_no_in_original_data import i_file_no_in_original
from dumpfiledata_h5py_i_file_no_in_original_data import dumpfiledata_h5py_i_file_no
from readfiledata_h5py_i_file_no_in_SelectData import readfiledata_h5py_ifilenoinSelectData
from multiPDEs_overall_setup import multiPDEs_overallsetup
from NIMROD_pde_operator_parameters import NIMROD_pde_operator_parameters_defination

manual_seed_value_set = 0
torch.manual_seed(manual_seed_value_set)
np.random.seed(manual_seed_value_set)
epsilon_inPlottingErrorNormalization = 1e-6

T_in_steadystate=1
T_steadystate=1
number_of_layers = 1
if_IncludeSteadyState = False #True #True
if_HyperDiffusivity_case= True
if_2ndRunHyperDiffusivity_case = True
if_3D =True

if_postTraingAndTesting_ContourPlotsOfTestingData = True
if_GTCLinearNonLinear_case = False 
if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot = False

if_model_parameters_load = False
if_model_jit_torchCompile = False
if_postprocessing_relativeErrorEachTestDataSample = False
if_postprocessing_inferenceTimeTestData = False
if_intermediate_parameter_update = False

Option_NormalizingTrainTestData = 1 #2 or 1 or by default no normaliztion  True
OneByPowerTransformationFactorOfData = 1.0
if_model_Nimrod_STFNO_global = True
random_seed_i_file_no_in_SelectData = 0
factor_ntrain_by_ntrainPlusntest = 0.50
if if_HyperDiffusivity_case:
    ntrain = 66 # 12 # 4 #10 #* 3 #100
    ntest  = 66 # 12 # 4 #10 #* 3#20
else: 
    ntrain = 8 # 12 # 4 #10 #* 3 #100
    ntest  = 8 # 12 # 4 #10 #* 3#20
sub = 1
S = 64 #256 # 64 #32 # #256
T_in = 1 #10 #10 #10#200

T_out = 1 #40 #20 #300 
T_out_sub_time_consecutiveIterator_factor = T_out  ##Must be a factor of T_out
step = T_out_sub_time_consecutiveIterator_factor
print(' T_in =',T_in)
print(' T_out =',T_out)
print(' step =',step)
print(' number of layers',number_of_layers)
step = 1
modes = 12
width = 20
S_r = S 
S_theta = S
S_n_phi = 64 #S
i_nphi_plot2DContour = 0
S_r_hdf5file = S_r # >= S_r
S_theta_hdf5file = S_theta  # >= S_theta
S_phi_hdf5file = 64 # >= S_phi


S_r = S 
S_theta = S
S_n_phi = 64 #S
i_nphi_plot2DContour = 0
S_r_hdf5file = S_r # >= S_r
S_theta_hdf5file = S_theta  # >= S_theta
S_phi_hdf5file = 64 # >= S_phi

batch_size = 1 #1 # 20
learning_rate = 0.001
epochs = 1 #500 #1000
iterations = epochs*(ntrain//batch_size)
nx_r =S_r
dr_step_guess_cfl = 0.5
phi=0
phi_begn =0 
phi_end = 2*np.pi  

if if_3D:
    nx_phi = S_n_phi    
else:
    nx_phi = 1

nx_theta = S_theta #8
theta_begn =0
theta_end = 2*np.pi-2*np.pi*1.0/S_theta
r_cntr = 1.744
r_begn_theta0 = 2.108
r_end_theta0 = 2.442

globl_r_begn_estmt,globl_r_end_estmt = globl_r_begn_r_end_estmt_theta(S,nx_theta, r_cntr, r_begn_theta0, r_end_theta0)

n_beg = 0
r_theta_phi, i_file_no_in, path = i_file_no_in_original(nx_r, nx_theta,theta_begn, theta_end,r_cntr, globl_r_end_estmt,globl_r_begn_estmt,phi,if_HyperDiffusivity_case,if_2ndRunHyperDiffusivity_case, n_beg,if_3D,nx_phi,phi_begn,phi_end)

(fieldlist_parm_lst,
    fieldlist_parm_vector_lst,
    fieldlist_parm_eq_range, 
    fieldlist_parm_vector_chosen, 
    fieldlist_parm_eq_vector_train_global_lst,
    input_parameter_order, 
    mWidth_input_parameters, 
    nWidth_output_parameters ) = NIMROD_pde_operator_parameters_defination(
                                if_2ndRunHyperDiffusivity_case, S, if_3D)

if if_3D:
    data_dump_hdf5file = torch.zeros(S_r_hdf5file,S_theta_hdf5file,S_n_phi)
else: 
    data_dump_tmp = torch.zeros(S,S)
if_dumpfiledata= False
if if_dumpfiledata:
    dumpfiledata_h5py_i_file_no(fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,S,i_file_no_in,r_theta_phi, path,if_3D)

if_readdumpfiledata= True

S_r = nx_r #S
S_theta = nx_theta #S
S_n_phi = nx_phi #64 #S
i_nphi_plot2DContour = 0
S_r_hdf5file = S_r # >= S_r
S_theta_hdf5file = S_theta  # >= S_theta
S_phi_hdf5file = nx_phi #64 # >= S_phi
nlvls =50
log_param= False
# grid_x_lin = np.arange(S_r) #linspace(0,1,num=y.size()[-1])
# grid_y_lin = np.arange(S_theta) #linspace(0,1,num=y.size()[-2])
# grid_x, grid_y = np.meshgrid(grid_x_lin, grid_y_lin)
print(' S_r=',S_r, 'S_theta=',S_theta,'S_n_phi=',S_n_phi)
if if_HyperDiffusivity_case:
    if if_3D:
        path_data_read ='/global/cfs/cdirs/mp127/nimrod/nimrod_hdf513_simulationII3Dhypdifusvity_dump_data/npvjb_S32_dump_data_2/'
    else:
        if if_2ndRunHyperDiffusivity_case:
            if S == 64:
                path_data_read ='../../../../../nimrod_hdf510_ZheBaihypdifusv_dump_data/npvjb_S64_dump_data_corrected_combinedTemperature/'    
            else:
                path_data_read ='../../nimrod_hdf510_ZheBaihypdifusv_dump_data/npvjb_S32_dump_data/'
        else:    
            if S == 64:
                path_data_read ='../../nimrod_hdf507_hypdifusv_dump_data/npvbj_S64_dump_data/'
            else:
                path_data_read ='../../nimrod_hdf507_hypdifusv_dump_data/npvbj__dump_data/'
else:
    path_data_read ='../../nimrod_hdf506_kinematic_dump_data/npvjb_S32_dump_data/'

print(' Reading h5py data from the path:',path_data_read)

if if_readdumpfiledata:
    (data_read_global,
     data_read_global_mean,data_read_global_std,
     data_read_global_eachTimeStep_mean,data_read_global_eachTimeStep_std,
     i_file_no_in_SelectData,
     path_data_read ) = readfiledata_h5py_ifilenoinSelectData(
                        if_HyperDiffusivity_case, if_2ndRunHyperDiffusivity_case,nx_r ,
                        fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,
                        path_data_read,
                        Option_NormalizingTrainTestData,i_file_no_in,
                        OneByPowerTransformationFactorOfData,
                        nx_phi,nx_theta,if_3D
                        )
total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData  = len(i_file_no_in_SelectData) - T_in - T_out 
if total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData < 0:
    exit(1)
ntrain = int(total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData * factor_ntrain_by_ntrainPlusntest)
ntest = total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData - ntrain
sequential_splitting_option = False # False is default for random splitting
if sequential_splitting_option: 
    startofpatternlist_i_file_no_in_SelectData = list(range( ntrain ))
else:
    startofpatternlist_i_file_no_in_SelectData = list(range( total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData ))
random.seed(random_seed_i_file_no_in_SelectData)
random.shuffle(startofpatternlist_i_file_no_in_SelectData)
if sequential_splitting_option: 
    startofpatternlist_i_file_no_in_SelectData.extend( range(ntrain,total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData) ) 

print(' ntrain = ',ntrain)
print(' ntest = ',ntest)
print(' snapshots for training: ',[i_file_no_in_SelectData[i] for i in startofpatternlist_i_file_no_in_SelectData[:ntrain] ])
print(' snapshots for testing: ',[i_file_no_in_SelectData[i]  for i in startofpatternlist_i_file_no_in_SelectData[-ntest:] ])

iterations = epochs*(ntrain//batch_size)
nbeg=0
i_nphi_plot2DContour = 0

n_phi_plot_count = 4
n_phi_plot_begin_factor = S_n_phi // 10.0
phi_i__plot_scalar_plane_psi_theta = 0 # Integer between 0 to S_n_phi

multiPDEs_overallsetup(
            data_read_global,
            data_read_global_mean,data_read_global_std,
            data_read_global_eachTimeStep_mean,
            data_read_global_eachTimeStep_std,
            ntrain,ntest,
            S,S_r,S_theta,
            r_theta_phi, 
            T_in,T_out, T_in_steadystate,
            if_IncludeSteadyState, 
            n_beg, startofpatternlist_i_file_no_in_SelectData,
            if_model_Nimrod_STFNO_global  ,
            epochs,
            T_out_sub_time_consecutiveIterator_factor, step,
            batch_size,number_of_layers,learning_rate,iterations,
            i_file_no_in_SelectData, 
            modes, width,
            fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,
            fieldlist_parm_eq_vector_train_global_lst,
            if_model_parameters_load,
            if_postprocessing_relativeErrorEachTestDataSample,
            if_postTraingAndTesting_ContourPlotsOfTestingData,
            if_postprocessing_inferenceTimeTestData,
            if_model_jit_torchCompile,        
            if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
            OneByPowerTransformationFactorOfData,
            log_param,
            nlvls,
            epsilon_inPlottingErrorNormalization,
            input_parameter_order, 
            mWidth_input_parameters, 
            nWidth_output_parameters,
            if_intermediate_parameter_update,
            if_3D,
            S_n_phi, phi_begn, phi_end, i_nphi_plot2DContour,
            theta_begn, theta_end, 
            r_cntr, globl_r_begn_estmt,globl_r_end_estmt,
            n_phi_plot_count, n_phi_plot_begin_factor
            )
# exit() 