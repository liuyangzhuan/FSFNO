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
import random
from stfno.utilities3 import *
from i_file_no_in_original_data import i_file_no_in_original
from dumpfiledata_h5py_i_file_no_in_original_data import dumpfiledata_h5py_i_file_no
from readfiledata_h5py_i_file_no_in_SelectData import readfiledata_h5py_ifilenoinSelectData
from multiPDEs_overall_setup import multiPDEs_overallsetup
from GTC_pde_operator_parameters import GTC_pde_operator_parameters_defination

manual_seed_value_set = 0
torch.manual_seed(manual_seed_value_set)
np.random.seed(manual_seed_value_set)
error_codes = ["Uncompleted run", "Uncorrect gtc.out format",
               "Unfinished history", "NaN detected"]

epsilon_inPlottingErrorNormalization = 1e-6
random_seed_i_file_no_in_SelectData = 0
T_in_steadystate=1
T_steadystate=1
number_of_layers = 4
if_IncludeSteadyState = False #True #True
if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot = False # True #True
Option_NormalizingTrainTestData = 3 #2 or 1 or by default no normaliztion  True
if_model_Nimrod_STFNO_global = True #False
if_model_parameters_load = False
if_load_best_model_parameters_test_l2_step__if_model_parameters_load_True = False
random_seed_i_file_no_in_SelectData = 0
factor_ntrain_by_ntrainPlusntest = 0.50
sub = 1
T_in = 1 #10 #10 #10#200
T_out = 1 #40 #20 #300 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
T_out_sub_time_consecutiveIterator_factor = T_out  ##Must be a factor of T_out
step = T_out_sub_time_consecutiveIterator_factor
print(' T_in =',T_in)
print(' T_out =',T_out)
print(' step =',step)
print(' number of layers',number_of_layers)
modes = 12
width = 20
batch_size = 1 #32 #64
learning_rate = 0.001
epochs = 500 # 1
epochs_ofWeigthModification = epochs
epochs_ofWeigthModificationFactor = epochs + 2
OneByPowerTransformationFactorOfData = 1.0

S =64
nx_r =S #64 #32 
phi=0
nx_theta = S #8
nx_phi = S
theta_begn =0
theta_end = 2*np.pi-2*np.pi*1.0/S

n_beg=0
r_cntr, globl_r_end_estmt,globl_r_begn_estmt,if_HyperDiffusivity_case,if_2ndRunHyperDiffusivity_case = 0,1,0, False, False
i_file_no_in, path = i_file_no_in_original(nx_r, nx_theta,theta_begn, theta_end,r_cntr, globl_r_end_estmt,globl_r_begn_estmt,phi,if_HyperDiffusivity_case,if_2ndRunHyperDiffusivity_case, n_beg)

(fieldlist_parm_lst,
    fieldlist_parm_vector_lst,
    fieldlist_parm_eq_range, 
    fieldlist_parm_vector_chosen, 
    fieldlist_parm_eq_vector_train_global_lst,
    input_parameter_order, 
    mWidth_input_parameters, 
    nWidth_output_parameters ) = GTC_pde_operator_parameters_defination()
if_dumpfiledata= False
if if_dumpfiledata:
    dumpfiledata_h5py_i_file_no(fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,S,i_file_no_in,r_theta_phi, path)
if_readdumpfiledata= True
path_data_read ='/global/cfs/cdirs/mp127/nimrod/GTC_timeSequenceData/common_11__sfluidne__pressureiperp__pressureipara__flowi__apara_2d__phi_2d__densityi__dumpData_corrected/'
if if_readdumpfiledata:
    for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst[:1]):
            for i_fieldlist_parm_eq_selected, fieldlist_parm_eq_selected in enumerate(range(fieldlist_parm_eq_range)):
                for i_fieldlist_parm_vector_i, fieldlist_parm_vector_i in  enumerate(range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst])):
                    strn_read_path = path_data_read + 'data/'+fieldlist_parm_lst_i+'_eq' + str(fieldlist_parm_eq_selected)+'_vec' +str(fieldlist_parm_vector_i) +'.hdf5'
                    h5file = h5py.File(strn_read_path,'r')
                    data_read = h5file[fieldlist_parm_lst_i]
                    data_read = data_read[()]
                    h5file.close()
                    nx_r     = data_read.shape[-2] 
                    nx_theta = data_read.shape[-1]
S_r = nx_r #S
S_theta = nx_theta #S
S_n_phi = nx_phi #S
i_nphi_plot2DContour = 0
S_r_hdf5file = S_r # >= S_r
S_theta_hdf5file = S_theta  # >= S_theta
S_phi_hdf5file = nx_phi # >= S_phi
nlvls =50
log_param= False
grid_x_lin = np.arange(S_r) #linspace(0,1,num=y.size()[-1])
grid_y_lin = np.arange(S_theta) #linspace(0,1,num=y.size()[-2])
grid_x, grid_y = np.meshgrid(grid_x_lin, grid_y_lin)
print(' S=',S)

print(' Reading h5py data from the path',path_data_read)
if if_readdumpfiledata:
    (data_read_global,
     i_file_no_in_SelectData,
     path_data_read
        ) = readfiledata_h5py_ifilenoinSelectData(
            if_HyperDiffusivity_case, if_2ndRunHyperDiffusivity_case,S,
            fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,
            path_data_read,
            Option_NormalizingTrainTestData,i_file_no_in,
            OneByPowerTransformationFactorOfData,
            nx_r, 
            nx_theta 
            )

nx_r     = nx_r 
nx_theta = nx_theta -1 
S_r = nx_r #S
S_theta = nx_theta #S
S_n_phi = nx_phi #S
i_nphi_plot2DContour = 0
S_r_hdf5file = S_r # >= S_r
S_theta_hdf5file = S_theta  # >= S_theta
S_phi_hdf5file = nx_phi # >= S_phi
nlvls =50
log= False
grid_x_lin = np.arange(S_r) #linspace(0,1,num=y.size()[-1])
grid_y_lin = np.arange(S_theta) #linspace(0,1,num=y.size()[-2])
grid_x, grid_y = np.meshgrid(grid_x_lin, grid_y_lin)
data_read_global_mean = torch.zeros(len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
data_read_global_std  = torch.zeros(len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
data_read_global_eachTimeStep_mean = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
data_read_global_eachTimeStep_std  = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
data_read_global_eachTimeStep_std_logRMS  = torch.ones(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
    x_cordinates_pmeshplot = data_read_global[ 0,
                                    7 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    y_cordinates_pmeshplot = data_read_global[ 0,
                                    8 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    x_cordinates_pmeshplot_periodic = x_cordinates_pmeshplot # np.hstack( (x_cordinates_pmeshplot,x_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
    y_cordinates_pmeshplot_periodic = y_cordinates_pmeshplot #np.hstack( (y_cordinates_pmeshplot,y_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
else:
    x_cordinates_pmeshplot = data_read_global[ 0,
                                    7 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    y_cordinates_pmeshplot = data_read_global[ 0,
                                    8 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    x_cordinates_pmeshplot_periodic = np.hstack( (x_cordinates_pmeshplot,x_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
    y_cordinates_pmeshplot_periodic = np.hstack( (y_cordinates_pmeshplot,y_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
x_cordinates_pmeshplot_periodic = np.power(x_cordinates_pmeshplot_periodic,OneByPowerTransformationFactorOfData )
y_cordinates_pmeshplot_periodic = np.power(y_cordinates_pmeshplot_periodic,OneByPowerTransformationFactorOfData )
data_read_global = data_read_global[:,:,:,:,:,:-1]
if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
    x_cordinates_pmeshplot = data_read_global[ 0,
                                    7 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    y_cordinates_pmeshplot = data_read_global[ 0,
                                    8 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    x_cordinates_pmeshplot_periodic = x_cordinates_pmeshplot # np.hstack( (x_cordinates_pmeshplot,x_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
    y_cordinates_pmeshplot_periodic = y_cordinates_pmeshplot #np.hstack( (y_cordinates_pmeshplot,y_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
else:
    x_cordinates_pmeshplot = data_read_global[ 0,
                                    7 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    y_cordinates_pmeshplot = data_read_global[ 0,
                                    8 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                    ,0,0,:,:]
    x_cordinates_pmeshplot_periodic = np.hstack( (x_cordinates_pmeshplot,x_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
    y_cordinates_pmeshplot_periodic = np.hstack( (y_cordinates_pmeshplot,y_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
    pass
x_cordinates_pmeshplot_periodic = np.power(x_cordinates_pmeshplot_periodic,OneByPowerTransformationFactorOfData )
y_cordinates_pmeshplot_periodic = np.power(y_cordinates_pmeshplot_periodic,OneByPowerTransformationFactorOfData )
print(' x_cordinates_pmeshplot_periodic.shape ', x_cordinates_pmeshplot_periodic.shape )
print(' y_cordinates_pmeshplot_periodic.shape ', y_cordinates_pmeshplot_periodic.shape )

for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst):
        for i_fieldlist_parm_eq_selected, fieldlist_parm_eq_selected in enumerate(range(fieldlist_parm_eq_range)):
            for i_fieldlist_parm_vector_i, fieldlist_parm_vector_i in  enumerate(range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst])):
                data_read_global_mean[i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow( torch.mean(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] )  ,  1.0/1.0)
                data_read_global_std [i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow(  torch.std(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] ) ,  1.0/1.0)
                for i_fieldlist_parm_ifilenoin_i, fieldlist_parm_ifilenoin_i in  enumerate(range(len(i_file_no_in))): 
                    if Option_NormalizingTrainTestData == 3:
                        data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i]  = 0.0
                        data_read_global_eachTimeStep_std[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i]  = torch.sqrt(torch.mean(torch.pow(data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:], 2)))
                        data_read_global_eachTimeStep_std_logRMS [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.log( data_read_global_eachTimeStep_std[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] )
                    elif Option_NormalizingTrainTestData == 2:
                        data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow( torch.mean(data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] ) , 1.0) 
                        data_read_global_eachTimeStep_std [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow(  torch.std(data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] ) , 1.0) 
                    elif Option_NormalizingTrainTestData == 1:
                        data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow( torch.mean(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] )  , 1.0) 
                        data_read_global_eachTimeStep_std [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow(  torch.std(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] )  , 1.0) 
                    else:
                        data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 0.0
                        data_read_global_eachTimeStep_std [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 1.0
                        data_read_global_mean[i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 0.0 #torch.mean(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] ) 
                        data_read_global_std [i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 1.0 # torch.std(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] )
                    if data_read_global_eachTimeStep_std[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] == 0:
                        exit(1) 
                    else:
                        data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:]  = (
                                torch.pow(((data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:]
                                -data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i])
                                /data_read_global_eachTimeStep_std[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] ), 1.0/1.0 ) )
sign_ofPowerTransformationOfData = torch.sign(data_read_global[:,:,:,:,:])
data_read_global[:,:,:,:,:] = torch.pow( torch.abs(data_read_global[:,:,:,:,:]).to(torch.float64),1.0/ OneByPowerTransformationFactorOfData) * sign_ofPowerTransformationOfData
if torch.any(torch.isnan(data_read_global[:,:,:,:,:])): #data_dump[:,:,:,:])):                    
    print("data_read_global:", data_read_global)
    nan_mask = torch.isnan(data_read_global[:,:,:,:,:])
    print("NaN mask:", nan_mask)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)
    print("Indices of NaN values:", nan_indices)
    exit(0)
if torch.any(torch.isnan(data_read_global_eachTimeStep_std_logRMS[:,:,:])): #data_dump[:,:,:,:])):                    
    print("B data_read_global:", data_read_global_eachTimeStep_std_logRMS)
    nan_mask = torch.isnan(data_read_global_eachTimeStep_std_logRMS[:,:,:])
    print("B NaN mask:", nan_mask)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)
    print("B Indices of NaN values:", nan_indices)
    exit(0)
if ( (torch.any( data_read_global_eachTimeStep_std_logRMS[:,:,:]==0 ))): #data_dump[:,:,:,:])):                    
    print("C data_read_global:", data_read_global_eachTimeStep_std_logRMS)
    nan_mask = torch.is_nonzero(data_read_global_eachTimeStep_std_logRMS[:,:,:])
    print("C NaN mask:", nan_mask)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)
    print("C Indices of NaN values:", nan_indices)
    exit(0)
total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData  = len(i_file_no_in_SelectData) - T_in - T_out 
if total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData < 0:
    exit(0)
startofpatternlist_i_file_no_in_SelectData = list(range( total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData ))
random.seed(random_seed_i_file_no_in_SelectData)
random.shuffle(startofpatternlist_i_file_no_in_SelectData)
ntrain = int(total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData * factor_ntrain_by_ntrainPlusntest)
ntest = total_number_of_set_ntrain_plus_ntest_possible_i_file_no_in_SelectData - ntrain
print(' ntrain = ',ntrain)
print(' ntest = ',ntest)

i_file_no_in_SelectData = np.array(i_file_no_in_SelectData)
startofpatternlist_i_file_no_in_SelectData = np.array(startofpatternlist_i_file_no_in_SelectData)
print(' snapshots for training: ',i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[:ntrain]])
print(' snapshots for testing: ',i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[-ntest:]])
iterations = epochs*(ntrain//batch_size)



multiPDEs_overallsetup(
            data_read_global,
            data_read_global_eachTimeStep_std,
            data_read_global_eachTimeStep_std_logRMS,
            ntrain,ntest,
            S,S_r,S_theta , T_in,T_out, T_in_steadystate,
            if_IncludeSteadyState, 
            n_beg, startofpatternlist_i_file_no_in_SelectData,
            if_model_Nimrod_STFNO_global  ,
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
            )
exit(1) 
