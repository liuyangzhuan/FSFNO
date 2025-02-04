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

import numpy
import torch
import h5py
# import os
# import sys

def readfiledata_h5py_ifilenoinSelectData(
        if_HyperDiffusivity_case, if_2ndRunHyperDiffusivity_case,S_r,
        fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,
        path_data_read,
        Option_NormalizingTrainTestData,i_file_no_in,
        OneByPowerTransformationFactorOfData,
        S_theta,S_n_phi, if_3D
        ):    
    if if_3D:
        data_read_global_hdf5file = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst), S_r,S_theta,S_n_phi)
        data_read_global          = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst), S_r,S_theta,S_n_phi)
    else: 
        data_read_global = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst), S_r,S_theta)
    for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst):
            if fieldlist_parm_lst_i=='e':
                fieldlist_parm_lst_i_read = 'te'
                pass
            elif fieldlist_parm_lst_i=='i':
                fieldlist_parm_lst_i_read = 'ti'
                pass
            else:
                fieldlist_parm_lst_i_read = fieldlist_parm_lst_i
            for i_fieldlist_parm_eq_selected, fieldlist_parm_eq_selected in enumerate(range(fieldlist_parm_eq_range)):
                for i_fieldlist_parm_vector_i, fieldlist_parm_vector_i in  enumerate(range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst])):
                    strn_read_path = path_data_read + 'data/'+fieldlist_parm_lst_i+'_eq' + str(fieldlist_parm_eq_selected)+'_vec' +str(fieldlist_parm_vector_i) +'.hdf5'
                    h5file = h5py.File(strn_read_path,'r')
                    data_read = h5file[fieldlist_parm_lst_i_read]
                    data_read = data_read[()]
                    h5file.close()
                    if if_3D:
                        data_read_global_hdf5file[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:,:] = torch.from_numpy(data_read).to(data_read_global)
                    else: 
                        data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] = torch.from_numpy(data_read).to(data_read_global)
    if if_3D:
        data_read_global = data_read_global_hdf5file[:,:,:,:,:S_r,:S_theta,:S_n_phi]
    if torch.any(torch.isnan(data_read_global[:,:,:,:,:,:])): #data_dump[:,:,:,:])):
            exit(1)
    if if_HyperDiffusivity_case:
        if if_2ndRunHyperDiffusivity_case:
            i_file_no_in_SelectData=[
                12000, 12200, 12400, 12600, 12800, 13000, 13200, 13400, 13600, 13800, 14000, 14200, 14400, 14600, 14800, 15000, 15200, 15400, 15600, 15800, 16000, 16200, 16400, 16600, 16800, 17000, 17200, 17400, 17600, 17800, 18000, 18200, 18400, 18600, 18800, 19000, 19200, 19400, 19600, 19800
                , 20000, 20200, 20400, 20600, 20800, 21000, 21200, 21400, 21600, 21800, 22000, 22200, 22400, 22600, 22800, 23000, 23200, 23400, 23600, 23800, 24000, 24200, 24400, 24600, 24800, 25000, 25200, 25400, 25600, 25800, 26000, 26200, 26400, 26600, 26800, 27000, 27200, 27400, 27600, 27800, 28000, 28200, 28400, 28600, 28800, 29000, 29200, 29400, 29600, 29800
                , 30000, 30200, 30400, 30600, 30800, 31000, 31200, 31400, 31600, 31800, 32000, 32200, 32400, 32600, 32800, 33000, 33200, 33400, 33600, 33800, 34000, 34200, 34400, 34600, 34800, 35000, 35200, 35400, 35600, 35800, 36000, 36200, 36400, 36600, 36800, 37000, 37200, 37400, 37600, 37800, 38000, 38200, 38400, 38600, 38800, 39000, 39200, 39400, 39600, 39800
                , 40000, 40200, 40400, 40600, 40800, 41000, 41200, 41400, 41600, 41800, 42000, 42200, 42400, 42600, 42800, 43000, 43200, 43400, 43600, 43800, 44000, 44200, 44400, 44600, 44800, 45000, 45200, 45400, 45600, 45800, 46000, 46200, 46400, 46600, 46800, 47000, 47200, 47400, 47600, 47800, 48000, 48200, 48400, 48600, 48800, 49000, 49200, 49400, 49600, 49800
                , 50000, 50200, 50400, 50600, 50800, 51000, 51200, 51400, 51600, 51800, 52000, 52200, 52400, 52600, 52800, 53000, 53200, 53400, 53600, 53800, 54000, 54200, 54400, 54600, 54800, 55000, 55200, 55400, 55600, 55800, 56000, 56200, 56400, 56600, 56800, 57000, 57200, 57400, 57600, 57800, 58000, 58200, 58400, 58600, 58800, 59000, 59200, 59400, 59600, 59800
                , 60000, 60200, 60400, 60600, 60800, 61000, 61200, 61400, 61600, 61800, 62000, 62200, 62400, 62600, 62800, 63000, 63200, 63400, 63600
            ]
        else:       
            i_file_no_in_SelectData =   [
                        45156,45230,45303,45377,45453,45589,45665,45799,45883,45970,   46060,
                        46200,46297,46398,46495,46590,46675,46750,46811,46918,47045,   47121,
                        47156,47199,47339,47504,47611,47792,47896,48069,48231,48398,   48516,
                        48636,48749,48859,49031,49156,49192,49375,49494,49677,49779,   49943,
                        50129,50249,50359,50457,50548,50644,50744,50914,51085,51156,   51173,
                        51255,51410,51567,51660,51813,51961,52050,52129,52214,52300,   52393,
                        52480,52572,52661,52745,52823,52907,52998,53091,53156,53182,   53272,
                        53362,53507,53591,53681,53778,53868,53954,54041,54184,54334,   54489,
                        54640,54723,54807,54889,54980,55137,55156,55288,55428,55581,   55742,
                        55899,56050,56139,56291,56382,56473,56619,56764,56853,57006,   57099,
                        57156,57189,57343,57496,57577,57654,57747,57869,57957,58034,   58116,
                        58193,58284,58378,58532,58621,58707,58860,59010,59102,59156,   59259,
                        59408,59565,59724,59800,59897,59990,60084,60224,60378,60528,   60688,
                        60838,60928,61022,61114,61156,61205,61296,61384,61476,61565,   61659,
                        61753,61852,61950,62046,62143,62231,62383,62543,62707,62803,   62890,
                        62979,63071,63156,63163,63258,63358,63451,63543,63629,63712,   63864,
                        64016,64105,64263,64350,64435,64521,64615,64708,64803,64899,   64986,
                        65074,65156,65158,65247,65339,65435,65531,65695,65837,65990,   66153,
                        66326,66473,66552,66634,66720,66889,67058,67156,67208,67352,   67506,
                        67674,67767,67857,67948,68016,68106,68259,68363,68473,68581,   68684,
                        68793,68893,68985,69089,69156,69205,69326,69444,69550,69678,   69793,
                        69921,70057,70187,70311,70439,70568,70672,70800,70931,71063,   71156,
                        71185,71310,71446,71565,71696,71829,71960,72085,72212,72341,   72464,
                        72681,72807,72937,73059,73156,73190,73312,73437,73568,73688,   73814
                        ]
    else:
        i_file_no_in_SelectData = [   
                        334,   667,  1000,      1334,  
                        1667,  2001,  2334,      2667,  
                        3001,  3334,  3667,      4001,  
                        5291,  7291,  9291,      11291, 
                    14239, 15952, 18197,      20279, 
                    20341, 20405, 20466,      20533, 
                    20596, 20654, 20711,      20775, 
                    20833, 20890, 20935,      20988, 
                    ]
    k=-1
    for i in range(len(i_file_no_in)):
        k=k+1
        if k<len(i_file_no_in_SelectData):
            if i_file_no_in[i] != i_file_no_in_SelectData[k]  :
                data_read_global = numpy.delete(data_read_global, (k), axis=0)
                k=k-1
        else:
                data_read_global = numpy.delete(data_read_global, (k), axis=0)
                k=k-1
    data_read_global_mean = torch.zeros(len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
    data_read_global_std  = torch.zeros(len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
    i_file_no_in = i_file_no_in_SelectData
    data_read_global_eachTimeStep_mean = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
    data_read_global_eachTimeStep_std  = torch.zeros(len(i_file_no_in),len(fieldlist_parm_lst),fieldlist_parm_eq_range,max(fieldlist_parm_vector_lst))
    
    for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst):
                for i_fieldlist_parm_eq_selected, fieldlist_parm_eq_selected in enumerate(range(fieldlist_parm_eq_range)):
                    for i_fieldlist_parm_vector_i, fieldlist_parm_vector_i in  enumerate(range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst])):
                        data_read_global_mean[i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow( torch.mean(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...] )  ,  1.0/1.0)
                        data_read_global_std [i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow(  torch.std(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...] ) ,  1.0/1.0)
                        
                        for i_fieldlist_parm_ifilenoin_i, fieldlist_parm_ifilenoin_i in  enumerate(range(len(i_file_no_in))): 
                            if Option_NormalizingTrainTestData == 2:
                                data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow( torch.mean(data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...] ) , 1.0) 
                                data_read_global_eachTimeStep_std [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow(  torch.std(data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...] ) , 1.0) 
                            elif Option_NormalizingTrainTestData == 1:                                
                                data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow( torch.mean(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...] )  , 1.0) 
                                data_read_global_eachTimeStep_std [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = torch.pow(  torch.std(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...] )  , 1.0) 
                            else:
                                data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 0.0
                                data_read_global_eachTimeStep_std [i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 1.0
                                data_read_global_mean[i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 0.0 #torch.mean(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] ) 
                                data_read_global_std [i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] = 1.0 # torch.std(data_read_global[:,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,:,:] )
    for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst):
                for i_fieldlist_parm_eq_selected, fieldlist_parm_eq_selected in enumerate(range(fieldlist_parm_eq_range)):
                    for i_fieldlist_parm_vector_i, fieldlist_parm_vector_i in  enumerate(range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst])):
                        for i_fieldlist_parm_ifilenoin_i, fieldlist_parm_ifilenoin_i in  enumerate(range(len(i_file_no_in))): 
                            if data_read_global_eachTimeStep_std[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] == 0:
                                print('readfiledata_h5py_i_file_no_in_SelectData.py exit(1)')
                                exit(1) 
                            else:
                                data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...]  = (
                                            torch.pow(((data_read_global[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i,...]
                                            -data_read_global_eachTimeStep_mean[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i])
                                            /data_read_global_eachTimeStep_std[i_fieldlist_parm_ifilenoin_i,i_fieldlist_parm_lst,i_fieldlist_parm_eq_selected,i_fieldlist_parm_vector_i] ), 1.0/1.0 ) )
    sign_ofPowerTransformationOfData = torch.sign(data_read_global[...])
    data_read_global[...] = torch.pow( torch.abs(data_read_global[...]).to(torch.float64),1.0/ OneByPowerTransformationFactorOfData) * sign_ofPowerTransformationOfData
    if torch.any(torch.isnan(data_read_global[...])): #data_dump[:,:,:,:])):                    
        print(" data_read_global:", data_read_global)
        nan_mask = torch.isnan(data_read_global[...])
        print(" NaN mask:", nan_mask)
        nan_indices = torch.nonzero(nan_mask, as_tuple=True)
        print(" Indices of NaN values:", nan_indices)
    print(' Data Normalized')
    return (data_read_global,data_read_global_mean,data_read_global_std,data_read_global_eachTimeStep_mean,data_read_global_eachTimeStep_std,i_file_no_in_SelectData,path_data_read)