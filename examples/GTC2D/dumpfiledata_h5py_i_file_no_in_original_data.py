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

import torch
import h5py
import numpy as np

def dumpfiledata_h5py_i_file_no(fieldlist_parm_lst,fieldlist_parm_eq_range,fieldlist_parm_vector_lst,S,i_file_no_in,r_theta_phi, path):
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    dir_name = "data/"
    test_files = os.listdir(dir_name)
    for item in test_files:
        if item.endswith(".hdf5"):
            os.remove(os.path.join(dir_name, item))
    ret_0 = parse_snapshot(path+'snap'+(i_file_no_in[0])+'.out')
    ret_minus1 = parse_snapshot(path+'snap'+(i_file_no_in[-1])+'.out')
    for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst):
                    for fieldlist_parm_eq_selected in range(fieldlist_parm_eq_range):
                        for fieldlist_parm_vector_i in range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst]):
                            strn_dump_path = 'data/'+fieldlist_parm_lst_i+'_eq' + str(fieldlist_parm_eq_selected)+'_vec' +str(fieldlist_parm_vector_i) +'.hdf5'
                            hdf = h5py.File(strn_dump_path, "a")
                            dset = hdf.create_dataset(
                                name  = fieldlist_parm_lst_i, # "my_data",
                                shape = (0,ret_0[fieldlist_parm_lst_i].shape[0],ret_0[fieldlist_parm_lst_i].shape[1]),
                                maxshape = (None, ret_0[fieldlist_parm_lst_i].shape[0],ret_0[fieldlist_parm_lst_i].shape[1]), # None means that this dimension can be extended
                                dtype = "float32"
                                )
                            hdf.close()
                            if ret_0[fieldlist_parm_lst_i].shape !=  ret_minus1[fieldlist_parm_lst_i].shape:
                                exit(1)
    for i_i_file_no_in, i_file_no_in_i in enumerate(i_file_no_in):
        ret_i = parse_snapshot(path+'snap'+i_file_no_in_i+'.out')
        for i_fieldlist_parm_lst, fieldlist_parm_lst_i in enumerate(fieldlist_parm_lst):
                        eval_field = ret_i[fieldlist_parm_lst_i]
                        if np.any(np.isnan(eval_field)):
                            exit(1)
                        data_dump_tmp = eval_field
                        for fieldlist_parm_eq_selected in range(fieldlist_parm_eq_range):
                            for fieldlist_parm_vector_i in range(fieldlist_parm_vector_lst[i_fieldlist_parm_lst]):
                                strn_dump_path = 'data/'+fieldlist_parm_lst_i+'_eq' + str(fieldlist_parm_eq_selected)+'_vec' +str(fieldlist_parm_vector_i) +'.hdf5'
                                hdf = h5py.File(strn_dump_path, "a")
                                dset = hdf[fieldlist_parm_lst_i]
                                dset.resize(dset.shape[0]+1, axis=0)
                                dset[-1:] = data_dump_tmp
                                hdf.close()
    print('Dumped the data. Now rerun with if_dumpfiledata = False instead of if_dumpfiledata = True')
    exit(1)