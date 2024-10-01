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


#-----------------------------------------------------------------
#     Mustafa Rahman, Yang Liu
#     Copyright 2024
#     Lawrence Berkeley National Laboratory
#     All Rights Reserved
#-----------------------------------------------------------------

#     Revision 1.1  2024/08/20 15:30:00  mustafar
#     Original source.

#     FSFNO code: Fusion Simultion Fourier Neural Operator code 
#-----------------------------------------------------------------
 
def NIMROD_pde_operator_parameters_defination(if_2ndRunHyperDiffusivity_case, S):
    fieldlist_parm_lst = 'npvbjei'
    fieldlist_parm_vector_lst =[ 3,1,3,3,3,1,1] #Vector # Will consider a global array with Max. of this 
    if if_2ndRunHyperDiffusivity_case: 
        fieldlist_parm_eq_range=5   ## Will consider a global array with Max. of this 
    else:
        if S == 64: 
            fieldlist_parm_eq_range=4   ## Will consider a global array with Max. of this 
        else:
            fieldlist_parm_eq_range=4   ## Will consider a global array with Max. of this     
    fieldlist_parm_vector_chosen = [4,4,4,4,4] # Will consider a global array with Max. of this 
    fieldlist_parm_eq_vector_train_global_lst  =\
        [   
            [
                [
                    [   [ ['v',0,0],['v',0,1],['v',0,2],
                        ['n',0,1],['n',0,2],
                        ['e',0,0],['i',0,0],
                        ['b',0,0],['b',0,1],['b',0,2]                                                
                        ],    
                                                    [   ['v',0,0],['v',0,1],['v',0,2]
                                                        ,['n',0,1],['n',0,2]
                                                        ,['e',0,0],['i',0,0]
                                                        ,['b',0,0],['b',0,1],['b',0,2]                                                 
                                                    ] 
                    ]
                ]
            ]
        ]

    input_parameter_order = [[0,1,2,5,6,7,8,9],[3,4,0,1,2],[5,0,1,4,7,8,9],[6,0,1,2],[7,8,9,0,1,2]]
    mWidth_input_parameters  = [ 8, 5, 7, 4, 6]
    nWidth_output_parameters = [ 3, 2, 1, 1, 3]
    print(' paramters:',fieldlist_parm_lst)
    count = 1
    for i in range(0,len(mWidth_input_parameters)):  
        print("   ",count,':',count+nWidth_output_parameters[i]-1," - F_(",mWidth_input_parameters[i],',',nWidth_output_parameters[i],')')
        count = count + nWidth_output_parameters[i]
    return (fieldlist_parm_lst,
            fieldlist_parm_vector_lst, 
            fieldlist_parm_eq_range, 
            fieldlist_parm_vector_chosen, 
            fieldlist_parm_eq_vector_train_global_lst,
            input_parameter_order, 
            mWidth_input_parameters, 
            nWidth_output_parameters )