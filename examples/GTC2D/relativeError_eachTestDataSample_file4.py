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

from stfno.utilities3 import *

def  relativeErrorEachTestDataSample_file4(
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
        ):
    for ep in range(epochs,epochs+1):
        file4 = open(strn_epochs_dump_path_file4, "a")  
        myloss_MaxNormRelList = LpLoss_fieldElements(size_average=True)
        myloss_MaxNormRelList_2 = LpLoss_fieldElements_2_logRMS_RegressionModel(size_average=False)
        with torch.no_grad():
          count = -1
          for (i_testloader, (xx, yy)), (i_testloader_2_logRMS_RegressionModel, (xx_2_logRMS_RegressionModel, yy_2_logRMS_RegressionModel)) in zip(enumerate(test_loader), enumerate(test_loader_2_logRMS_RegressionModel )):
            if i_testloader == i_testloader_2_logRMS_RegressionModel:
                loss_MaxNormRelList = 0
                xx = xx.to(device)
                yy = yy.to(device)
                xx_2_logRMS_RegressionModel = xx_2_logRMS_RegressionModel.to(device)
                yy_2_logRMS_RegressionModel = yy_2_logRMS_RegressionModel.to(device)
                count= count +1 
                for t in range(0, T_out*sum_vector_u_elements_i_iter  , T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter ):
                    y = yy[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter)]
                    y_2_logRMS_RegressionModel = yy_2_logRMS_RegressionModel[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter)]
                    im = xx
                    im_2_logRMS_RegressionModel = xx_2_logRMS_RegressionModel
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    if t == 0:
                        pred_2_logRMS_RegressionModel = im_2_logRMS_RegressionModel
                    else:
                        pred_2_logRMS_RegressionModel = torch.cat((pred_2_logRMS_RegressionModel, im_2_logRMS_RegressionModel), -1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_tmp = xx[...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                    xx = torch.cat((xx[...,T_out_sub_time_consecutiveIterator_factor * step*sum_vector_u_elements_i_iter:], im), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx = torch.cat((xx, xx_tmp[:]), dim=-1)
                        exit(1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_tmp_2_logRMS_RegressionModel = xx_2_logRMS_RegressionModel [...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                    xx_2_logRMS_RegressionModel = torch.cat((xx_2_logRMS_RegressionModel[...,T_out_sub_time_consecutiveIterator_factor * step*sum_vector_u_elements_i_iter:], im_2_logRMS_RegressionModel), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_2_logRMS_RegressionModel = torch.cat((xx_2_logRMS_RegressionModel, xx_tmp_2_logRMS_RegressionModel[:]), dim=-1)
                        exit(1)
                    if t == 0:
                        loss_MaxNormRelList = myloss_MaxNormRelList (im, y)                
                    else:
                        loss_MaxNormRelList += myloss_MaxNormRelList (im, y)                
                    if t == ( T_out*sum_vector_u_elements_i_iter  - T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter):
                        for i_test_batch_sizes, test_batch_sizes_i in enumerate(range(y.size()[0])):
                            im_subtime = im[i_test_batch_sizes,:,:,: ].unsqueeze(0) 
                            y_subtime  =  y[i_test_batch_sizes,:,:,: ].unsqueeze(0) 
                            loss_MaxNormRelList_eachTimeStep = myloss_MaxNormRelList(im_subtime, y_subtime).item()                
                            im_subtime_exp_2_logRMS_RegressionModel = torch.exp(im_2_logRMS_RegressionModel[i_test_batch_sizes,: ]).unsqueeze(0)
                            y_subtime_exp_2_logRMS_RegressionModel  = torch.exp( y_2_logRMS_RegressionModel[i_test_batch_sizes,: ]).unsqueeze(0)
                            loss_MaxNormRelList_eachTimeStep_exp_2_logRMS_RegressionModel = myloss_MaxNormRelList_2(im_subtime_exp_2_logRMS_RegressionModel, y_subtime_exp_2_logRMS_RegressionModel ).item()
                            im_subtime_2_logRMS_RegressionModel = (im_2_logRMS_RegressionModel[i_test_batch_sizes,: ]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                            y_subtime_2_logRMS_RegressionModel  = ( y_2_logRMS_RegressionModel[i_test_batch_sizes,: ]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                            loss_MaxNormRelList_eachTimeStep_2_logRMS_RegressionModel = myloss_MaxNormRelList(im_subtime_2_logRMS_RegressionModel, y_subtime_2_logRMS_RegressionModel ).item()
                            im_subtime_2_logRMS_RegressionModel = (im_2_logRMS_RegressionModel[i_test_batch_sizes,: ])
                            y_subtime_2_logRMS_RegressionModel  = ( y_2_logRMS_RegressionModel[i_test_batch_sizes,: ])
                            im_subtime_2_logRMS_RegressionModel_multiplied = im_subtime*((im_subtime_2_logRMS_RegressionModel))
                            y_subtime_2_logRMS_RegressionModel_multiplied  =  y_subtime*((y_subtime_2_logRMS_RegressionModel))
                            loss_MaxNormRelList_eachTimeStep_Including_2_logRMS_RegressionModel = myloss_MaxNormRelList(im_subtime_2_logRMS_RegressionModel_multiplied, y_subtime_2_logRMS_RegressionModel_multiplied ).item()
                            str_file4= ( 
                            str(i_testloader*batch_size +i_test_batch_sizes)  
                            +',' +  str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out] ).zfill(5) 
                            +','+ str( loss_MaxNormRelList_eachTimeStep  )
                            +','+ str(T_out) 
                            +','+ str( ( (loss_MaxNormRelList.item()) * ((t+1)/ (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter)) / (T_out / step)) ) #.cpu().detach().numpy()
                            +','+ str(t//sum_vector_u_elements_i_iter)
                            +','+ str( (myloss_MaxNormRelList (im, y).item() ) /ntest ) #.cpu().detach().numpy() )       
                            +','+ str( (myloss_MaxNormRelList (im_subtime, y_subtime) ).item()) #.cpu().detach().numpy()  )       
                            +','+ str(i_testloader*batch_size +i_test_batch_sizes)
                            +','+ str( loss_MaxNormRelList_eachTimeStep_2_logRMS_RegressionModel)
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[0].item()) ) ) )
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[1].item()) ) ) )
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[2].item()) ) ) )
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[3].item()) ) ) )
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[4].item()) ) ) )
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[5].item()) ) ) )
                            +','+ str((np.exp((y_subtime_2_logRMS_RegressionModel[6].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[0].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[1].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[2].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[3].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[4].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[5].item()) ) ) )
                            +','+ str((np.exp((im_subtime_2_logRMS_RegressionModel[6].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[0].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[1].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[2].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[3].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[4].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[5].item()) ) ) )
                            +','+ str((((y_subtime_2_logRMS_RegressionModel[6].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[0].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[1].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[2].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[3].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[4].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[5].item()) ) ) )
                            +','+ str((((im_subtime_2_logRMS_RegressionModel[6].item()) ) ) )
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[0].item())-np.exp(im_subtime_2_logRMS_RegressionModel[0].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[0].item())) )  
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[1].item())-np.exp(im_subtime_2_logRMS_RegressionModel[1].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[1].item())) )  
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[2].item())-np.exp(im_subtime_2_logRMS_RegressionModel[2].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[2].item())) ) 
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[3].item())-np.exp(im_subtime_2_logRMS_RegressionModel[3].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[3].item())) ) 
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[4].item())-np.exp(im_subtime_2_logRMS_RegressionModel[4].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[4].item())) ) 
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[5].item())-np.exp(im_subtime_2_logRMS_RegressionModel[5].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[5].item())) ) 
                            +','+ str((abs(np.exp(y_subtime_2_logRMS_RegressionModel[6].item())-np.exp(im_subtime_2_logRMS_RegressionModel[6].item()) ) )/abs(np.exp(y_subtime_2_logRMS_RegressionModel[6].item())) ) 
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[0].item()-(im_subtime_2_logRMS_RegressionModel[0].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[0].item()) )  
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[1].item()-(im_subtime_2_logRMS_RegressionModel[1].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[1].item()) )  
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[2].item()-(im_subtime_2_logRMS_RegressionModel[2].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[2].item()) ) 
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[3].item()-(im_subtime_2_logRMS_RegressionModel[3].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[3].item()) )
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[4].item()-(im_subtime_2_logRMS_RegressionModel[4].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[4].item()) ) 
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[5].item()-(im_subtime_2_logRMS_RegressionModel[5].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[5].item()) ) 
                            +','+ str((abs(y_subtime_2_logRMS_RegressionModel[6].item()-(im_subtime_2_logRMS_RegressionModel[6].item()) ) )/abs(y_subtime_2_logRMS_RegressionModel[6].item()) ) 
                            +','+ str(torch.norm(torch.exp(y_subtime_2_logRMS_RegressionModel) ).item() )
                            +','+ str(torch.norm(torch.exp(im_subtime_2_logRMS_RegressionModel)).item())
                            +','+ str( loss_MaxNormRelList_eachTimeStep_exp_2_logRMS_RegressionModel)
                            +','+ str( loss_MaxNormRelList_eachTimeStep_Including_2_logRMS_RegressionModel)
                            + '\n' )
                            file4.write(str_file4) 
        file4.close()