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
from stfno.utilities3 import *
import os 
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize


def contourplotting(data_read_global,
        data_read_global_mean,data_read_global_std,
        data_read_global_eachTimeStep_mean,
        data_read_global_eachTimeStep_std,
        ntrain,
        r_theta_phi, 
        T_out,
        startofpatternlist_i_file_no_in_SelectData,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        epochs,
        T_out_sub_time_consecutiveIterator_factor, step,
        batch_size,
        i_file_no_in_SelectData, 
        if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot,
        OneByPowerTransformationFactorOfData,
        log_param,
        nlvls,
        epsilon_inPlottingErrorNormalization,
        model,
        test_loader
        ):
    for ep in range(epochs,epochs+1):
        with torch.no_grad():
            count = -1
            for i_testloader,(xx, yy) in enumerate(test_loader):
                xx = xx.to(device)
                yy = yy.to(device)
                count= count +1 
                for t in range(0, T_out*sum_vector_u_elements_i_iter  , T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter ):
                    y = yy[..., t:t + (T_out_sub_time_consecutiveIterator_factor *sum_vector_u_elements_i_iter)]
                    im = model(xx)
                    for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                        fieldlist_parm_lst_i = fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]
                        fieldlist_parm_eq_selected = fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                        fieldlist_parm_vector_i = fieldlist_parm_eq_vector_train_global_lst_i_j_k[2] 
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx_tmp = xx[...,(T_out * step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter)) +1:]
                    xx = torch.cat((xx[...,T_out_sub_time_consecutiveIterator_factor * step*sum_vector_u_elements_i_iter:], im), dim=-1)
                    if step*(sum_vector_u_elements_i_iter - sum_vector_a_elements_i_iter +1) > step:
                        xx = torch.cat((xx, xx_tmp[:]), dim=-1)
                        exit(1)    
                    if ep == (epochs): 
                        for i_test_batch_sizes, test_batch_sizes_i in enumerate(range(y.size()[0])):
                          item_of_sum_vector_test_elements_i = -1
                          for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                            item_of_sum_vector_test_elements_i += 1    
                            fieldlist_parm_lst_i       = fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]
                            fieldlist_parm_eq_selected = fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                            fieldlist_parm_vector_i    = fieldlist_parm_eq_vector_train_global_lst_i_j_k[2] 
                            if fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'n':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 0
                            elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'e':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 1
                            elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'v':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 2
                            elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'b':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 3
                            elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'i':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 4
                            elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'p':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 5                                    
                            elif fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] == 'j':
                                fieldlist_parm_eq_vector_train_global_lst_i_j_0 = 6                                            
                            else:
                                exit(1)
                            if (  ( (i_testloader*batch_size +i_test_batch_sizes)!=4 )    # 73568
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=118 )# 70057
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=41 ) # 65990
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=24 ) # 62046
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=93 ) # 58116
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=62 ) # 53954
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=17 ) # 49192
                                and ( (i_testloader*batch_size +i_test_batch_sizes)!=16 ) # 45970   
                                ):
                                continue
                                # pass
                            # if (  ( (i_testloader*batch_size +i_test_batch_sizes)!=-1 )    # 73568   
                            #     ):
                            #     continue
                            #     pass
                            if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst))):
                                os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)))
                            if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i)):
                                os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i))
                            if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected))):
                                os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)))
                            if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i))):
                                os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)))
                            if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes))):
                                os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)))
                            if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/')):
                                os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/'))
                            if not os.path.exists(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst))):
                                os.makedirs(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)))
                            if not os.path.exists(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i)):
                                os.makedirs(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i))
                            if not os.path.exists(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected))):
                                os.makedirs(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)))
                            if not os.path.exists(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i))):
                                os.makedirs(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)))
                            if not os.path.exists(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes))):
                                os.makedirs(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)))
                            if not os.path.exists(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/')):
                                os.makedirs(('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/'))
                            if not os.path.exists(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst))):
                                os.makedirs(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)))
                            if not os.path.exists(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i)):
                                os.makedirs(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i))
                            if not os.path.exists(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected))):
                                os.makedirs(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)))
                            if not os.path.exists(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i))):
                                os.makedirs(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)))
                            if not os.path.exists(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes))):
                                os.makedirs(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)))
                            if not os.path.exists(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/')):
                                os.makedirs(('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/'))
                            norm = None
                            for itr_T_out_By_T_out_sub_time_consecutiveIterator_factor, T_out_By_T_out_sub_time_consecutiveIterator_factor_i in enumerate( range(T_out_sub_time_consecutiveIterator_factor) ):
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    x_cordinates_pmeshplot = data_read_global[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ),
                                                                    7 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
                                    y_cordinates_pmeshplot = data_read_global[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ),
                                                                    8 #fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2],:,:]
                                    x_cordinates_pmeshplot_periodic = x_cordinates_pmeshplot # np.hstack( (x_cordinates_pmeshplot,x_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
                                    y_cordinates_pmeshplot_periodic = y_cordinates_pmeshplot #np.hstack( (y_cordinates_pmeshplot,y_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
                                else:
                                    x_cordinates_pmeshplot = r_theta_phi[0,:,:]
                                    y_cordinates_pmeshplot = r_theta_phi[1,:,:]  
                                    x_cordinates_pmeshplot_periodic = np.hstack( (x_cordinates_pmeshplot,x_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
                                    y_cordinates_pmeshplot_periodic = np.hstack( (y_cordinates_pmeshplot,y_cordinates_pmeshplot[:,0].reshape(-1, 1)  ) )
                                x_cordinates_pmeshplot_periodic = np.power(x_cordinates_pmeshplot_periodic,OneByPowerTransformationFactorOfData )
                                y_cordinates_pmeshplot_periodic = np.power(y_cordinates_pmeshplot_periodic,OneByPowerTransformationFactorOfData )
                                fig = plt.figure(figsize=(3, 3))
                                ax = fig.add_subplot(111)
                                Vi = np.array (im.cpu().numpy()[i_test_batch_sizes ,:,:,item_of_sum_vector_test_elements_i] )                  
                                Vi  = (Vi* data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                field_xy =  Vi 
                                fmin = np.min(field_xy)
                                fmax = np.max(field_xy)
                                norm = None
                                if (fmin is not None and fmax is not None):
                                            if log_param and fmin>0.:
                                                levels = np.logspace(numpy.log10(fmin), numpy.log10(fmax), nlvls)
                                                norm = colors.LogNorm()
                                            else:
                                                levels = np.linspace(fmin, fmax, nlvls)
                                else:
                                            levels = nlvls
                                levels = np.linspace(fmin, fmax, nlvls)
                                grid_x, grid_y = np.meshgrid(np.arange(0, field_xy.shape[1]), np.arange(0, field_xy.shape[0]))
                                cf = ax.contourf(grid_x, grid_y, field_xy, levels, norm=norm,cmap=cm.gist_rainbow_r)
                                last_axes = plt.gca()
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="6%", pad=0.04)
                                fig.colorbar(cf, cax=cax, orientation='vertical')
                                plt.sca(last_axes)
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_lst_i+'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_selected) +'_'  +str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5)+ "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_pred'
                                ax.set_title(filename_str, fontsize=12, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 1 as .png:',strpicpng)
                                print('   saving figure count 1 as .pdf:',strpicpdf)
                                fig.savefig(strpicpng, bbox_inches='tight')
                                fig.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig)
                                fig0 = plt.figure(figsize=(3, 3))
                                ax0 = fig0.add_subplot(111)
                                Vi = np.array (im.cpu().numpy()[i_test_batch_sizes ,:,:,item_of_sum_vector_test_elements_i] )                  
                                Vi  = (np.power(Vi,OneByPowerTransformationFactorOfData) * data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_eachTimeStep_mean[(startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ),fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                Vi = np.array (im.cpu().numpy()[i_test_batch_sizes ,:,:,item_of_sum_vector_test_elements_i] )                  
                                Vi  = (Vi* data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                c = ax0.pcolormesh((x_cordinates_pmeshplot_periodic),( y_cordinates_pmeshplot_periodic), (field_xy__periodic), cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                fig0.colorbar(c, ax=ax0) 
                                ax0.set_xlabel('x',fontsize=10)
                                ax0.set_ylabel('y',fontsize=10)
                                filepath_str= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_lst_i+'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_selected) +'_'  +str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5)+ "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter) + '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_pred'
                                ax0.set_title(filename_str, fontsize=6, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 2 as .png:',strpicpng)
                                print('   saving figure count 2 as .pdf:',strpicpdf)
                                fig0.savefig(strpicpng, bbox_inches='tight')
                                fig0.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig0)
                                fig2 = plt.figure(figsize=(3, 3))
                                ax2 = fig2.add_subplot(111)
                                Vi = np.array (im.cpu().numpy()[i_test_batch_sizes ,:,:,item_of_sum_vector_test_elements_i] )                  
                                Vi  = (Vi* data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                norm1 = Normalize(vmin=-1, vmax=1)
                                c = ax2.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap='RdBu', shading='gouraud')#cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                cbar=fig2.colorbar(c, ax=ax2) 
                                formatter = ScalarFormatter(useMathText=True)
                                formatter.set_scientific(True)
                                formatter.set_powerlimits((0, 0))  # Adjust the range for scientific notation
                                cbar.ax.yaxis.set_major_formatter(formatter)
                                ax2.set_xticklabels([])
                                ax2.set_yticklabels([])
                                ax2.xaxis.set_ticks([])
                                ax2.yaxis.set_ticks([])
                                ax2.grid(False)
                                ax2.spines['top'].set_visible(False)
                                ax2.spines['right'].set_visible(False)
                                ax2.spines['bottom'].set_visible(False)
                                ax2.spines['left'].set_visible(False) 
                                filepath_str= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_lst_i+'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_selected) +'_'  +str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5)+ "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_pred'
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 3 as .png:',strpicpng)
                                print('   saving figure count 3 as .pdf:',strpicpdf)
                                fig2.savefig(strpicpng, bbox_inches='tight')
                                fig2.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig2)
                                fig = plt.figure(figsize=(3, 3))
                                ax = fig.add_subplot(111)
                                Vi = np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )
                                Vi  = (np.power(Vi,OneByPowerTransformationFactorOfData)* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_eachTimeStep_mean[(startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ),fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                Vi = np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] )
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                field_xy =  Vi 
                                fmin = np.min(field_xy)
                                fmax = np.max(field_xy)
                                norm = None
                                if (fmin is not None and fmax is not None):
                                            if log_param and fmin>0.:
                                                levels = np.logspace(np.log10(fmin), np.log10(fmax), nlvls)
                                                norm = colors.LogNorm()
                                            else:
                                                levels = np.linspace(fmin, fmax, nlvls)
                                else:
                                            levels = nlvls
                                levels = np.linspace(fmin, fmax, nlvls)
                                grid_x, grid_y = np.meshgrid(np.arange(0, field_xy.shape[1]), np.arange(0, field_xy.shape[0]))
                                cf = ax.contourf(grid_x, grid_y, field_xy, levels, norm=norm,cmap=cm.gist_rainbow_r)
                                last_axes = plt.gca()
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="6%", pad=0.04)
                                fig.colorbar(cf, cax=cax, orientation='vertical')
                                plt.sca(last_axes)
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_trut'
                                ax.set_title(filename_str, fontsize=12, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 4 as .png:',strpicpng)
                                print('   saving figure count 4 as .pdf:',strpicpdf)
                                fig.savefig(strpicpng, bbox_inches='tight')
                                fig.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig)
                                fig0 = plt.figure(figsize=(3, 3))
                                ax0 = fig0.add_subplot(111)
                                Vi = np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )
                                Vi  = (np.power(Vi,OneByPowerTransformationFactorOfData)* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_eachTimeStep_mean[(startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ),fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                Vi = np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] )
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                c = ax0.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                fig0.colorbar(c, ax=ax0) 
                                filepath_str= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_trut'
                                ax0.set_title(filename_str, fontsize=6, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 5 as .png:',strpicpng)
                                print('   saving figure count 5 as .pdf:',strpicpdf)
                                fig0.savefig(strpicpng, bbox_inches='tight')
                                fig0.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig0)
                                fig2 = plt.figure(figsize=(3, 3))
                                ax2 = fig2.add_subplot(111)
                                Vi = np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )
                                Vi  = (np.power(Vi,OneByPowerTransformationFactorOfData)* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_eachTimeStep_mean[(startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ),fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                Vi = np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] )
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                norm1 = Normalize(vmin=-1, vmax=1)
                                c = ax2.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap='RdBu', shading='gouraud')#cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                cbar=fig2.colorbar(c, ax=ax2) 
                                formatter = ScalarFormatter(useMathText=True)
                                formatter.set_scientific(True)
                                formatter.set_powerlimits((0, 0))  # Adjust the range for scientific notation
                                cbar.ax.yaxis.set_major_formatter(formatter)
                                ax2.set_xticklabels([])
                                ax2.set_yticklabels([])
                                ax2.xaxis.set_ticks([])
                                ax2.yaxis.set_ticks([])
                                ax2.grid(False)
                                ax2.spines['top'].set_visible(False)
                                ax2.spines['right'].set_visible(False)
                                ax2.spines['bottom'].set_visible(False)
                                ax2.spines['left'].set_visible(False) 
                                filepath_str= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_trut'
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 6 as .png:',strpicpng)
                                print('   saving figure count 6 as .pdf:',strpicpdf)
                                fig2.savefig(strpicpng, bbox_inches='tight')
                                fig2.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig2)
                                fig = plt.figure(figsize=(3, 3))
                                ax = fig.add_subplot(111)
                                Vi =  np.power( np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )  ,OneByPowerTransformationFactorOfData)- np.power(np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] ) ,OneByPowerTransformationFactorOfData)
                                Vi  = (Vi* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy())
                                Vi =  np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] ) 
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) # + - data_dump_mean.cpu().detach().numpy()
                                field_xy =  Vi 
                                fmin = np.min(field_xy)
                                fmax = np.max(field_xy)
                                norm = None
                                if (fmin is not None and fmax is not None):
                                            if log_param and fmin>0.:
                                                levels = np.logspace(np.log10(fmin), np.log10(fmax), nlvls)
                                                norm = colors.LogNorm()
                                            else:
                                                levels = np.linspace(fmin, fmax, nlvls)
                                else:
                                            levels = nlvls
                                levels = np.linspace(fmin, fmax, nlvls)
                                grid_x, grid_y = np.meshgrid(np.arange(0, field_xy.shape[1]), np.arange(0, field_xy.shape[0]))
                                cf = ax.contourf(grid_x, grid_y, field_xy, levels, norm=norm,cmap=cm.gist_rainbow_r)
                                last_axes = plt.gca()
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="6%", pad=0.04)
                                fig.colorbar(cf, cax=cax, orientation='vertical')
                                plt.sca(last_axes)
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_eror'
                                ax.set_title(filename_str, fontsize=12, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 7 as .png:',strpicpng)
                                print('   saving figure count 7 as .pdf:',strpicpdf)
                                fig.savefig(strpicpng, bbox_inches='tight')
                                fig.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig)
                                fig0 = plt.figure(figsize=(3, 3))
                                ax0 = fig0.add_subplot(111)
                                Vi =  np.power( np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )  ,OneByPowerTransformationFactorOfData)- np.power(np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] ) ,OneByPowerTransformationFactorOfData)
                                Vi  = (Vi* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) 
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                c = ax0.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                fig0.colorbar(c, ax=ax0) 
                                filepath_str= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_eror'
                                ax0.set_title(filename_str, fontsize=6, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 8 as .png:',strpicpng)
                                print('   saving figure count 8 as .pdf:',strpicpdf)
                                fig0.savefig(strpicpng, bbox_inches='tight')
                                fig0.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig0)
                                fig2 = plt.figure(figsize=(3, 3))
                                ax2 = fig2.add_subplot(111)
                                Vi =  np.power( np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )  ,OneByPowerTransformationFactorOfData)- np.power(np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] ) ,OneByPowerTransformationFactorOfData)
                                Vi  = (Vi* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) 
                                Vi =  np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] ) 
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) # + - data_dump_mean.cpu().detach().numpy()
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                norm1 = Normalize(vmin=-1, vmax=1)
                                c = ax2.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap='RdBu', shading='gouraud')#cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                cbar=fig2.colorbar(c, ax=ax2) 
                                formatter = ScalarFormatter(useMathText=True)
                                formatter.set_scientific(True)
                                formatter.set_powerlimits((0, 0))  # Adjust the range for scientific notation
                                cbar.ax.yaxis.set_major_formatter(formatter)
                                ax2.set_xticklabels([])
                                ax2.set_yticklabels([])
                                ax2.xaxis.set_ticks([])
                                ax2.yaxis.set_ticks([])
                                ax2.grid(False)
                                ax2.spines['top'].set_visible(False)
                                ax2.spines['right'].set_visible(False)
                                ax2.spines['bottom'].set_visible(False)
                                ax2.spines['left'].set_visible(False)
                                filepath_str= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_eror'
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 9 as .png:',strpicpng)
                                print('   saving figure count 9 as .pdf:',strpicpdf)
                                fig2.savefig(strpicpng, bbox_inches='tight')
                                fig2.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig2) 
                                fig = plt.figure(figsize=(3, 3))
                                ax = fig.add_subplot(111)
                                Vi =    (
                                            (
                                                (     
                                                    np.array 
                                                    (
                                                        im.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                    ) 
                                                    - 
                                                    np.array 
                                                    (
                                                        y.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                    )  
                                                )
                                                *
                                                data_read_global_std
                                                [
                                                    fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                ].cpu().detach().numpy()
                                            )
                                            /
                                            (
                                                np.max
                                                (
                                                    np.array 
                                                    (
                                                        y.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                        *
                                                        data_read_global_std
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy()
                                                    )                                                                      
                                                    +
                                                    (
                                                        data_read_global_mean
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy() 
                                                    )
                                                )                                                                
                                            )
                                        )
                                field_xy =  Vi 
                                fmin = np.min(field_xy)
                                fmax = np.max(field_xy)
                                norm = None
                                if (fmin is not None and fmax is not None):
                                            if log_param and fmin>0.:
                                                levels = np.logspace(np.log10(fmin), np.log10(fmax), nlvls)
                                                norm = colors.LogNorm()
                                            else:
                                                levels = np.linspace(fmin, fmax, nlvls)
                                else:
                                            levels = nlvls
                                levels = np.linspace(fmin, fmax, nlvls)
                                grid_x, grid_y = np.meshgrid(np.arange(0, field_xy.shape[1]), np.arange(0, field_xy.shape[0]))
                                cf = ax.contourf(grid_x, grid_y, field_xy, levels, norm=norm,cmap=cm.gist_rainbow_r)
                                last_axes = plt.gca()
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="6%", pad=0.04)
                                fig.colorbar(cf, cax=cax, orientation='vertical')
                                plt.sca(last_axes)
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_MxNm'
                                ax.set_title(filename_str, fontsize=12, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 10 as .png:',strpicpng)
                                print('   saving figure count 10 as .pdf:',strpicpdf)
                                fig.savefig(strpicpng, bbox_inches='tight')
                                fig.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig)
                                fig0 = plt.figure(figsize=(3, 3))
                                ax0 = fig0.add_subplot(111)
                                Vi =    (
                                            (
                                                (     
                                                    np.array 
                                                    (
                                                        im.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                    ) 
                                                    - 
                                                    np.array 
                                                    (
                                                        y.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                    )  
                                                )
                                                *
                                                data_read_global_std
                                                [
                                                    fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                ].cpu().detach().numpy()
                                            )
                                            /
                                            (
                                                np.max
                                                (
                                                    np.array 
                                                    (
                                                        y.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                        *
                                                        data_read_global_std
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy()
                                                    )                                                                      
                                                    +
                                                    (
                                                        data_read_global_mean
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy() 
                                                    )
                                                )                                                                
                                            )
                                        )
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                c = ax0.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                fig0.colorbar(c, ax=ax0) 
                                filepath_str= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_MxNm'
                                ax0.set_title(filename_str, fontsize=6, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 11 as .png:',strpicpng)
                                print('   saving figure count 11 as .pdf:',strpicpdf)
                                fig0.savefig(strpicpng, bbox_inches='tight')
                                fig0.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig0)
                                fig2 = plt.figure(figsize=(3, 3))
                                ax2 = fig2.add_subplot(111)
                                Vi =    (
                                            (
                                                (     
                                                    np.array 
                                                    (
                                                        im.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                    ) 
                                                    - 
                                                    np.array 
                                                    (
                                                        y.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                    )  
                                                )
                                                *
                                                data_read_global_std
                                                [
                                                    fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                    ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                ].cpu().detach().numpy()
                                            )
                                            /
                                            (
                                                np.max
                                                (
                                                    np.array 
                                                    (
                                                        y.cpu().numpy()
                                                        [
                                                            i_test_batch_sizes
                                                            ,:
                                                            ,:
                                                            ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                        * sum_vector_u_elements_i_iter 
                                                                        + item_of_sum_vector_test_elements_i
                                                        ]
                                                        *
                                                        data_read_global_std
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy()
                                                    )                                                                      
                                                    +
                                                    (
                                                        data_read_global_mean
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy() 
                                                    )
                                                )                                                                
                                            )
                                        )
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                norm1 = Normalize(vmin=-1, vmax=1)
                                max_field_xy__periodic = np.max( np.abs( field_xy__periodic ) )
                                min_field_xy__periodic = -max_field_xy__periodic
                                c = ax2.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap='RdBu', shading='gouraud', vmin = min_field_xy__periodic, vmax = max_field_xy__periodic)
                                cbar=fig2.colorbar(c, ax=ax2) 
                                formatter = ScalarFormatter(useMathText=True)
                                formatter.set_scientific(True)
                                formatter.set_powerlimits((0, 0))  # Adjust the range for scientific notation
                                cbar.ax.yaxis.set_major_formatter(formatter)
                                ax2.set_xticklabels([])
                                ax2.set_yticklabels([])
                                ax2.xaxis.set_ticks([])
                                ax2.yaxis.set_ticks([])
                                ax2.grid(False)
                                ax2.spines['top'].set_visible(False)
                                ax2.spines['right'].set_visible(False)
                                ax2.spines['bottom'].set_visible(False)
                                ax2.spines['left'].set_visible(False)
                                filepath_str= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_MxNm'
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 12 as .png:',strpicpng)
                                print('   saving figure count 12 as .pdf:',strpicpdf)
                                fig2.savefig(strpicpng, bbox_inches='tight')
                                fig2.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig2)
                                y_tmp_zero_divisor =   np.copy(np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] ) )
                                zero_indices = np.where( (abs(y_tmp_zero_divisor *  data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() )  + ( data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() )) ==0)
                                y_tmp_zero_divisor[zero_indices] = y_tmp_zero_divisor[zero_indices] + epsilon_inPlottingErrorNormalization
                                fig = plt.figure(figsize=(3, 3))
                                ax = fig.add_subplot(111)
                                Vi =  (     np.array (im.cpu().numpy()[0,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] )  )/  ( (np.max(abs(y_tmp_zero_divisor ))                                                                                                   ) + ( data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() / data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() ) )
                                Vi =    (
                                            (     
                                                np.array 
                                                (
                                                    im.cpu().numpy()
                                                    [
                                                        i_test_batch_sizes
                                                        ,:
                                                        ,:
                                                        ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                    * sum_vector_u_elements_i_iter 
                                                                    + item_of_sum_vector_test_elements_i
                                                    ]
                                                ) 
                                                - 
                                                np.array 
                                                (
                                                    y.cpu().numpy()
                                                    [
                                                        i_test_batch_sizes,:,:
                                                        ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                    * sum_vector_u_elements_i_iter 
                                                                    + item_of_sum_vector_test_elements_i
                                                    ]
                                                )  
                                            )
                                            /
                                            (
                                                np.abs # np.max
                                                (
                                                    (
                                                        y_tmp_zero_divisor
                                                    )  
                                                    +
                                                    (
                                                        data_read_global_mean
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy() 
                                                        /
                                                        data_read_global_std
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy()
                                                    )
                                                )                                                                
                                            )
                                        )
                                field_xy =  Vi 
                                fmin = np.min(field_xy)
                                fmax = np.max(field_xy)
                                norm = None
                                if (fmin is not None and fmax is not None):
                                            if log_param and fmin>0.:
                                                levels = np.logspace(np.log10(fmin), np.log10(fmax), nlvls)
                                                norm = colors.LogNorm()
                                            else:
                                                levels = np.linspace(fmin, fmax, nlvls)
                                else:
                                            levels = nlvls
                                levels = np.linspace(fmin, fmax, nlvls)
                                grid_x, grid_y = np.meshgrid(np.arange(0, field_xy.shape[1]), np.arange(0, field_xy.shape[0]))
                                cf = ax.contourf(grid_x, grid_y, field_xy, levels, norm=norm,cmap=cm.gist_rainbow_r)
                                last_axes = plt.gca()
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="6%", pad=0.04)
                                fig.colorbar(cf, cax=cax, orientation='vertical')
                                plt.sca(last_axes)
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_nrml'
                                ax.set_title(filename_str, fontsize=12, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 13 as .png:',strpicpng)
                                print('   saving figure count 13 as .pdf:',strpicpdf)
                                fig.savefig(strpicpng, bbox_inches='tight')
                                fig.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig)
                                fig0 = plt.figure(figsize=(3, 3))
                                ax0 = fig0.add_subplot(111)
                                Vi   =  ( (   np.power( np.array( (im.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )),OneByPowerTransformationFactorOfData) - np.power( np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] ),OneByPowerTransformationFactorOfData) )* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) / ( ( (np.max(abs( np.power(np.array (y_tmp_zero_divisor                                                                                                                                               ),OneByPowerTransformationFactorOfData) *data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()          ) + (data_read_global_eachTimeStep_mean[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()  ) ) )))  
                                Vi =  (     np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] )  )/  ( (np.max(abs(y_tmp_zero_divisor[:,:]))                                                                                                   ) + ( data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() / data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() ) )
                                Vi =    (
                                            (     
                                                np.array 
                                                (
                                                    im.cpu().numpy()
                                                    [
                                                        i_test_batch_sizes
                                                        ,:
                                                        ,:
                                                        ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                    * sum_vector_u_elements_i_iter 
                                                                    + item_of_sum_vector_test_elements_i
                                                    ]
                                                ) 
                                                - 
                                                np.array 
                                                (
                                                    y.cpu().numpy()
                                                    [
                                                        i_test_batch_sizes,:,:
                                                        ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                    * sum_vector_u_elements_i_iter 
                                                                    + item_of_sum_vector_test_elements_i
                                                    ]
                                                )  
                                            )
                                            /
                                            (
                                                np.abs# np.max
                                                (
                                                    (
                                                        y_tmp_zero_divisor
                                                    )  
                                                    +
                                                    (
                                                        data_read_global_mean
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy() 
                                                        /
                                                        data_read_global_std
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy()
                                                    )
                                                )                                                                
                                            )
                                        )
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                c = ax0.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                fig0.colorbar(c, ax=ax0) 
                                filepath_str= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_nrml'
                                ax0.set_title(filename_str, fontsize=6, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 14 as .png:',strpicpng)
                                print('   saving figure count 14 as .pdf:',strpicpdf)
                                fig0.savefig(strpicpng, bbox_inches='tight')
                                fig0.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig0)
                                fig2 = plt.figure(figsize=(3, 3))
                                ax2 = fig2.add_subplot(111)
                                Vi   =  ( (   np.power( np.array( (im.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] )),OneByPowerTransformationFactorOfData) - np.power( np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,T_out_By_T_out_sub_time_consecutiveIterator_factor_i * sum_vector_u_elements_i_iter + item_of_sum_vector_test_elements_i] ),OneByPowerTransformationFactorOfData) )* data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) / ( ( (np.max(abs( np.power(np.array (y_tmp_zero_divisor                                                                                                                                               ),OneByPowerTransformationFactorOfData) *data_read_global_eachTimeStep_std[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()          ) + (data_read_global_eachTimeStep_mean[ (startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i ), fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()  ) ) )))  
                                Vi =  (     np.array (im.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[i_test_batch_sizes,:,:,item_of_sum_vector_test_elements_i] )  )/  ( (np.max(abs(y_tmp_zero_divisor ))                                                                                                   ) + ( data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() / data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() ) )
                                Vi =    (
                                            (     
                                                np.array 
                                                (
                                                    im.cpu().numpy()
                                                    [
                                                        i_test_batch_sizes
                                                        ,:
                                                        ,:
                                                        ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                    * sum_vector_u_elements_i_iter 
                                                                    + item_of_sum_vector_test_elements_i
                                                    ]
                                                ) 
                                                - 
                                                np.array 
                                                (
                                                    y.cpu().numpy()
                                                    [
                                                        i_test_batch_sizes,:,:
                                                        ,T_out_By_T_out_sub_time_consecutiveIterator_factor_i 
                                                                    * sum_vector_u_elements_i_iter 
                                                                    + item_of_sum_vector_test_elements_i
                                                    ]
                                                )  
                                            )
                                            /
                                            (
                                                np.abs
                                                (
                                                    (
                                                        y_tmp_zero_divisor
                                                    )  
                                                    +
                                                    (
                                                        data_read_global_mean
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy() 
                                                        /
                                                        data_read_global_std
                                                        [
                                                            fieldlist_parm_eq_vector_train_global_lst_i_j_0
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                                            ,fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]
                                                        ].cpu().detach().numpy()
                                                    )
                                                )                                                                
                                            )
                                        )
                                field_xy =  Vi 
                                if if_GTCLinearNonLinear_case_xy_cordinates_pmeshplot:                                                
                                    field_xy__periodic = field_xy
                                else:
                                    field_xy__periodic = np.hstack( (field_xy,field_xy[:,0].reshape(-1, 1)) )
                                norm1 = Normalize(vmin=-1, vmax=1)
                                c = ax2.pcolormesh(x_cordinates_pmeshplot_periodic, y_cordinates_pmeshplot_periodic, field_xy__periodic, cmap='RdBu', shading='gouraud')#cmap=cm.gist_rainbow_r , shading='gouraud')#cmap='RdBu', shading='gouraud')
                                cbar=fig2.colorbar(c, ax=ax2) 
                                formatter = ScalarFormatter(useMathText=True)
                                formatter.set_scientific(True)
                                formatter.set_powerlimits((0, 0))  # Adjust the range for scientific notation
                                cbar.ax.yaxis.set_major_formatter(formatter)
                                ax2.set_xticklabels([])
                                ax2.set_yticklabels([])
                                ax2.xaxis.set_ticks([])
                                ax2.yaxis.set_ticks([])
                                ax2.grid(False)
                                ax2.spines['top'].set_visible(False)
                                ax2.spines['right'].set_visible(False)
                                ax2.spines['bottom'].set_visible(False)
                                ax2.spines['left'].set_visible(False)
                                filepath_str= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/' )
                                filepath_str_pdf= ('./plots_pmesh_RdBu/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(i_testloader*batch_size +i_test_batch_sizes)+'/pdf/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(i_testloader*batch_size +i_test_batch_sizes)+ntrain]+t//sum_vector_u_elements_i_iter + T_out_By_T_out_sub_time_consecutiveIterator_factor_i] ).zfill(5) + "__ntst"+str(i_testloader*batch_size +i_test_batch_sizes)+"_ep"+ str(ep) + '_T' + str(t//sum_vector_u_elements_i_iter)+ '_Tsub'+str(T_out_By_T_out_sub_time_consecutiveIterator_factor_i)+'_nrml'
                                ax2.set_title(filename_str, fontsize=6, loc='left')
                                strpicpng= filepath_str+filename_str+'.png'
                                strpicpdf= filepath_str_pdf+filename_str+'.pdf'
                                print('   saving figure count 15 as .png:',strpicpng)
                                print('   saving figure count 15 as .pdf:',strpicpdf)
                                fig2.savefig(strpicpng, bbox_inches='tight')
                                fig2.savefig(strpicpdf, bbox_inches='tight')
                                plt.close(fig2)
                                plt.close('all')
                                if count ==1:   
                                    continue # Dummy Condition  to just run the loop for number of count. Can comment this continue to run loop for all the paramters.
                                    pass
                plt.close('all')