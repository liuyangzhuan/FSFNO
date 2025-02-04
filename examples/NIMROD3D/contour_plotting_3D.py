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
import numpy
import numpy as np


def colorbar123(ax, fig, labels=None, units=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        last_axes = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.04)
        cbar = plt.colorbar(fig, cax=cax)#, format=fmt)
        plt.sca(last_axes)
        if labels is not None:
            cbar.set_ticks(labels)
            cbar.set_ticklabels(['{:.0e}'.format(it) for it in labels])
        if units is not None:
            cbar.ax.set_title(units)
        return cbar
    
def plot_3D(domain, field ,
                n_phi_plot_count,
                n_phi_plot_begin_factor,
                nx_r, nx_theta,
                nx_phi,
                filepath_str, filename_str):    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    rzPhi_domain_crossSection = numpy.zeros((3, nx_r, nx_theta,n_phi_plot_count))
    field_crossSection  = numpy.zeros((nx_r, nx_theta,n_phi_plot_count))
    for i_phi in range(0,n_phi_plot_count):
        rzPhi_domain_crossSection[ :,:,:,i_phi] = domain[ :,:,:,int(((i_phi*nx_phi)//n_phi_plot_count) + n_phi_plot_begin_factor)]
        field_crossSection[ :,:,i_phi] = field[ :,:,int(((i_phi*nx_phi)//n_phi_plot_count) + n_phi_plot_begin_factor)]
    xyz_coord = numpy.zeros(numpy.shape(rzPhi_domain_crossSection))
    xyz_coord[0] = rzPhi_domain_crossSection[0] * numpy.cos(rzPhi_domain_crossSection[2])
    xyz_coord[1] = rzPhi_domain_crossSection[0] * numpy.sin(rzPhi_domain_crossSection[2])
    xyz_coord[2] = rzPhi_domain_crossSection[1]
    domain_0 = xyz_coord[0][~numpy.isnan(field_crossSection)]
    domain_1 = xyz_coord[1][~numpy.isnan(field_crossSection)]
    domain_2 = xyz_coord[2][~numpy.isnan(field_crossSection)]
    field_masked_2 = field_crossSection[~numpy.isnan(field_crossSection)]
    ax.scatter(domain_0, domain_1, domain_2, c=field_masked_2, cmap=cm.gist_rainbow_r) #,vmin=vmin_field, vmax=vmax_field,alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    cbar  = plt.colorbar(ax.scatter(domain_0, domain_1, domain_2, c=field_masked_2, cmap=cm.gist_rainbow_r))
    cbar.set_label(filename_str)
    strpicpng= filepath_str+filename_str+'.png'
    print('saving figure', strpicpng)
    plt.savefig(strpicpng, bbox_inches='tight')
    plt.close()
    plt.close('all')
    return

def plot_scalar_plane_psi_theta(plane, field,
            i_nphi_plot2DContour,theta_begn,theta_end, nx_r, nx_theta,
            ax=None, title=None, fmin=None, fmax=None,
            units=None, nlvls=50, log=False, labels=None, cbar=True,
            show=False, filepath_str=None, filename_str=None):
        grid = plane
        p1 = numpy.array([0,theta_begn,i_nphi_plot2DContour])
        p2 = numpy.array([1,theta_begn,i_nphi_plot2DContour])
        p3 = numpy.array([0,theta_end,i_nphi_plot2DContour])
        line1 = numpy.linspace(p1, p2, nx_r).transpose()
        line2 = numpy.linspace(p1, p3, nx_theta).transpose()
        grid = numpy.zeros((3, nx_r, nx_theta))
        grid[:, :, 0] = line1
        grid[:, 0, :] = line2
        for i in range(1, nx_r):
            for j in range(1, nx_theta):
                grid[0, i, j] = line1[0, i] + (line2[0, j] - line2[0, 0])
                grid[1, i, j] = line1[1, i] + (line2[1, j] - line2[1, 0])
                grid[2, i, j] = line1[2, i] + (line2[2, j] - line2[2, 0])
        if ax is None:
            fig, ax = plt.subplots()
        norm = None
        if (fmin is not None and fmax is not None):
            if log and fmin>0.:
                levels = numpy.logspace(numpy.log10(fmin), numpy.log10(fmax), nlvls)
                norm = colors.LogNorm()
            else:
                levels = numpy.linspace(fmin, fmax, nlvls)
        else:
            levels = nlvls
        cf = ax.contourf(grid[0], grid[1], field, levels, norm=norm,cmap=cm.gist_rainbow_r)
        cb = None
        if (cbar):
            cb = colorbar123(ax, cf, labels, units)
        if labels is not None:
            cf2 = ax.contour(cf, levels=labels, colors='r')
            if cb is not None:
                cb.add_lines(cf2)
        ax.set_title(filename_str,fontsize=14, loc='left')
        if title is not None:
            ax.set_title(title, fontsize=12, loc='left')
        strpicpng= filepath_str+filename_str+'.png'
        print('saving figure', strpicpng)
        #print('saving figure', strpicpdf)
        #plt.savefig(strpicpdf, bbox_inches='tight')
        plt.savefig(strpicpng, bbox_inches='tight')
        plt.close()
        if show:
            plt.tight_layout()
            plt.show()
        return

@staticmethod
def plot_grid(points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(points[0], points[1], points[2])
    ax.set_xlabel("X1 Axis")
    ax.set_ylabel("X2 Axis")
    ax.set_zlabel("X3 Axis")
    ax.set_title("Grid Points")
    plt.show()
    return

def contourplotting3D(
        data_read_global_mean,data_read_global_std,
        ntrain,
        S_r,S_theta,
        r_theta_phi, 
        T_out,
        startofpatternlist_i_file_no_in_SelectData,
        i_fieldlist_parm_eq_vector_train_global_lst, fieldlist_parm_eq_vector_train_global_lst_i_j,
        sum_vector_a_elements_i_iter, sum_vector_u_elements_i_iter,
        epochs,
        T_out_sub_time_consecutiveIterator_factor, step,
        i_file_no_in_SelectData, 
        model,
        test_loader,
        S_n_phi, i_nphi_plot2DContour,
        theta_begn, theta_end, 
        n_phi_plot_count, n_phi_plot_begin_factor 
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
                        item_of_sum_vector_test_elements_i = -1
                        for k_fieldlist_parm_eq_vector_train_global_lst_i_j, fieldlist_parm_eq_vector_train_global_lst_i_j_k in enumerate(fieldlist_parm_eq_vector_train_global_lst_i_j[1]):
                                item_of_sum_vector_test_elements_i += 1    
                                fieldlist_parm_lst_i       = fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]
                                fieldlist_parm_eq_selected = fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]
                                fieldlist_parm_vector_i    = fieldlist_parm_eq_vector_train_global_lst_i_j_k[2] 
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
                                    stop
                                if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst))):
                                    os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)))
                                if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i)):
                                    os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i))
                                if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected))):
                                    os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)))
                                if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i))):
                                    os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)))
                                if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count))):
                                    os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count)))
                                if not os.path.exists(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count)+'/pdf/')):
                                    os.makedirs(('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count)+'/pdf/'))
                                if not os.path.exists(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst))):
                                    os.makedirs(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)))
                                if not os.path.exists(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i)):
                                    os.makedirs(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i))
                                if not os.path.exists(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected))):
                                    os.makedirs(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)))
                                if not os.path.exists(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i))):
                                    os.makedirs(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)))
                                if not os.path.exists(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count))):
                                    os.makedirs(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count)))
                                if not os.path.exists(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count)+'/pdf/')):
                                    os.makedirs(('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_lst_i+'/eq' + str(fieldlist_parm_eq_selected)+'/vec' +str(fieldlist_parm_vector_i)+'/ntst'+str(count)+'/pdf/'))
                                Vi = np.array (im.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] )                  
                                Vi  = (Vi* data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )
                                # test_u_range = range(n_beg+(T_in*1),n_beg+((T_out+T_in)*1),1)
                                filename_str= fieldlist_parm_lst_i+'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_selected) +'_'  +str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(count)+ntrain]+t] ).zfill(5)+ "__ntst"+str(count)+"_ep"+ str(ep) + '_T' + str(t)+'_pred'
                                numpy.save((filepath_str + filename_str), Vi)                                
                                plot_3D(r_theta_phi, Vi ,
                                        n_phi_plot_count,
                                        n_phi_plot_begin_factor,
                                        S_r, S_theta,
                                        S_n_phi,
                                        filepath_str=filepath_str, filename_str=filename_str)                                
                                # plot_3D(r_theta_phi, Vi,
                                #     r_cntr,phi,
                                #     phi_begn = phi_begn, phi_end = phi_end,
                                #     theta_begn = theta_begn, theta_end = theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     nx_phi=nx_phi,
                                #     filepath_str=filepath_str, filename_str=filename_str)
                                filepath_str_2d= ('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )
                                # plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                #     r_cntr,phi,theta_begn,theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     filepath_str=filepath_str_2d, filename_str=filename_str)                                
                                plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                        i_nphi_plot2DContour,theta_begn,theta_end, S_r, S_theta,
                                        ax=None, title=None, fmin=None, fmax=None,
                                        units=None, nlvls=50, log=False, labels=None, cbar=True,
                                        show=False, filepath_str=filepath_str_2d, filename_str=filename_str)
                                Vi = np.array (y.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] )
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) + data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(count)+ntrain]+t] ).zfill(5) + "__ntst"+str(count)+"_ep"+ str(ep) + '_T' + str(t)+'_trut'
                                numpy.save((filepath_str + filename_str), Vi)
                                plot_3D(r_theta_phi, Vi ,
                                        n_phi_plot_count,
                                        n_phi_plot_begin_factor,
                                        S_r, S_theta,
                                        S_n_phi,
                                        filepath_str=filepath_str, filename_str=filename_str)                                
                                # plot_3D(r_theta_phi, Vi,
                                #     r_cntr,phi,
                                #     phi_begn = phi_begn, phi_end = phi_end,
                                #     theta_begn = theta_begn, theta_end = theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     nx_phi=nx_phi,
                                #     filepath_str=filepath_str, filename_str=filename_str)
                                filepath_str_2d= ('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )
                                # plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                #     r_cntr,phi,theta_begn,theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     filepath_str=filepath_str_2d, filename_str=filename_str)
                                plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                        i_nphi_plot2DContour,theta_begn,theta_end, S_r, S_theta,
                                        ax=None, title=None, fmin=None, fmax=None,
                                        units=None, nlvls=50, log=False, labels=None, cbar=True,
                                        show=False, filepath_str=filepath_str_2d, filename_str=filename_str)
                                Vi =  np.array (im.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] ) 
                                Vi = (Vi*data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy()) # + - data_dump_mean.cpu().detach().numpy()
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' + str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(count)+ntrain]+t] ).zfill(5)+ "__ntst"+str(count)+"_ep"+ str(ep) + '_T' + str(t)+'_eror'
                                numpy.save((filepath_str + filename_str), Vi)
                                plot_3D(r_theta_phi, Vi ,
                                        n_phi_plot_count,
                                        n_phi_plot_begin_factor,
                                        S_r, S_theta,
                                        S_n_phi,
                                        filepath_str=filepath_str, filename_str=filename_str)                                
                                # plot_3D(r_theta_phi, Vi,
                                #     r_cntr,phi,
                                #     phi_begn = phi_begn, phi_end = phi_end,
                                #     theta_begn = theta_begn, theta_end = theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     nx_phi=nx_phi,
                                #     filepath_str=filepath_str, filename_str=filename_str)
                                filepath_str_2d= ('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )
                                # plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                #     r_cntr,phi,theta_begn,theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     filepath_str=filepath_str_2d, filename_str=filename_str)
                                plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                        i_nphi_plot2DContour,theta_begn,theta_end, S_r, S_theta,
                                        ax=None, title=None, fmin=None, fmax=None,
                                        units=None, nlvls=50, log=False, labels=None, cbar=True,
                                        show=False, filepath_str=filepath_str_2d, filename_str=filename_str)
                                Vi =  (  np.array (im.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] ) -  np.array (y.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i]) )/ (abs( np.array (y.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] ) + ( data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() / data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() ) ) )
                                filepath_str= './plots_Colrs/'+ str(count)+'/'                     
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' ) 
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) +'_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' +str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(count)+ntrain]+t] ).zfill(5)+ "__ntst"+str(count)+"_ep"+ str(ep) + '_T' + str(t)+'_nrml'
                                numpy.save((filepath_str + filename_str), Vi)
                                plot_3D(r_theta_phi, Vi ,
                                        n_phi_plot_count,
                                        n_phi_plot_begin_factor,
                                        S_r, S_theta,
                                        S_n_phi,
                                        filepath_str=filepath_str, filename_str=filename_str)                                
                                # plot_3D(r_theta_phi, Vi,
                                #     r_cntr,phi,
                                #     phi_begn = phi_begn, phi_end = phi_end,
                                #     theta_begn = theta_begn, theta_end = theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     nx_phi=nx_phi,
                                #     filepath_str=filepath_str, filename_str=filename_str)
                                filepath_str_2d= ('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' ) 
                                # plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                #     r_cntr,phi,theta_begn,theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     filepath_str=filepath_str_2d, filename_str=filename_str)
                                plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                        i_nphi_plot2DContour,theta_begn,theta_end, S_r, S_theta,
                                        ax=None, title=None, fmin=None, fmax=None,
                                        units=None, nlvls=50, log=False, labels=None, cbar=True,
                                        show=False, filepath_str=filepath_str_2d, filename_str=filename_str)                                
                                Vi =  (     np.array (im.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] ) - np.array (y.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i] )  )/  ( (np.max(abs(y.cpu().numpy()[0,:,:,:,item_of_sum_vector_test_elements_i]))                                                                                                   ) + ( data_read_global_mean[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() / data_read_global_std[fieldlist_parm_eq_vector_train_global_lst_i_j_0,fieldlist_parm_eq_vector_train_global_lst_i_j_k[1],fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]].cpu().detach().numpy() ) )
                                Vi = Vi
                                filepath_str= ('./plots_Colrs/'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )            
                                filename_str= fieldlist_parm_eq_vector_train_global_lst_i_j_k[0] +'_vec'+str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2]) + '_eq' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1]) +'_' +str(i_file_no_in_SelectData[startofpatternlist_i_file_no_in_SelectData[(count)+ntrain]+t] ).zfill(5)+ "__ntst"+str(count)+"_ep"+ str(ep) + '_T' + str(t)+'_MxNm'
                                numpy.save((filepath_str + filename_str), Vi)
                                plot_3D(r_theta_phi, Vi ,
                                        n_phi_plot_count,
                                        n_phi_plot_begin_factor,
                                        S_r, S_theta,
                                        S_n_phi,
                                        filepath_str=filepath_str, filename_str=filename_str)                                
                                # plot_3D(r_theta_phi, Vi,
                                #     r_cntr,phi,
                                #     phi_begn = phi_begn, phi_end = phi_end,
                                #     theta_begn = theta_begn, theta_end = theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     nx_phi=nx_phi,
                                #     filepath_str=filepath_str, filename_str=filename_str)
                                filepath_str_2d= ('./plots_Colrs/nPhi'+ str(i_nphi_plot2DContour)+'_2d_'+str(i_fieldlist_parm_eq_vector_train_global_lst)+'/'+fieldlist_parm_eq_vector_train_global_lst_i_j_k[0]+'/eq' + str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[1])+'/vec' +str(fieldlist_parm_eq_vector_train_global_lst_i_j_k[2])+'/ntst' + str(count)+'/' )            
                                # plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                #     r_cntr,phi,theta_begn,theta_end,r_begn_estmt=None,r_end_estmt=None, nx_r=nx_r, nx_theta=nx_theta,
                                #     filepath_str=filepath_str_2d, filename_str=filename_str)
                                plot_scalar_plane_psi_theta(r_theta_phi[ :,:,:,i_nphi_plot2DContour], Vi[:,:,i_nphi_plot2DContour],
                                        i_nphi_plot2DContour,theta_begn,theta_end, S_r, S_theta,
                                        ax=None, title=None, fmin=None, fmax=None,
                                        units=None, nlvls=50, log=False, labels=None, cbar=True,
                                        show=False, filepath_str=filepath_str_2d, filename_str=filename_str)  
                                # print('411 count',count)
                                if count == 0:                                   
                                    break
                                    continue # Dummy Condition  to just run the loop for number of count. Can comment this continue to run loop for all the paramters.
                                    pass                       
                plt.close('all')
                #print('424 count',count)
                # if count == 1:   
                if True:   
                    break
                    continue # Dummy Condition  to just run the loop for number of count. Can comment this continue to run loop for all the paramters.
                    pass                              