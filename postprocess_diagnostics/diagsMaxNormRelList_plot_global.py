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

import matplotlib.pyplot as pl
from scipy.interpolate import interp1d,splev,UnivariateSpline
import os
import h5py
from stfno.utilities3 import *
from timeit import default_timer
import matplotlib as mpl


def read_diags_filename_WO1stline(filename):
    with open(filename) as f:
        i=0
        data_diags = []
        for line in f:
            i=i+1
            if i != 1 : 
                content = line.strip('\n')
                line_list=[]
                for s in content.split():
                    s = s.strip(',')
                    s = s.split(",")
                    line_list.append(  s )
                line_list = np.array( line_list)
                line_data = line_list.astype(float)
                data_diags.append(line_data)
    data_diags = np.array(data_diags,dtype = np.float32)
    data_diags = np.transpose(data_diags)
    return data_diags

def plot_epochs():
    linewidthValue = 1
    linestylekeys = [ '-','--','-.','--','-','--','-.','--', (0, (linewidthValue, linewidthValue)),(0,(5*linewidthValue,2*linewidthValue)),'-',(0, (5*linewidthValue, 2*linewidthValue, linewidthValue, 2*linewidthValue)),'-.',(0, (3, 5, 1, 5, 1, 5)), (0, ()) , (0, (1, 10))  ,  (0, (1, 5)), (0, (1, 1)),  (0, (5, 10)), (0, (5, 5)),(0, (5, 1)), (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),(0, (3, 10, 1, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))                   ,(0, ()) , (0, (1, 10))  ,  (0, (1, 5)), (0, (1, 1)),  (0, (5, 10)), (0, (5, 5)),(0, (5, 1)), (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),(0, (3, 10, 1, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))                   ,(0, ()) , (0, (1, 10))  ,  (0, (1, 5)), (0, (1, 1)),  (0, (5, 10)), (0, (5, 5)),(0, (5, 1)), (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),(0, (3, 10, 1, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))                   ,(0, ()) , (0, (1, 10))  ,  (0, (1, 5)), (0, (1, 1)),  (0, (5, 10)), (0, (5, 5)),(0, (5, 1)), (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),(0, (3, 10, 1, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))                   ]
    linewidthValue = 1.2
    linewidthArry=np.array( [ linewidthValue,linewidthValue,linewidthValue,linewidthValue,linewidthValue,linewidthValue,linewidthValue,4.8,4.75,4.7,4.6,4.5,4.4,4.3,4.25,4.2,4.1,4,3.9,3.8,3.75,3.7,3.6,3.5,3.4,3.3,3.25,3.2,3.1,3,2.9,2.8,2.75,2.7,2.6,2.5,2.4,2.3,2.25,2.2,2.1,2.,1.9,1.8,1.75,1.7,1.6,1.5,1.4,1.3,1.25,1.2,1.1,1,0.95]) 
    markerstr = ['v','^','>','<','o','*','+','x','o','*','8','s','p','*','h','+','x','D','1','2','3','4'            , 'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ]
    colorstr = [ 'darkred','darkgreen','darkblue','k','saddlebrown','darkorange','brown','olive','c','m','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink' ,                 ((1., 0., 0.)),((0.,1., 0.)),((0., 0., 1.)) ,((0.75, 0.25, 0.)) ]
    markersizestr= np.array( [(1.8+1.8),(1.6+2),(1.4+2.2),(0.6+1.2+2.4),(1.+2.6),(.8+2.8), (4.9+1.0),(4.8+1.0),(4.75+1.0),(4.7+1.0),(4.6+1.0),(4.5+1.0),(4.4+1.0),(4.3+1.0),(4.25+1.0),(4.2+1.0),(4.1+1.0),(4+1.0),(3.9+1.0),(3.8+1.0),(3.75+1.0),(3.7+1.0),(3.6+1.0),(3.5+1.0),(3.4+1.0),(3.3+1.0),(3.25+1.0),(3.2+1.0),(3.1+1.0),(3+1.0),(2.9+1.0),(2.8+1.0),(2.75+1.0),(2.7+1.0),(2.6+1.0),(2.5+1.0),(2.4+1.0),(2.3+1.0),(2.25+1.0),(2.2+1.0),(2.1+1.0),(2.+1.0),(1.9+1.0),(1.8+1.0),(1.75+1.0),(1.7+1.0),(1.6+1.0),(1.5+1.0),(1.4+1.0),(1.3+1.0),(1.25+1.0),(1.2+1.0),(1.1+1.0),(1+1.0),(0.95)] )
    markersizestr=markersizestr-1.0 -.8 
    markeredgecolorstr = [ ((1., 0., 0.)),((0.,0., 1.)),((0., 0., 1.)),((0.75, 0.25, 0.)),'c','m','c',((0., 0., 0.)),'darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ]
    markeredgecolorstr = [ 'darkred','darkgreen','darkblue','k','c','m','k','c','c','m','saddlebrown','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink' ,                 ((1., 0., 0.)),((0.,1., 0.)),((0., 0., 1.)) ,((0.75, 0.25, 0.)) ]

    folders_considered = [ 
        #'../57_14/', \
        './', \
        ]

    legends_compnt = [        
        #r'$ep$',\
        #r'$L1$',\
        r'error',\
        'train_l1_step / ntrain / (T / step)',\
        'train_l1_full / ntrain',\
        'test_l1_step / ntest / (T / step)',\
        'test_l1_full / ntest',\
        #r'count_params(model)' ,\
        ]    
    print('len(folders_considered[0]',len(folders_considered[0]),'folders_considered[0]',folders_considered[0] )
    directory = 'data'
    data_file_starting_part =['epochs_diagsMaxNormRelList_' ]
    data_file_correspondingLegends =['L1' ]

    fig1 , ax_1 = pl.subplots()
    fig2 , ax_2 = pl.subplots()
    all_filename_ii = ''
    
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    if not os.path.exists('./plots/diagsMaxNormRelList/pdf/'):
        os.makedirs('./plots/diagsMaxNormRelList/pdf/')
    # filename_ii=filename_i.replace('.txt','')
    if not os.path.exists(('./plots/diagsMaxNormRelList/'+all_filename_ii)):
        os.makedirs(('./plots/diagsMaxNormRelList/'+all_filename_ii))
    if not os.path.exists(('./plots/diagsMaxNormRelList/'+all_filename_ii+'/pdf/')):
        os.makedirs(('./plots/diagsMaxNormRelList/'+all_filename_ii+'/pdf/'))
    # iterate over files in that directory
    for filename_i in os.listdir(directory):
        #if filename_i.endswith('_diags.txt'):   # printing file name of desired extension else:
        for  data_file_starting_part_itr in  data_file_starting_part:
            if filename_i.startswith(data_file_starting_part_itr) :   # printing file name of desired extension else:
                file_i = os.path.join(directory, filename_i)
                # checking if it is a file
                if os.path.isfile(file_i):
                    print(file_i)
                    for k in range(0,len([folders_considered[0] ])):
                        fig1 , ax_1 = pl.subplots()
                        fig2 , ax_2 = pl.subplots()

                        m=-1
                        for i in  folders_considered: 
                            m=m+1
                            filename = file_i
                            print('filename =',filename)
                            diags_data = read_diags_filename_WO1stline(filename)

                            ax_1.plot(diags_data[0,:],diags_data[2,:],linestyle=linestylekeys[1],linewidth=linewidthArry[1],marker= markerstr[1],color=colorstr[1] ,markersize= markersizestr[1],markeredgecolor =markeredgecolorstr[1])
                            ax_2.plot(diags_data[1,:],diags_data[2,:],linestyle=linestylekeys[2],linewidth=linewidthArry[2],marker= markerstr[2],color=colorstr[2] ,markersize= markersizestr[2],markeredgecolor =markeredgecolorstr[2])

                        # filenameWithoutDotTXT = filename[:-4]
                        if not os.path.exists('./plots/'):
                            os.makedirs('./plots/')
                        if not os.path.exists('./plots/diagsMaxNormRelList/pdf/'):
                            os.makedirs('./plots/diagsMaxNormRelList/pdf/')
                        filename_ii=filename_i.replace('.txt','')
                        if not os.path.exists(('./plots/diagsMaxNormRelList/'+filename_ii)):
                            os.makedirs(('./plots/diagsMaxNormRelList/'+filename_ii))
                        if not os.path.exists(('./plots/diagsMaxNormRelList/'+filename_ii+'/pdf/')):
                            os.makedirs(('./plots/diagsMaxNormRelList/'+filename_ii+'/pdf/'))
                
                        ax_1.set_xlabel(r'$test\ sample\ number$') #r'$y^+$'
                        ax_1.set_ylabel(legends_compnt[0])
                        ax_1.grid()
                        strpicpdf= './plots/diagsMaxNormRelList/' +filename_ii + '/pdf/'+ str(k)+'t2minustt1'+'.pdf'
                        strpicpng= './plots/diagsMaxNormRelList/' +filename_ii + '/'    + str(k)+'t2minustt1'+'.png'
                        print('strpicpng=',strpicpng)
                        print('strpicpdf=',strpicpdf)
                        fig1.savefig(strpicpdf, bbox_inches='tight')
                        fig1.savefig(strpicpng, bbox_inches='tight')
                        #plt.show(block=False)
                        #timen.sleep(1)
                        pl.close()

                        ax_1.set_yscale('log')
                        strpicpdf= './plots/diagsMaxNormRelList/' +filename_ii +'/pdf/'+str(k)+'t2minustt1_ylog'+'.pdf'
                        strpicpng= './plots/diagsMaxNormRelList/' +filename_ii +'/'    +str(k)+'t2minustt1_ylog'+'.png'
                        print('strpicpng=',strpicpng)
                        print('strpicpdf=',strpicpdf)
                        fig1.savefig(strpicpdf, bbox_inches='tight')
                        fig1.savefig(strpicpng, bbox_inches='tight')
                        #plt.show(block=False)
                        #timen.sleep(1)
                        pl.close()

                        ax_2.set_xlabel(r'time step $k$') #r'$y^+$'
                        ax_2.set_ylabel(legends_compnt[0])
                        strpicpdf= './plots/diagsMaxNormRelList/' +filename_ii+'/pdf/' + str(k)+'train_l1_stepBYntrainBy__TBystep__'+'.pdf'
                        strpicpng= './plots/diagsMaxNormRelList/' +filename_ii+'/'     + str(k)+'train_l1_stepBYntrainBy__TBystep__'+'.png'
                        ax_2.grid()
                        print('strpicpng=',strpicpng)
                        print('strpicpdf=',strpicpdf)
                        fig2.savefig(strpicpdf, bbox_inches='tight')
                        fig2.savefig(strpicpng, bbox_inches='tight')
                        #plt.show(block=False)
                        #timen.sleep(1)
                        pl.close()

                        ax_2.set_yscale('log')
                        strpicpdf= './plots/diagsMaxNormRelList/' +filename_ii+'/pdf/'+str(k)+'train_l1_stepBYntrainBy__TBystep___ylog'+'.pdf'
                        strpicpng= './plots/diagsMaxNormRelList/' +filename_ii+'/'    +str(k)+'train_l1_stepBYntrainBy__TBystep___ylog'+'.png'
                        print('strpicpng=',strpicpng)
                        print('strpicpdf=',strpicpdf)
                        fig2.savefig(strpicpdf, bbox_inches='tight')
                        fig2.savefig(strpicpng, bbox_inches='tight')

                        pl.close('all')
plot_epochs()