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

import matplotlib.pyplot as pl
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.interpolate import interp1d,splev,UnivariateSpline
import os
import h5py
from timeit import default_timer
import matplotlib as mpl
import numpy as np

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
    linewidthValue = 4
    linewidthArry=np.array( [ linewidthValue,linewidthValue,linewidthValue,linewidthValue,linewidthValue,linewidthValue,linewidthValue,4.8,4.75,4.7,4.6,4.5,4.4,4.3,4.25,4.2,4.1,4,3.9,3.8,3.75,3.7,3.6,3.5,3.4,3.3,3.25,3.2,3.1,3,2.9,2.8,2.75,2.7,2.6,2.5,2.4,2.3,2.25,2.2,2.1,2.,1.9,1.8,1.75,1.7,1.6,1.5,1.4,1.3,1.25,1.2,1.1,1,0.95]) 
    markerstr = ['v','^','>','<','o','*','+','x','o','*','8','s','p','*','h','+','x','D','1','2','3','4'            , 'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ,'o','v','^','<','>','8','s','p','*','h','+','x','D','1','2','3','4'            ]
    colorstr = [ 'darkred','darkgreen','darkblue','k','saddlebrown','darkorange','brown','olive','c','m','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink' ,                 ((1., 0., 0.)),((0.,1., 0.)),((0., 0., 1.)) ,((0.75, 0.25, 0.)) ]
    colorstr_yang=['orangered','darkslateblue']    
    markersizestr= np.array( [(1.8+1.8),(1.6+2),(1.4+2.2),(0.6+1.2+2.4),(1.+2.6),(.8+2.8), (4.9+1.0),(4.8+1.0),(4.75+1.0),(4.7+1.0),(4.6+1.0),(4.5+1.0),(4.4+1.0),(4.3+1.0),(4.25+1.0),(4.2+1.0),(4.1+1.0),(4+1.0),(3.9+1.0),(3.8+1.0),(3.75+1.0),(3.7+1.0),(3.6+1.0),(3.5+1.0),(3.4+1.0),(3.3+1.0),(3.25+1.0),(3.2+1.0),(3.1+1.0),(3+1.0),(2.9+1.0),(2.8+1.0),(2.75+1.0),(2.7+1.0),(2.6+1.0),(2.5+1.0),(2.4+1.0),(2.3+1.0),(2.25+1.0),(2.2+1.0),(2.1+1.0),(2.+1.0),(1.9+1.0),(1.8+1.0),(1.75+1.0),(1.7+1.0),(1.6+1.0),(1.5+1.0),(1.4+1.0),(1.3+1.0),(1.25+1.0),(1.2+1.0),(1.1+1.0),(1+1.0),(0.95)] )
    markersizestr=markersizestr-1.0 -.8 
    markeredgecolorstr = [ ((1., 0., 0.)),((0.,0., 1.)),((0., 0., 1.)),((0.75, 0.25, 0.)),'c','m','c',((0., 0., 0.)),'darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ]
    markeredgecolorstr = [ 'darkred','darkgreen','darkblue','k','c','m','k','c','c','m','saddlebrown','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink'                    ,'b','g','r','c','m','y','grey','k','darkorange','brown','olive','darkgoldenrod','rosybrown','darksalmon','orangered','darkkhaki','darkgreen','darkslateblue','royalblue','indigo','violet','crimson','pink' ,                 ((1., 0., 0.)),((0.,1., 0.)),((0., 0., 1.)) ,((0.75, 0.25, 0.)) ]
    folders_considered = [ 
        './',\
        '/global/cfs/cdirs/mp127/nimrod/nimrod_hdf509_hyperdiffusivity_read_data_corrected6_2_OrignalFNOruns/vnpb_S64_read_data_01_F_10_10__vnpb_123_0__FNO2d_glob_orig_Layers4__Tin5__Tout5__IncludeSteadyStateFalse_epochs500_TmaxOne_WithMaxNormOverall_WithMaxNormEachSample_RemoveAllPrint_SaveLoadModel_GPUrun_19/',\
        ]
    folders_considered_correspondingLegends =['FS-FNO','Baseline-FNO' ]
    legends_compnt = [        
        't2 - t1',\
        'norm',\
        'norm',\
        'norm',\
        'L2 error',\
        ]
    print('len(folders_considered[0]',len(folders_considered[0]),'folders_considered[0]',folders_considered[0] )
    directory = 'data'
    data_file_starting_part =['epochs_diags_' ]
    data_file_correspondingLegends =['']
    fig1 , ax_1 = pl.subplots()
    fig2 , ax_2 = pl.subplots()
    fig3 , ax_3 = pl.subplots()
    fig4 , ax_4 = pl.subplots()
    fig5 , ax_5 = pl.subplots(figsize=(9, 6))
    all_filename_ii = ''.join(data_file_starting_part)
                        

    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    if not os.path.exists('./plots/diags_together/pdf/'):
        os.makedirs('./plots/diags_together/pdf/')
    if not os.path.exists(('./plots/diags_together/'+all_filename_ii)):
        os.makedirs(('./plots/diags_together/'+all_filename_ii))
    if not os.path.exists(('./plots/diags_together/'+all_filename_ii+'/pdf/')):
        os.makedirs(('./plots/diags_together/'+all_filename_ii+'/pdf/'))

    strn_epochsAverageTime_dump_path_file = './plots/diags_together/'+all_filename_ii +'/averageTimeforTrainingperepochsPlusTestingperepochs.txt'
    file2 = open(strn_epochsAverageTime_dump_path_file, "a") 

    m=-1
    legstr_each = []
    for k_folders_considered, folders_considered_k in enumerate(folders_considered):
        print(' k_folders_considered',k_folders_considered, 'folders_considered', folders_considered)
        folders_considered_plus_directory = folders_considered_k + directory
        print(' folders_considered_plus_directory',folders_considered_plus_directory)
        
        for filename_i in os.listdir(folders_considered_plus_directory):            
            for  itr_data_file_starting_part,data_file_starting_part_itr in enumerate(data_file_starting_part):
                print(' data_file_starting_part_itr',data_file_starting_part_itr)
                print(' filename_i',filename_i)
                if filename_i.startswith(data_file_starting_part_itr) :   # printing file name of desired extension else:
                    print(' all_filename =ii',all_filename_ii)
                    file_i = os.path.join(folders_considered_plus_directory, filename_i)
                    print(' file_i',file_i)
                    print(' data_file_starting_part_itr',data_file_starting_part_itr)
                    print(' all_filename_ii',all_filename_ii)
                    # checking if it is a file
                    if os.path.isfile(file_i):
                            print(file_i)
                            m=m+1
                            legstr_each.append( folders_considered_correspondingLegends[(k_folders_considered)] + data_file_correspondingLegends[itr_data_file_starting_part]) 
                            filename = file_i
                            filename_ii=filename_i.replace('.txt','')
                            print('filename =',filename)
                            diags_data = read_diags_filename_WO1stline(filename)
                            ax_1.scatter(diags_data[0,:],diags_data[1,:],s= 1.2,linestyle='None',linewidth=linewidthArry[1],marker= markerstr[1+5*m],color=colorstr[1+5*m], alpha = 0.6,label= legstr_each[m])
                            ax_2.scatter(diags_data[0,:],diags_data[2,:],s= 1.2,linestyle='None',linewidth=linewidthArry[2],marker= markerstr[2+5*m],color=colorstr[2+5*m],alpha = 0.6,label= legstr_each[m])
                            ax_3.scatter(diags_data[0,:],diags_data[3,:],s= 1.2,linestyle='None',linewidth=linewidthArry[3],marker= markerstr[3+5*m],color=colorstr[3+5*m] ,alpha = 0.6,label= legstr_each[m])
                            ax_4.scatter(diags_data[0,:],diags_data[4,:],s= 1.2,linestyle='None',linewidth=linewidthArry[4],marker= markerstr[4+5*m],color=colorstr[4+5*m] ,alpha = 0.6,label= legstr_each[m])
                            ax_5.scatter(diags_data[0,:],diags_data[5,:],s= 1.2,linestyle='None',linewidth=linewidthArry[5],marker= markerstr[5+5*m],color=colorstr_yang[m],alpha = 0.9,label= legstr_each[m])
                            print( ' strn_epochsAverageTime_dump_path_file:' +strn_epochsAverageTime_dump_path_file+' filename : ' + filename +':- ' +  str(np.average(  diags_data[1,:] ) ) +' sec')
                            str_file2= ( filename +': ' + str(np.average(  diags_data[1,:] ) ) +'\n')
                            file2.write(str_file2)
    file2.close()                   
    legstr = legstr_each
    print('diags_plot_global.py 175 legstr',legstr)
    ax_1.legend(legstr,loc='best',prop={'size': 16.5},frameon=False,framealpha =1.0 )# ,facecolor='w' )
    ax_1.set_xlabel('Epoch') #r'$y^+$'
    ax_1.set_ylabel(legends_compnt[0])
    ax_1.grid()
    strpicpdf= './plots/diags_together/' +all_filename_ii + '/pdf/'+ 't2minustt1'+'.pdf' # +str(k)
    strpicpng= './plots/diags_together/' +all_filename_ii + '/'    + 't2minustt1'+'.png' # +str(k)
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig1.savefig(strpicpdf, bbox_inches='tight')
    fig1.savefig(strpicpng, bbox_inches='tight')
    pl.close()
    ax_1.set_yscale('log')
    strpicpdf= './plots/diags_together/' +all_filename_ii +'/pdf/'+'t2minustt1_ylog'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii +'/'    +'t2minustt1_ylog'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig1.savefig(strpicpdf, bbox_inches='tight')
    fig1.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()

    ax_2.legend(legstr,loc='best',prop={'size': 16.5},frameon=False,framealpha =1.0 )# ,facecolor='w' )
    ax_2.set_xlabel('Epoch') #r'$y^+$'
    ax_2.set_ylabel(legends_compnt[1])
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/' +'train_l2_stepBYntrainBy__TBystep__'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'     +'train_l2_stepBYntrainBy__TBystep__'+'.png'
    ax_2.grid()
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig2.savefig(strpicpdf, bbox_inches='tight')
    fig2.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()
    ax_2.set_yscale('log')
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/'+'train_l2_stepBYntrainBy__TBystep___ylog'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'    +'train_l2_stepBYntrainBy__TBystep___ylog'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig2.savefig(strpicpdf, bbox_inches='tight')
    fig2.savefig(strpicpng, bbox_inches='tight')

    ax_3.legend(legstr,loc='best',prop={'size': 16.5},frameon=False,framealpha =1.0 )# ,facecolor='w' )
    ax_3.set_xlabel('Epoch') #r'$y^+$'
    ax_3.set_ylabel(legends_compnt[2])
    ax_3.grid()
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/'+'train_l2_fullBYntrain'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'    +'train_l2_fullBYntrain'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig3.savefig(strpicpdf, bbox_inches='tight')
    fig3.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()
    ax_3.set_yscale('log')
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/' +'train_l2_fullBYntrain_ylog'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'     +'train_l2_fullBYntrain_ylog'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig3.savefig(strpicpdf, bbox_inches='tight')
    fig3.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()


    ax_4.legend(legstr,loc='best',prop={'size': 16.5},frameon=False,framealpha =1.0 )# ,facecolor='w' )
    ax_4.set_xlabel('Epoch') #r'$y^+$'
    ax_4.set_ylabel(legends_compnt[3])
    ax_4.grid()
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/' +'test_l2_stepBYntestBY__TBYstep__'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'     +'test_l2_stepBYntestBY__TBYstep__'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig4.savefig(strpicpdf, bbox_inches='tight')
    fig4.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()
    ax_4.set_yscale('log')
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/'+'test_l2_stepBYntestBY__TBYstep___ylog'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'    +'test_l2_stepBYntestBY__TBYstep___ylog'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig4.savefig(strpicpdf, bbox_inches='tight')
    fig4.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()
    labelsize=20
    ax_5.legend(legstr,loc='best',prop={'size': 16.5},frameon=False,framealpha =1.0,fontsize=labelsize )# ,facecolor='w' )
    ax_5.set_xlabel('Epoch',fontsize=labelsize) #r'$y^+$'
    # ax_5.set_ylabel(legends_compnt[4],fontsize=labelsize)
    ax_5.tick_params(axis='both', which='major', labelsize=labelsize)
    ax_5.grid()
    ax_5.text(50, 0.3, r'$T_{i}=5,T_{o}=5$', fontsize=30, color='black')
    strpicpdf= './plots/diags_together/'  +all_filename_ii+'/pdf/'+'test_l2_fullBYntest'+'.pdf'
    strpicpng= './plots/diags_together/'  +all_filename_ii+'/'    +'test_l2_fullBYntest'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig5.savefig(strpicpdf, bbox_inches='tight')
    fig5.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()
    ax_5.set_yscale('log')
    yticks = [0.01, 0.1, 1]
    ax_5.set_yticks(yticks)
    
    strpicpdf= './plots/diags_together/' +all_filename_ii+'/pdf/' +'test_l2_fullBYntest_ylog'+'.pdf'
    strpicpng= './plots/diags_together/' +all_filename_ii+'/'     +'test_l2_fullBYntest_ylog'+'.png'
    print('strpicpng=',strpicpng)
    print('strpicpdf=',strpicpdf)
    fig5.savefig(strpicpdf, bbox_inches='tight')
    fig5.savefig(strpicpng, bbox_inches='tight')
    #plt.show(block=False)
    #timen.sleep(1)
    pl.close()
    pl.close('all')
    for k_folders_considered, folders_considered_k in enumerate(folders_considered):
        print(' k_folders_considered',k_folders_considered, 'folders_considered', folders_considered)
        folders_considered_plus_directory = folders_considered_k + directory
        print(' folders_considered_plus_directory',folders_considered_plus_directory)
        for filename_i in os.listdir(folders_considered_plus_directory):                     
            for  itr_data_file_starting_part,data_file_starting_part_itr in enumerate(data_file_starting_part):
                print(' data_file_starting_part_itr',data_file_starting_part_itr)
                print(' filename_i',filename_i)
                if filename_i.startswith(data_file_starting_part_itr) :   # printing file name of desired extension else:
                    print(' all_filename =ii',all_filename_ii)
                    file_i = os.path.join(folders_considered_plus_directory, filename_i)
                    print(' file_i',file_i)
                    print(' data_file_starting_part_itr',data_file_starting_part_itr)
                    print(' all_filename_ii',all_filename_ii)
                    if os.path.isfile(file_i):
                            print(file_i)
                            m=m+1
                    file_i = os.path.join(folders_considered_plus_directory, filename_i)
                    print(' file_i',file_i)
                    print(' data_file_starting_part_itr',data_file_starting_part_itr)
                    print(' all_filename_ii',all_filename_ii)
                    if os.path.isfile(file_i):
                            print(file_i)
                            m=m+1
                            legstr_each.append( folders_considered[(k_folders_considered)] + data_file_correspondingLegends[itr_data_file_starting_part]) 
                            filename = file_i
                            filename_ii=filename_i.replace('.txt','')
                            print('filename =',filename)
                            diags_data = read_diags_filename_WO1stline(filename)
                            fig1 , ax_1 = pl.subplots()
                            fig2 , ax_2 = pl.subplots()
                            fig3 , ax_3 = pl.subplots()
                            fig4 , ax_4 = pl.subplots()
                            fig5 , ax_5 = pl.subplots()
                            ax_1.plot(diags_data[0,:],diags_data[1,:],linestyle=linestylekeys[1],linewidth=linewidthArry[1],marker= markerstr[1],color=colorstr[1] ,markersize= markersizestr[1],markeredgecolor =markeredgecolorstr[1])
                            ax_2.plot(diags_data[0,:],diags_data[2,:],linestyle=linestylekeys[2],linewidth=linewidthArry[2],marker= markerstr[2],color=colorstr[2] ,markersize= markersizestr[2],markeredgecolor =markeredgecolorstr[2])
                            ax_3.plot(diags_data[0,:],diags_data[3,:],linestyle=linestylekeys[3],linewidth=linewidthArry[3],marker= markerstr[3],color=colorstr[3] ,markersize= markersizestr[3],markeredgecolor =markeredgecolorstr[3])
                            ax_4.plot(diags_data[0,:],diags_data[4,:],linestyle=linestylekeys[4],linewidth=linewidthArry[4],marker= markerstr[4],color=colorstr[4] ,markersize= markersizestr[4],markeredgecolor =markeredgecolorstr[4])
                            ax_5.plot(diags_data[0,:],diags_data[5,:],linestyle=linestylekeys[5],linewidth=linewidthArry[5],marker= markerstr[5],color=colorstr[5] ,markersize= markersizestr[5],markeredgecolor =markeredgecolorstr[5])
                            filenameWithoutDotTXT = filename[:-4]
                            if not os.path.exists('./plots/'):
                                os.makedirs('./plots/')
                            if not os.path.exists('./plots/diags/pdf/'):
                                os.makedirs('./plots/diags/pdf/')
                            filename_ii=filename_i.replace('.txt','')
                            if not os.path.exists(('./plots/diags/'+filename_ii)):
                                os.makedirs(('./plots/diags/'+filename_ii))
                            if not os.path.exists(('./plots/diags/'+filename_ii+'/pdf/')):
                                os.makedirs(('./plots/diags/'+filename_ii+'/pdf/'))
                            strn_epochsAverageTime_dump_path_file = './plots/diags/'+filename_ii +'/averageTimeforTrainingperepochsPlusTestingperepochs_2.txt'
                            file3 = open(strn_epochsAverageTime_dump_path_file, "a") 
                            print( '449 strn_epochsAverageTime_dump_path_file:' +strn_epochsAverageTime_dump_path_file+' filename_ii : ' + filename_ii +':- ' +  str(np.average(  diags_data[1,:] ) ) +' sec')
                            str_file3= ( filename_ii +': ' + str(np.average(  diags_data[1,:] ) ) +' sec \n' )
                            file3.write(str_file3)
                            file3.close()                   

                            ax_1.set_xlabel('Epoch') #r'$y^+$'
                            ax_1.set_ylabel(legends_compnt[0])
                            ax_1.grid()
                            strpicpdf= './plots/diags/' +filename_ii + '/pdf/'+ str(k_folders_considered)+'t2minustt1'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii + '/'    + str(k_folders_considered)+'t2minustt1'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig1.savefig(strpicpdf, bbox_inches='tight')
                            fig1.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()
                            ax_1.set_yscale('log')
                            strpicpdf= './plots/diags/' +filename_ii +'/pdf/'+str(k_folders_considered)+'t2minustt1_ylog'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii +'/'    +str(k_folders_considered)+'t2minustt1_ylog'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig1.savefig(strpicpdf, bbox_inches='tight')
                            fig1.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()

                            ax_2.set_xlabel('Epoch') #r'$y^+$'
                            ax_2.set_ylabel(legends_compnt[1])
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/' + str(k_folders_considered)+'train_l2_stepBYntrainBy__TBystep__'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'     + str(k_folders_considered)+'train_l2_stepBYntrainBy__TBystep__'+'.png'
                            ax_2.grid()
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig2.savefig(strpicpdf, bbox_inches='tight')
                            fig2.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()
                            ax_2.set_yscale('log')
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/'+str(k_folders_considered)+'train_l2_stepBYntrainBy__TBystep___ylog'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'    +str(k_folders_considered)+'train_l2_stepBYntrainBy__TBystep___ylog'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig2.savefig(strpicpdf, bbox_inches='tight')
                            fig2.savefig(strpicpng, bbox_inches='tight')

                            ax_3.set_xlabel('Epoch') #r'$y^+$'
                            ax_3.set_ylabel(legends_compnt[2])
                            ax_3.grid()
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/'+str(k_folders_considered)+'train_l2_fullBYntrain'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'    +str(k_folders_considered)+'train_l2_fullBYntrain'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig3.savefig(strpicpdf, bbox_inches='tight')
                            fig3.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()
                            ax_3.set_yscale('log')
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/' +str(k_folders_considered)+'train_l2_fullBYntrain_ylog'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'     +str(k_folders_considered)+'train_l2_fullBYntrain_ylog'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig3.savefig(strpicpdf, bbox_inches='tight')
                            fig3.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()

                            ax_4.set_xlabel('Epoch') #r'$y^+$'
                            ax_4.set_ylabel(legends_compnt[3])
                            ax_4.grid()
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/' +str(k_folders_considered)+'test_l2_stepBYntestBY__TBYstep__'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'     +str(k_folders_considered)+'test_l2_stepBYntestBY__TBYstep__'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig4.savefig(strpicpdf, bbox_inches='tight')
                            fig4.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()
                            ax_4.set_yscale('log')
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/'+ str(k_folders_considered)+'test_l2_stepBYntestBY__TBYstep___ylog'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'    + str(k_folders_considered)+'test_l2_stepBYntestBY__TBYstep___ylog'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig4.savefig(strpicpdf, bbox_inches='tight')
                            fig4.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()

                            ax_5.set_xlabel('Epoch') #r'$y^+$'
                            ax_5.set_ylabel(legends_compnt[4])
                            ax_5.grid()
                            strpicpdf= './plots/diags/'  +filename_ii+'/pdf/'+ str(k_folders_considered)+'test_l2_fullBYntest'+'.pdf'
                            strpicpng= './plots/diags/'  +filename_ii+'/'    + str(k_folders_considered)+'test_l2_fullBYntest'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig5.savefig(strpicpdf, bbox_inches='tight')
                            fig5.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()
                            ax_5.set_yscale('log')
                            strpicpdf= './plots/diags/' +filename_ii+'/pdf/' +str(k_folders_considered)+'test_l2_fullBYntest_ylog'+'.pdf'
                            strpicpng= './plots/diags/' +filename_ii+'/'     +str(k_folders_considered)+'test_l2_fullBYntest_ylog'+'.png'
                            print('strpicpng=',strpicpng)
                            print('strpicpdf=',strpicpdf)
                            fig5.savefig(strpicpdf, bbox_inches='tight')
                            fig5.savefig(strpicpng, bbox_inches='tight')
                            #plt.show(block=False)
                            #timen.sleep(1)
                            pl.close()

                            pl.close('all')
plot_epochs()