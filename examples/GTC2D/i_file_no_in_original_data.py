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


def i_file_no_in_original(nx_r, nx_theta,theta_begn, theta_end,r_cntr, globl_r_end_estmt,globl_r_begn_estmt,phi,if_HyperDiffusivity_case,if_2ndRunHyperDiffusivity_case, n_beg):
    
    path = '/global/cfs/cdirs/m808/liuyangz/23_nlgki_eta0_moresnap/'
    i_file_no_in=[
            '0004000','0004100','0004200','0004300','0004400','0004500','0004600','0004700','0004800','0004900',
            '0005000','0005100','0005200','0005300','0005400','0005500','0005600','0005700','0005800','0005900',
            '0006000','0006100','0006200','0006300','0006400','0006500','0006600','0006700','0006800','0006900',
            '0007000','0007100','0007200','0007300','0007400','0007500','0007600','0007700','0007800','0007900', 
            '0008000','0008100','0008200','0008300','0008400','0008500','0008600','0008700','0008800','0008900', 
            '0009000','0009100','0009200','0009300','0009400','0009500','0009600','0009700','0009800','0009900', 
            '0010000','0010100','0010200','0010300','0010400','0010500','0010600','0010700','0010800','0010900', 
            '0011000','0011100','0011200','0011300','0011400','0011500','0011600','0011700','0011800','0011900', 
            '0012000','0012100','0012200','0012300','0012400','0012500','0012600','0012700','0012800','0012900', 
            '0013000','0013100','0013200','0013300','0013400','0013500','0013600','0013700','0013800','0013900', 
            '0014000','0014100','0014200','0014300','0014400','0014500','0014600','0014700','0014800','0014900', 
            '0015000','0015100','0015200','0015300','0015400','0015500','0015600','0015700','0015800','0015900', 
            '0016000','0016100','0016200','0016300','0016400','0016500','0016600','0016700','0016800','0016900', 
            '0017000','0017100','0017200','0017300','0017400','0017500','0017600','0017700','0017800','0017900', 
            '0018000','0018100','0018200','0018300','0018400','0018500','0018600','0018700','0018800','0018900', 
            '0019000',
            ]
    return i_file_no_in, path