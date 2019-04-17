import numpy as np
import copy
import json
import os
import os.path as osp
import sys
import time
import itertools
import google.protobuf as pb
import random

from argparse import ArgumentParser
from pprint import pprint

import subprocess
from scipy import interpolate
from scipy.interpolate import interp1d
from itertools import groupby

#np.set_printoptions(threshold=np.nan)
sys.path.insert(0, osp.join('../../utils'))

from basic import *
from enc_map import *
from enc_model import *
from accuracy_metric import *


def Cost_Opt(Ctarget):
    # -------------------------------
    # -------------------------------
    #        Initial setting
    # -------------------------------
    # -------------------------------

    # Parameters --------------------------------------------------------------
    Sel_conv1_comp = 1  #  compressed layer type {0 : comp w/ conv1} {1 : comp w/o conv1}
    Sel_conv1 = 1  # initial rank of conv1 {0 : conv1 = from ENC-Map} {1 : conv1 = half} {2 : conv1 = max}
    Sel_comp = 0 # baseline complexity {0 : only conv} {1 : only conv (all cost)} {2 : all layers}
    Sel_norm = 1 # type of layer-wise accuracy metric {0 : PCA energy} {1 : validation accuracy}
    Sel_acc = 1 # selection crietria of accuracy metric, {1 : top-1 acc} {5 : top-5 acc}
    Sel_top = 40 # number of validation set {1 : ENC-Model} {over 2 : ENC-Inf}
    Sel_metric = 0 # type of whole-layer accuracy metric {0 : Ac - combination} {1 : Am - val acc} {2 : Ap - PCA energy}
    # -------------------------------------------------------------------------

    # Initial Settings for Combinatorial Space ---------------------
    num_sub_group = 1 # maximum number of sub-groups in a top group  (vgg16 : 1 / res56 : 4)
    Sel_space = 0.1 # space margin {0 ~ 1.0} {0 : ENC-Map}
    Sel_margin = 0.005 # complexity margin
    Sel_density = 0.165 # average density ratio {0.0 ~ 0.99}
    # --------------------------------------------------------------

    W_orig, C_orig, FC_cost, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C = Net_spec(Sel_comp)
    g = Net[:,3]
    a = Net[:,7]/Net[:,1]
    w = Net[:,6]/Net[:,1]
    r10 = np.around(Net[:,1]/10)
    r1_orig = [max(np.around(Net[i,1]/100),1) for i in range(len(Net))]

    # complexity setting
    c = a

    # Load the eigenvalue cumsum
    eigen_cumsum, eigenvalue = Eigen_cumsum_load(net_type, g)

    # -----------------------------------------------
    # ENC-Map : Single-shot Determination
    # -----------------------------------------------
    print 'Sel_space : ',Sel_space
    print '=========== ENC-Map  ==========='

    # --------------------------------------------------
    command = 'addpath({}); enc_map({}, {}, {}, {}, {}); exit;'.format("'../../utils'",Ctarget[0], Sel_space, Sel_comp, 1, Sel_conv1)
    subprocess.call(['matlab','-nodisplay','-nodesktop', '-nosplash', '-r', command])
    matlab_tmp = np.loadtxt('tmp_/MATLAB_result.txt')

    Rmax_ = matlab_tmp[:L]
    Rmin_ = matlab_tmp[L:L*2]

    Rmax_ = np.array([min(Rmax_[i],Net[i,1]) for i in range(L)])
    Rmax  = np.array([max(Rmax_[i],r10[i]) for i in range(L)])
    Rmin  = np.array([max(Rmin_[i],r10[i]) for i in range(L)])

    print 'Rmax : ',Rmax.tolist()
    C_acc = [Rmax.dot(a)/C_orig,Rmin.dot(a)/C_orig]
    W_acc = [Rmax.dot(w)/W_orig,Rmin.dot(w)/W_orig]
    print 'C_acc : {}, W_acc : {}'.format(C_acc, W_acc)
    # --------------------------------------------------

    if Sel_space == 0:
        return Rmax, C_acc[0], W_acc[0]

    # --------------------------------------------------
    # ENC-Model/ENC-Inf : Combinatorial Determination
    # --------------------------------------------------
    R_Amax = Rmax
    Cdelta0 = C_acc[0] - Ctarget[0]
    C_max = C_acc[0]
    C_max_init = C_acc[0]
    flag_end = 0 
    k = 0
    R_list = np.empty([20,L],dtype=float)
    C_list = np.empty([20,],dtype=float)
    A_list = np.empty([20,],dtype=float)
    no_group_flag = 0
    r1_new = r1_orig
    # ---------------------------------------------------------------

    # Top-layer groupping : for equal 'c'
    group_init, group = Layer_group_top(c)

    # hierarchical layer groupping - group_init, group_start
    group_hier = Layer_group_hier(num_sub_group, group_init)


    # =========================================
    # Start Generation of Candidate Rank-sets ! 
    # =========================================
    while(flag_end < 1):
        flag_end = 1
        Cdelta = Cdelta0
        Cdelta_abs = Cdelta*C_orig

        r_max_tmp = np.array([max(Rmax[i]-Rmin[i], 0) for i in range(L)])
        r_max = np.array([min(r_max_tmp[i], round(Cdelta_abs/c[i])) for i in range(L)]) #### Original
        r_max = r_max.astype(int)

        #------------------------------------------------
        print '\nSTEP(0) : Setting Space Interval'
        #------------------------------------------------
        alpha = Layer_density_iter(r1_orig, L, r_max, Sel_conv1_comp, Sel_density)
        r1_new = np.asarray(np.round(np.asarray(r1_orig, dtype=int)*alpha)).astype(int)
        r1_new = np.array([min(max(1, r1_new[i]),r_max[i]) for i in range(len(r1_new))])

        #for i in range(len(r1_new)): r1_new[i] = r1_new[i] if r_max[i] > 0 else 1
        print '--- r1_new : ',
        for i in range(len(r1_new)): print r1_new[i],
        print '\n--- r_max : ',
        for i in range(len(r_max)): print r_max[i],

        r1_new_tmp = r1_new
        for i in range(len(r1_new)): r1_new_tmp[i] = r1_new[i] if r_max[i] > 0 else 1
        x_range = [np.array([i for i in xrange(0,r_max[j]+1,r1_new_tmp[j])]) for j in range(L)]
        if Sel_conv1_comp == 1:
            x_range[0] = np.zeros(1,dtype=int)

        density_avgL = Layer_density_cal(r1_new, Sel_conv1_comp)
        print '\n--- layer density = {}'.format(density_avgL)

        #------------------------------------------------
        print('\nSTEP(1) : Folding level-1 rank-set')
        # -----------------------------------------------
        # mem_list generation
        set_idx_group = np.array([ i for i in range(len(group_init)) if len(group_init[i]) > 1 ])
        set_idx_group, mem_list = Table_gen(set_idx_group, r_max, group_init, group, x_range, r1_new)
        print('--- table generation [done]')    
        sys.stdout.flush()

        #------------------------------------------------
        print('\nSTEP(2) : Folding level-2 & level-3 rank-set')
        sys.stdout.flush()
        # -----------------------------------------------
        range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot = Folding_layers(set_idx_group, num_sub_group, group_hier, r_max, r1_new, x_range)

        #------------------------------------------------
        print('\nSTEP(3) : Extraction of candidate sets')
        sys.stdout.flush()
        # -----------------------------------------------
        # make the top-level rank-sets
        X1, c_group = Top_layer_set(group_init, set_idx_group, x_range, a, w)
        print('--- top layer generation [done]')    
        sys.stdout.flush()

        # extract the candidate sets satisfying target complexity in the folded layers
        print '--- mem_list shape = ',np.shape(mem_list)
        Y_ = Extract_set_in_top_level_layer(num_sub_group, no_group_flag, mem_list, X1, set_idx_group, Sel_margin, Cdelta_abs, c_group, group_init, group_hier, range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Sel_metric , Ctarget[0], Sel_comp)
        sys.stdout.flush()

        # skip to (STEP 4) : the calculation of accuracy metric 
        if no_group_flag == 1:
            Y_ = Y
            print '-- (no groupping) : skip to "STEP 4"'
            break

    #------------------------------------------------
    print('\nSTEP(4) : Calculation of whole-layer accuracy metric')
    # -----------------------------------------------
    sys.stdout.flush()
    Y = Y_
    print np.shape(Y)
    print Rmax
    R_txt = (Rmax-Y).astype(np.int)

    if Sel_metric == 0:
        EQ1 = Calculate_Ac_matlab(Rmax, R_txt, eigenvalue, eigen_cumsum, group, Sel_comp)
    elif Sel_metric == 1:
        EQ1 = Calculate_Am_matlab(Rmax, R_txt, eigenvalue, eigen_cumsum, group, Sel_comp)
    else:
        EQ1 = Calculate_Ap(Rmax, R_txt, eigenvalue, eigen_cumsum, group, Sel_comp)

    Set = np.stack((R_txt.dot(a), R_txt.dot(w), EQ1), axis=-1)
    Set = np.concatenate((R_txt, Set),axis=1)
    Set_opt  = Set[Set[:,L+2].argsort()[::-1]]

    print('[TIME] : -- before accuracy check')
    toc()
    sys.stdout.flush()

    # ENC-Inf : maximum value of validation accuracy
    # -------------------------------------------------
    if Sel_top > 1 :
        High_A = Set_opt[:Sel_top]
        Rset_acc1, Rset_acc5, R_Amax, acc_max = Check_Acc_Train(High_A[:,:L], Sel_top, gpu_idx, gpu_num, net_type, Sel_acc)

    # ENC-Model : maximum value of whole-layer accuracy metric
    # ---------------------------------------------------------
    else :
        High_A = Set_opt[0]
        R_Amax = High_A[:L]
        acc_max = 0

    # Complexity of fianl rank configuration
    if Sel_comp == 1:
        C_max = (R_Amax.dot(a) + FC_cost)/C_orig
        W_max = (R_Amax.dot(w) + FC_cost)/W_orig
    else:
        C_max = (R_Amax.dot(a))/C_orig
        W_max = (R_Amax.dot(w))/W_orig

    print('\nFINAL Rank Configuration : ')
    for i in range(len(R_Amax)):
         print int(R_Amax[i]),
    print('\niter = {}, Rsize = {}, R_sel = {}, C_delta = {}, Cost = {}({}), Weight = {}({})'.format(k, len(Y), Sel_top, Cdelta, C_max, round(C_max*C_orig/1000000), W_max, round(W_max*W_orig/1000000)))

    print('\nEND ------- ')

    return R_Amax, C_max, W_max
 

def main(args):     
                    
    tmp = Net_spec(1)
    L = np.shape(tmp[8])[1]
    k = 0

    global net_type
    global gpu_num
    global gpu_idx

    # ----------------------------------------
    # Parameters
    # ----------------------------------------
    gpu_idx = map(int, list(args.gpu.split(','))) # available GPU indice
    #c_list = [0.248, 0.249, 0.25, 0.251, 0.252] # target complexity
    c_list = [float(args.tar_comp)] # target complexity
    # ----------------------------------------
    
    net_type = args.type
    gpu_num = len(gpu_idx)
    R_list = np.zeros([len(c_list),L])
    C_list = np.zeros(len(c_list))
    W_list = np.zeros(len(c_list))    

    # Netowrk Compression -----------
    for i in range(len(c_list)) :
        tic()
        Ctarget = [c_list[i], c_list[i]]
        R_list[k,:], C_list[k], W_list[k] = Cost_Opt(Ctarget)
        print('[Done] : Netsork Compression for complexity {} - {}'.format(c_list[i], i))
        toc()
        k += 1

    R_list = R_list[:k]
    C_list = C_list[:k]
    W_list = W_list[:k]

    # Accuracy check ---------------
    len_R = min(gpu_num, k)
    gpu_idx = gpu_idx[:len_R]
    gpu_num = len(gpu_idx) 
    A_val  = np.zeros([k, 2])
    A_test = np.zeros([k, 2])

    tmp = Check_Acc_Train(R_list, k, gpu_idx, gpu_num, net_type, 1)
    A_val[:,0] = tmp[0]
    A_val[:,1] = tmp[1]
    A_test[:,0], A_test[:,1] = Check_Acc_Test(R_list, gpu_idx, gpu_num, net_type)

    # File write (final result) --------
    filename = 'final_result_summary.txt'
    f = open(filename, 'a+')
    for i in range(len(R_list)):
        for j in range(L):
           f.write('{} '.format(R_list[i,j]))
        f.write('{} '.format(C_list[i]))
        f.write('{} '.format(W_list[i]))
        f.write('{} '.format(A_val[i][0]))
        f.write('{} '.format(A_val[i][1]))
        f.write('{} '.format(A_test[i][0]))
        f.write('{}\n'.format(A_test[i][1]))

  
if __name__ == '__main__':
    parser = ArgumentParser(description="Network Compression")
    parser.add_argument('--type')
    parser.add_argument('--tar_comp', help="compression rate, range: (0.0 ~ 1.0)")
    parser.add_argument('--gpu', help="avaiable gpu indice, ex) [0,2,3]")

    args = parser.parse_args()
    main(args)


