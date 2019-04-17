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
    Sel_space = 0 # space margin {0 : ENC-Map}
    Sel_conv1_comp = 1  #  compressed layer type {0 : comp w/ conv1} {1 : comp w/o conv1}
    Sel_conv1 = 1  # initial rank of conv1 {0 : conv1 = from ENC-Map} {1 : conv1 = half} {2 : conv1 = max}
    Sel_comp = 0 # baseline complexity {0 : only conv} {1 : only conv (all cost)} {2 : all layers}
    Sel_norm = 1 # type of layer-wise accuracy metric {0 : PCA energy} {1 : validation accuracy}
    # -------------------------------------------------------------------------

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
    if Sel_norm == 0: ## PCA-energy
        Map_out =  ENC_Map(Ctarget[0],eigen_cumsum, Sel_comp, Sel_space, Sel_conv1, 0, Sel_norm)
        Rmax_ = np.asarray(Map_out[2][:])

    elif Sel_norm == 1: ## Measurement-based
        command = 'addpath({}); enc_map({}, {}, {}, {}, {}); exit;'.format("'../../utils'",Ctarget[0], Sel_space, Sel_comp, 1, Sel_conv1)
        subprocess.call(['matlab','-nodisplay','-nodesktop', '-nosplash', '-r', command])
        matlab_tmp = np.loadtxt('tmp_/MATLAB_result.txt')
        Rmax_ = matlab_tmp[:L]

    Rmax_ = np.array([min(Rmax_[i],Net[i,1]) for i in range(L)])
    R_Amax  = np.array([max(Rmax_[i],r10[i]) for i in range(L)])
    print 'Rmax : ',R_Amax.tolist()

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
    print('\niter = {}, Cost = {}({}), Weight = {}({})'.format(0, C_max, round(C_max*C_orig/1000000), W_max, round(W_max*W_orig/1000000)))
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


