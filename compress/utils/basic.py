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
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from itertools import groupby


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


def sigmoid2(x, a, b):
     y = a / (1 + np.exp(x/b - 6 ,dtype=np.float64))
     return y


def load_config(config_file):
    with open(config_file, 'r') as fp:
        conf = json.load(fp)
    return conf


def json_dump(data, i):
    json_write = './nin/conf/config.json.auto' + '_{}'.format(i)
    with open(json_write, 'w') as outfile:
            json.dump(data, outfile)

def read_file_blank(path):
    results = []
    with open(path) as inputfile:
        for line in  inputfile:
	    results.append(line.strip().split(' '))

    return results


def Eigen_cumsum_load(net_type, g):
    filename = './base/eigenvalue.conf'
    with open(filename, 'r') as fp:
        eigen_file = json.load(fp)
    layer = np.asarray(eigen_file["layer"].items())
    eigen = eigen_file["eigen"].items()
    layer_idx = np.array([np.where(layer[:,1].astype(int) == i) for i in range(len(layer))]).reshape(len(layer),)
    eigenvalue = np.array([np.array(eigen_file["eigen"][layer[layer_idx[i]][0]]) for i in range(len(layer_idx))])
   
    for i in range(len(g)):
        if g[i] != 1:
            eigenvalue[i] = (eigenvalue[i][0]+eigenvalue[i][1])/g[i]
    eigen_cumsum = [[np.cumsum(eigenvalue[i])] for i in range(len(eigenvalue))]

    return eigen_cumsum, eigenvalue


def Net_spec(Sel_comp):

    R = read_file_blank('base/R_list.txt')
    #R_norm = read_file_blank('base/R_norm.txt')
    C = read_file_blank('base/C_norm.txt')
    W = read_file_blank('base/W_norm.txt')
    A = read_file_blank('base/A_norm.txt')
    
    
    R = np.asarray(R,dtype=int)
    C = np.asarray(C,dtype=np.float)
    W = np.asarray(W,dtype=np.float)
    A = np.asarray(A,dtype=np.float)

    R = np.insert(R,0,0,axis=0)
    C = np.insert(C,0,0,axis=0)
    W = np.insert(W,0,0,axis=0)
    A = np.insert(A,0,0,axis=0)

    # Net : C(1)   R(2) T(3) g(4) Fm(5) K(6) Weight(7)	FLOPs(8)
    Net = read_file_blank('base/Net_def.txt')
    Net = np.asarray(Net,dtype=float)

    fc_num = len(Net) - len(np.nonzero(Net[:,6]-Net[:,7])[0])

    if Sel_comp == 2:
        L = np.shape(Net)[0]
    else:
        L = np.shape(Net)[0]-fc_num

    FC_cost = np.sum(Net[-fc_num:,6])
    Net = Net[:L]
    W_orig = np.sum(Net[:,6])
    C_orig = np.sum(Net[:,7])

    R_max = R[-1]
    R_norm = R.astype(float)/R_max
     

    # Weight
    Wmax = W[-1]
    W_norm = W.astype(float)/Wmax

    # FLOPs
    Cmax = C[-1]
    C_norm = C.astype(float)/Cmax

    
    return W_orig, C_orig, FC_cost, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C


def Check_Acc_Test(High_A, gpu_idx, gpu_num, net_type):

    command = './link_dir.sh {}'.format('test')
    os.system(command)
    os.system('rm tmp_/*')
    Sel_top = np.shape(High_A)[0]
    if Sel_top%gpu_num != 0 :
        Sel_top_tmp = Sel_top - Sel_top%gpu_num
    else :
        Sel_top_tmp = Sel_top

    L = np.shape(High_A)[1]
    filename = 'tmp_/final_result_test.txt'
    f2 = open(filename, 'a+')
    
    for i in range(Sel_top_tmp):     
        if i%(int(Sel_top_tmp/gpu_num)) == 0:
            filename = 'tmp_/final_result_now_{}.txt'.format(gpu_idx[int(i/(Sel_top_tmp/gpu_num))])
            f1 = open(filename, 'w')

        for j in range(L):
            f1.write('{} '.format(int(High_A[i,j])))
            f2.write('{} '.format(int(High_A[i,j])))
            #print int(High_A[i,j]),
        f1.write('\n')
        f2.write('\n')
        #print('\n')

    if Sel_top%gpu_num != 0:

        for i in range(Sel_top%gpu_num):     
            for j in range(L):
                f1.write('{} '.format(int(High_A[Sel_top_tmp+i,j])))
                f2.write('{} '.format(int(High_A[Sel_top_tmp+i,j])))
                #print int(High_A[i,j]),
            f1.write('\n')
            f2.write('\n')
            #print('\n')

    f1.close()
    f2.close()

#                os.system('./make_comp.sh')

    print('START')
    command = ''
    for i in range(gpu_num) :
        command = command + './make_comp.sh {} {} & '.format(gpu_idx[i], 'test')
    os.system(command)
    print('PASS')

    check = np.zeros(gpu_num)
    while(1) :
        time.sleep(10)
        for i in range(gpu_num):
            tmp = read_file_blank('check/gpu{}.txt'.format(gpu_idx[i]))
            check[i] = np.asarray(tmp[0][0],dtype=int)
        if sum(check) == gpu_num:
            break


    Rmax_tmp  = np.zeros([gpu_num,L])
    Acc_tmp = np.zeros(gpu_num)
    Rset_acc1  = np.zeros(len(High_A))
    Rset_acc5  = np.zeros(len(High_A))
    for i in range(gpu_num) :
        filename_set = 'tmp_/stage1_{}_{}.txt'.format(net_type, gpu_idx[i])
        set_conf = read_file_blank(filename_set)
        set_conf = np.asarray(set_conf, dtype=float)
        acc_top5 =  set_conf[:,3]
        acc_top1 =  set_conf[:,2]
        acc_top = acc_top1
        if i == gpu_num-1 :
            Rset_acc1[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)+len(High_A)%gpu_num] = acc_top1
            Rset_acc5[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)+len(High_A)%gpu_num] = acc_top5
        else :
            Rset_acc1[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)] = acc_top1
            Rset_acc5[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)] = acc_top5

    os.system('rm decomp_model')
    return Rset_acc1, Rset_acc5




def Check_Acc_Train(High_A, Sel_top, gpu_idx, gpu_num, net_type, Sel_acc):

    command = './link_dir.sh {}'.format('val')
    os.system(command)
    os.system('rm tmp_/*')

    if Sel_top%gpu_num != 0 :
        Sel_top_tmp = Sel_top - Sel_top%gpu_num
    else :
        Sel_top_tmp = Sel_top

    L = np.shape(High_A)[1]


    filename = 'tmp_/final_result_train.txt'
    f2 = open(filename, 'a+')
    
    for i in range(Sel_top_tmp):     
        if i%(int(Sel_top_tmp/gpu_num)) == 0:
            filename = 'tmp_/final_result_now_{}.txt'.format(gpu_idx[int(i/(Sel_top_tmp/gpu_num))])
            f1 = open(filename, 'w')

        for j in range(L):
            f1.write('{} '.format(int(High_A[i,j])))
            f2.write('{} '.format(int(High_A[i,j])))
        f1.write('\n')
        f2.write('\n')

    if Sel_top%gpu_num != 0:

        for i in range(Sel_top%gpu_num):     
            for j in range(L):
                f1.write('{} '.format(int(High_A[Sel_top_tmp+i,j])))
                f2.write('{} '.format(int(High_A[Sel_top_tmp+i,j])))
            f1.write('\n')
            f2.write('\n')

    f1.close()
    f2.close()


    print('START')
    command = ''
    for i in range(gpu_num) :
        command = command + './make_comp.sh {} {} & '.format(gpu_idx[i], 'val')
    os.system(command)
    print('PASS')

    check = np.zeros(gpu_num)
    while(1) :
        time.sleep(10)
        for i in range(gpu_num):
            tmp = read_file_blank('check/gpu{}.txt'.format(gpu_idx[i]))
            check[i] = np.asarray(tmp[0][0],dtype=int)
        if sum(check) == gpu_num:
            break


    Rmax_tmp  = np.zeros([gpu_num,L])
    Acc_tmp = np.zeros(gpu_num)
    Acc_tmp5 = np.zeros(gpu_num)
    Rset_acc1  = np.zeros(len(High_A))
    Rset_acc5  = np.zeros(len(High_A))
    idx_ = np.zeros(gpu_num)
    for i in range(gpu_num) :
        filename_set = 'tmp_/stage1_{}_{}.txt'.format(net_type, gpu_idx[i])
        set_conf = read_file_blank(filename_set)
        set_conf = np.asarray(set_conf, dtype=float)
        acc_top5 =  set_conf[:,3]
        acc_top1 =  set_conf[:,2]
        acc_top = acc_top5 if Sel_acc == 5 else acc_top1

        if i == gpu_num-1 :
            Rset_acc1[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)+len(High_A)%gpu_num] = acc_top1
            Rset_acc5[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)+len(High_A)%gpu_num] = acc_top5
        else :
            Rset_acc1[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)] = acc_top1
            Rset_acc5[int(len(High_A)/gpu_num)*i:int(len(High_A)/gpu_num)*(i+1)] = acc_top5

        max_val = max(acc_top)
        max_idx = min(np.nonzero((acc_top==max_val)>0)[0])
        Rmax_tmp[i,:] = High_A[max_idx+i*int(Sel_top/gpu_num), :L]
        Acc_tmp[i] = acc_top[max_idx]
        Acc_tmp5[i] = acc_top5[max_idx]
        idx_[i] = max_idx

    max_val = max(Acc_tmp)
    max_idx = min(np.nonzero((Acc_tmp==max_val)>0)[0])

    max_idx_ = max_idx*int(len(High_A)/gpu_num) + idx_[max_idx]

    acc_max = Acc_tmp[max_idx]
    acc_max5 = Acc_tmp5[max_idx]
    R_Amax = Rmax_tmp[max_idx]

#    High_A = High_A[max_idx]

    # ENC-Model 
    print '== [Train] ENC-Model : ',
    for i in range(L): print int(High_A[0][i]), 
    print Rset_acc1[0], Rset_acc5[0]
    
    print '== [Train] ENC-Inf : ',
    for i in range(L): print int(R_Amax[i]), 
    print acc_max, acc_max5, max_idx_

    os.system('rm decomp_model')
    return Rset_acc1, Rset_acc5, R_Amax, acc_max



