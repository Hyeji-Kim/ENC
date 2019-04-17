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

from basic import *


def make_unique_points(L, E_norm, R_norm):
    E_norm_tmp = []
    R_norm_tmp = []
    E_idx = []
    for i in range(L):
        #print '0',E_norm[:,i], R_norm[:,i]
        idx_ = np.sum(E_norm[:,i]<1)+1
        E_norm_ = E_norm[:idx_,i]
        #print '1',E_norm_
        R_norm_ = R_norm[:idx_,i]
    
        E_norm_, idx_ = np.unique(E_norm_,return_index=True)
        R_norm_ = R_norm_[idx_]
        #print '2',E_norm_
        R_norm_, idx_ = np.unique(R_norm_,return_index=True)
        E_norm_ = E_norm_[idx_]
        #E_norm_[-1] = 1
        #print '3',E_norm_

        diff_E = np.array([E_norm_[k+1]-E_norm_[k] for k in range(len(E_norm_)-1)])
        diff_mask = (diff_E <= 0)

        tmp_a = E_norm_[1:]
        tmp_a = tmp_a[~diff_mask]
        tmp_r = R_norm_[1:]
        tmp_r = tmp_r[~diff_mask]


        idx_ = len(E_norm_) - np.sum(diff_mask)
#        E_norm_ = [E_norm_[0]].extend(tmp_a)
#        R_norm_ = [R_norm_[0]].extend(tmp_r)
        [E_norm_[0]].extend(tmp_a)
        [R_norm_[0]].extend(tmp_r)

        #print '4',E_norm_, R_norm_

        E_idx = E_idx + [idx_.tolist()]
        E_norm_tmp = E_norm_tmp + [E_norm_]
        R_norm_tmp = R_norm_tmp + [R_norm_]
        #sys.stdout.flush()
        #aaa = input()

    return E_idx,  E_norm_tmp, R_norm_tmp


def make_R(L, E_norm, R_norm_tmp, R, Amax):
    Rmax = np.empty(L,dtype=int)
    for i in range(L):
        x = E_norm[i]
        y = R_norm_tmp[i]
        Rmax[i] = np.round(interpolate.pchip_interpolate(x, y, Amax)*R[-1,i])
    return Rmax


def make_R_from_unique_points(L, E_norm, E_idx, R_norm_tmp, R, Amax):
    Rmax = np.empty(L,dtype=int)
    for i in range(L):
#        R_norm_tmp[E_idx[i],i]=1
        x = E_norm[:E_idx[i]+1,i]
        y = R_norm_tmp[:E_idx[i]+1,i]

        # refinement norm
#        if y[-1] < 1:
#            x[-1] = (1 - x[-2])*0.99 + x[-2]
#            x = np.append(x,1)
#            y = np.append(y,1)

        x, x_idx = np.unique(x,return_index=True)
        y = y[x_idx]
        y, y_idx = np.unique(y,return_index=True)
        x = x[y_idx]
        Rmax[i] = np.round(interpolate.pchip_interpolate(x, y, Amax)*R[-1,i])
    return Rmax

def Refinement_complexity(C_acc, Ctarget, space, delta):
    flag_end = 1
    if C_acc <= Ctarget+space:
        delta = delta+0.001
    else :
        delta = delta-0.001
    if (C_acc < (Ctarget +space + 0.001)) & ((Ctarget +space - 0.001) < C_acc) :
        flag_end = 0

    return delta, flag_end

def Calculate_C(r_min_vbmf, c, Sel_conv1, Sel_comp, VGG16_fc, C_orig):
    Test_R = np.copy(r_min_vbmf)
    Test_C = Test_R*c #Test_R.dot(c)

    if Sel_conv1 != 0:
        Test_C[:,0] = 0

    C_acc = np.sum(Test_C,axis=1,dtype=np.float64)
    if Sel_comp == 1:
        C_acc = (C_acc + VGG16_fc)/C_orig
    else :
        C_acc = (C_acc)/C_orig

    return C_acc


def Ro_determination(C_acc_list, a_norm, Ctarget, space, E_norm, E_idx, R_norm_tmp, R, c, Sel_conv1, Sel_comp, VGG16_fc, C_orig, L):

    flag_end = 1
    delta = 0
    k = 0
    while (flag_end>0) :
        # layer-wise accuracy metric
        Amax = interpolate.pchip_interpolate(C_acc_list, a_norm, Ctarget+space+delta)
        if Ctarget+space+delta > 1:
            Amax = 1.0

        # Ro determination
        #Rmax = make_R_from_unique_points(L, E_norm, E_idx, R_norm_tmp, R, Amax)
        #for i in range(len(R_norm_tmp)): R_norm_tmp[i][-1] = 1
        Rmax = make_R(L, E_norm, R_norm_tmp, R, Amax)

        # complexity
        Test_C = Rmax*c
        if Sel_conv1 != 0:
            Test_C[0] = 0

        if Sel_comp == 1:
            C_acc = (np.sum(Test_C,dtype=np.float64) + VGG16_fc)/C_orig
        else :
            C_acc = (np.sum(Test_C,dtype=np.float64))/C_orig

        # refinement of complexity
        delta, flag_end = Refinement_complexity(C_acc, Ctarget, space, delta)
        k = k+1
        if k > 500 :
            break
        
    return Rmax

def ENC_Map(Ctarget,eigen_cumsum, Sel_comp, space, Sel_conv1, Sel_cost, Sel_norm):

    # losd network specification
    W_orig, C_orig, VGG16_fc, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C = Net_spec(Sel_comp)

    N = np.shape(R_norm)[0]

    R = R[:,:L]
    C = C[:,:L]
    W = W[:,:L]
 
    a = Net[:,7]/Net[:,1]
    w = Net[:,6]/Net[:,1]


    # complexity setting ----------
    C_conv1 = Net[0,7]  
    c = a

    if Sel_conv1 == 1:
        C_conv1 = C_conv1/2

    if Sel_conv1 != 0:
        Ctarget = Ctarget - C_conv1/C_orig

    # make E norm ------------
    E_norm = np.empty([np.shape(R_norm)[0],L],dtype=float)
    A_norm = A.astype(float)/A[-1]

    for i in range(L):
        if Sel_norm == 0 :
            E_norm[:,i] = np.array([(eigen_cumsum[i][0][R[n,i]-1]-eigen_cumsum[i][0][0])/(eigen_cumsum[i][0][-1]-eigen_cumsum[i][0][0]) for n in range(N)])
        else :
            E_norm[:,i] = A_norm[:,i]
        E_norm[0,i] = 0

    # ====================================
    # 1. Calculate the Re
    # ====================================
    a_norm = np.arange(0.1, 1.01, 0.01)
#    R_norm_tmp = np.copy(R_norm)
    r_min_vbmf = np.empty([len(a_norm),L],dtype=int)

#    E_mask = E_norm<1
#    E_idx = np.sum(E_mask,0)

    E_idx, E_norm_tmp, R_norm_tmp = make_unique_points(L, E_norm, R_norm)
    for n in range(len(a_norm)):
        #r_min_vbmf[n] = make_R_from_unique_points(L, E_norm, E_idx, R_norm_tmp, R, a_norm[n])
        r_min_vbmf[n] = make_R(L, E_norm_tmp, R_norm_tmp, R, a_norm[n])

    # ====================================
    # 2. Calculate the complexity from Re
    # ====================================
    C_acc = Calculate_C(r_min_vbmf, c, Sel_conv1, Sel_comp, VGG16_fc, C_orig)
    print max(C_acc), Ctarget

    #aaa = input()
    # ===
    if Ctarget > max(C_acc):
        for i in range(len(R_norm_tmp)): R_norm_tmp[i][-1] = 1
        for n in range(len(a_norm)):
            r_min_vbmf[n] = make_R(L, E_norm_tmp, R_norm_tmp, R, a_norm[n])

        C_acc = Calculate_C(r_min_vbmf, c, Sel_conv1, Sel_comp, VGG16_fc, C_orig)


    # ====================================
    # 3. Ro Determination
    # ====================================
    a_norm = np.insert(a_norm,0,[0])
    C_acc_list = np.insert(C_acc,0,[0])
    C_acc_out = np.zeros(2)
    W_acc_out = np.zeros(2)

    C_acc_list, c_idx = np.unique(C_acc_list,return_index=True)
    a_norm = a_norm[c_idx]
    Rmax = Ro_determination(C_acc_list, a_norm, Ctarget, space, E_norm_tmp, E_idx, R_norm_tmp, R, c, Sel_conv1, Sel_comp, VGG16_fc, C_orig, L)
    Rmin = Ro_determination(C_acc_list, a_norm, Ctarget, -space, E_norm_tmp, E_idx, R_norm_tmp, R, c, Sel_conv1, Sel_comp, VGG16_fc, C_orig, L)

    if Sel_conv1 == 1 : # conv1 - half
        Rmax[0] = round(Net[0,1]/2)
        Rmin[0] = round(Net[0,1]/2)
#        C_acc_out[0] = Rmax.dot(a)/C_orig
#        C_acc_out[1] = Rmin.dot(a)/C_orig
#        C_acc_out = C_acc_out + C_conv1/C_orig

    elif Sel_conv1 == 2 : # conv1 - max
        Rmax[0] = round(Net[0,1])
        Rmin[0] = round(Net[0,1])
#        C_acc_out[0] = Rmax.dot(a)/C_orig
#        C_acc_out[1] = Rmin.dot(a)/C_orig
#        C_acc_out = C_acc_out + C_conv1/C_orig

    C_acc_out[0] = (Rmax.dot(a))/C_orig
    C_acc_out[1] = (Rmin.dot(a))/C_orig
    W_acc_out[0] = (Rmax.dot(w))/W_orig
    W_acc_out[1] = (Rmin.dot(w))/W_orig



    print('C range : {},   W range : {}'.format(C_acc_out, W_acc_out))
    print'Initial Rmax : ',
    for i in range(len(Rmax)):
         print int(Rmax[i]),


    print'\nInitial Rmin : ',
    for i in range(len(Rmin)):
         print int(Rmin[i]),

    print('\n')


    return C_acc_out, W_acc_out, Rmax, Rmin



