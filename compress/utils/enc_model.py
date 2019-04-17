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
#from caffe.proto.caffe_pb2 import NetParameter, LayerParameter


import subprocess
from scipy import interpolate
from scipy.interpolate import interp1d
from itertools import groupby

from basic import *

def Layer_group_top(c):
    a_idx_tmp = [len(list(j)) for i, j in groupby(c)]
    a_idx_tmp_cum =np.cumsum(a_idx_tmp).tolist()
    a_idx_tmp_cum = [0] + a_idx_tmp_cum[:-1]
    group_init = np.array([list(np.arange(a_idx_tmp[i])+a_idx_tmp_cum[i]) for i in range(len(a_idx_tmp))])
    #print 'c : ',c
    print 'group_init : ',group_init
    group = np.hstack(group_init.flat)

    return group_init, group


def Layer_group_hier(num_sub_group, group_init):
    group_hier = []
    for i in range(len(group_init)) :
        if len(group_init[i]) == 1 :
            tmp = group_init[i]
        else :
            if len(group_init[i])%int(len(group_init[i])/num_sub_group) != 0 :
                tmp_mod = len(group_init[i])%int(len(group_init[i])/num_sub_group)
                tmp = [zip(*[iter(group_init[i])]*int(len(group_init[i])/num_sub_group)) + [group_init[i][-tmp_mod:]]]
            else :
                tmp = [zip(*[iter(group_init[i])]*int(len(group_init[i])/num_sub_group))]

        group_hier = group_hier + tmp
    print 'group_hierarchy : ',group_hier
    return group_hier


# ============================
# ============================
# ============================
def Layer_density(r1_orig, tmp_alpha, L, r_max):
    r1_orig_tmp = np.round(np.asarray(r1_orig, dtype=int)*tmp_alpha)
    r1_orig_tmp = [max(r1_orig_tmp[i],1) for i in range(L)]
    
    r_max_tmp = np.copy(r_max)
    r_max_tmp = r_max_tmp[r_max_tmp>0]

    r1_new_tmp =  np.asarray(r1_orig_tmp,dtype=int)
    density_avgL = np.mean(np.divide(1.,r1_new_tmp))

    return density_avgL

def Layer_density_cal(r1_new, Sel_conv1_comp):
    if Sel_conv1_comp == 1:
        density_avgL = np.mean(np.divide(1.,np.asarray(r1_new,dtype=int)))
    else:
        density_avgL = np.mean(np.divide(1.,np.asarray(r1_new[1:],dtype=int)))

    return density_avgL


def Layer_density_iter(r1_orig, L, r_max, Sel_conv1_comp, C_density):
    density_avgL = Layer_density_cal(r1_orig, Sel_conv1_comp)
    tmp_alpha = density_avgL/C_density

    while density_avgL <= C_density :
        tmp_alpha = tmp_alpha*0.99
        if Sel_conv1_comp == 1:
            density_avgL = Layer_density(r1_orig[1:], tmp_alpha, L-1, r_max[1:])
        else:
            density_avgL = Layer_density(r1_orig, tmp_alpha, L, r_max)

    #layer_ratio = np.divide(x_range_num,r_max.astype(float))
    return tmp_alpha






# ======================================================
# ======================================================
# ======================================================
def Unfolded_mem_list(r_max, group_init, group, x_range):
    set_max_x = np.array([max(r_max[group_init[i]]) for i in group])
    set_max   = np.array(max(set_max_x))
    set_max_idx_x = np.where(set_max_x == set_max)[0]
    set_idx_group = set_max_idx_x
    mem_list = np.array(x_range[set_max_idx_x[0]],ndmin=2).T

    return set_idx_group, mem_list

def Folded_layer_mem_list(r_max, group_init, r1_new):
    set_idx_group = np.array([ i for i in range(len(group_init)) if len(group_init[i]) > 1 ])
    set_iter_sum_orig = np.array([np.arange(0,sum(r_max[group_init[i]])+1,min(r1_new[group_init[i]]))  for i in set_idx_group ])
    mem_list = np.asarray(list(itertools.product(*set_iter_sum_orig)))

    return set_idx_group, mem_list


def Table_gen(set_idx_group, r_max, group_init, group, x_range, r1_new):
    # Unfolded layers
    if set_idx_group.size == 0:
        no_group_flag = 1
        set_idx_group, mem_list = Unfolded_mem_list(r_max, group_init, group, x_range)

    # Folded layers
    else :
        set_idx_group, mem_list = Folded_layer_mem_list(r_max, group_init, r1_new)

    return set_idx_group, mem_list        
 

# ======================================================
# ======================================================
def Top_layer_set(group_init, set_idx_group, x_range, a, w):
    group_start = np.array([group_init[i][0] for i in range(len(group_init))])
    group_idx_element = np.array([group_init[i][0] for i in set_idx_group])
    set_iter_total = np.array([ [0] if group_start[i] in group_idx_element  else x_range[group_start[i]]  for i in range(len(group_start))])
    X1 = list(itertools.product(*set_iter_total))
    X1 = np.asarray(X1)

    c_group = np.array([ a[i] for i in group_start ])

    return X1, c_group


def Extract_set_in_top_level_layer(num_sub_group, no_group_flag, mem_list, X1, set_idx_group, Th_h, Cdelta_abs, c_group, group_init, group_hier, range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Sel_metric, Ct, Sel_comp):
    idx = 0
    stage2 = 0
    Y = []
    Y_ = []
    flag = 0
    step_m = 20
    step_y = 100
    sel_num = 10
    # NOTE: To further accelerate the network compression, process this part in multi-threads.
    if num_sub_group > 1:
        mem_list = mem_list[::step_m,:] if len(mem_list) > step_m else mem_list[::2,:]
    for m in range(len(mem_list)):
        x_range_sel = mem_list[m]
        Y1 = Filtering_with_target_complexity(X1, set_idx_group, x_range_sel, Th_h, Cdelta_abs, c_group, group_init)

        if len(Y1) != 0:
            if num_sub_group > 1:
                Y1 = Y1[::step_y,:] if len(Y1) > step_y else Y1[::2,:]

            if no_group_flag == 1:
                Y1 = Unfolded_layer_recover(x_range_sel, group_init, set_idx_group, Y1)

            sys.stdout.flush()
            uY, uIdx = np.unique(Y1[:,set_idx_group+1], axis=0, return_inverse=True)
            # Case 1 : Over 2-level sub-groups
            if num_sub_group > 1:
                Y_ = Make_sets_over_2_level_sub(Y1, uY, uIdx, group_init, group_hier, set_idx_group, range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Sel_metric, Ct, sel_num)
            # Case 2 : Single sub-group
            else :
                Y_ = Make_sets_single_level_sub(Y1, uY, uIdx, group_init, group_hier, set_idx_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Sel_metric, 1, Ct)

        sys.stdout.flush()
        if len(Y_) != 0:
            # concetenate Y
            if idx == 0:
                Y = Y_
            else:
                Y = np.concatenate((Y,Y_), axis=0)
            idx = 1

            if len(Y) > 600000 :
                Y = np.unique(Y, axis=0)
                if Sel_metric == 0:
                    Y = Ac_top(10000, Rmax, Y, eigen_cumsum, R_norm, A,  R, C_orig, c, L)
                elif Sel_metric == 1:
                    Y = Am_top(10000, Rmax, Y, eigen_cumsum, R_norm, A,  R, C_orig, c, L)
                elif Sel_metric == 2:
                    Y = Ap_top(10000, Rmax, Y, eigen_cumsum, R_norm, R, C_orig, c, L)


        if m%100 == 0:
            print '----- m: {}/{} set: {}'.format(m,len(mem_list), len(Y))

        # print statust
        #stage2 = Print_status(stage2, len(mem_list), m, len(Y)) if flag > 0 else Print_status(stage2, len(mem_list), m, len(Y_))
        #sys.stdout.flush()

    Y = np.unique(Y, axis=0)
    return Y


def Extract_set_in_top_level(num_sub_group, no_group_flag, mem_list, X1, set_idx_group, Th_h, Cdelta_abs, c_group, group_init, num):
    if num_sub_group > 1:
        Y = Extract_set_in_top_level_sub_layer(no_group_flag, mem_list, X1, set_idx_group, Th_h, Cdelta_abs, c_group, group_init, num)
    else:
        Y = Extract_set_in_top_level_single_layer(no_group_flag, mem_list, X1, set_idx_group, Th_h, Cdelta_abs, c_group, group_init, num)

    return Y



def Filtering_with_target_complexity(X1, set_idx_group, x_range_sel, Th_h, Cdelta_abs, c_group, group_init):
    X1[:,set_idx_group] = x_range_sel
    Y1 = np.zeros([len(X1),len(group_init)])
    Y1 = X1.dot(c_group)
    Y1.shape = (len(Y1),1)
    Y1 = np.concatenate((Y1,X1), axis=1)

    # filtter with complexity
    #-------------------------------
    y = Cdelta_abs
    Y1 = Y1[(Y1[:,0] < y*(1+Th_h))&(Y1[:,0] > y*(1-Th_h))].astype(int)

    return Y1


def Unfolded_layer_recover(x_range_sel, group_init, set_idx_group, Y1):
    set_iter_orig = np.array([x_range_sel],ndmin=3)
    setx_sum = np.array([set_iter_orig[i].dot(np.ones([len(group_init[set_idx_group[i]]),1])) for i in range(len(set_idx_group))])
    setx = np.array([np.concatenate((setx_sum[i], set_iter_orig[i]), axis=1).astype(int) for i in range(len(set_idx_group))])
    setx = [setx[i][setx[i][:,0] == x_range_sel[i]] for i in range(len(set_idx_group))]
    setx = [setx[i][:,1:] for i in range(len(set_idx_group))]

    setx_iter = np.array(list(itertools.product(*setx)))
    setx_repeat = np.array([np.tile(setx_iter[i],(len(Y1),1)) for i in range(len(setx_iter))]).reshape(len(setx_iter)*len(Y1),setx_iter.shape[1])

    Y1 = np.tile(Y1,(len(setx_iter),1))
    Y1 = np.delete(Y1,set_idx_group+1,axis=1)

    tmp = np.array([setx_repeat[:,0][j] for j in range(len(Y1))],ndmin=2).T
    Y1 = np.insert(Y1,[group_idx_element[0]+1],tmp,axis=1)

    return Y1


def Print_status(stage2, len_mem_list, m, len_Y1):
    stage1 = np.copy(stage2)
    if len_mem_list < 11 :
        stage2 = m
    else:
        stage2 = round(m/((len_mem_list-1)/10))
    if (stage2-stage1) != 0:
        print '{}%({})'.format(int(stage2)*10, len_Y1),

    return stage2


# ===============================================
# ===============================================
# ===============================================

def Over_2_level_group_gen(group_hier, set_idx_, r_max, r1_new, x_range):
    range_2 = []
    setx_iter_3_sum_ = []
    setx_iter_3_tmp = []
    range_3_ = []

    for i in range(len(group_hier[set_idx_])) :  # (2,3,4) (5,6,7) .. 
        idx = np.asarray(group_hier[set_idx_][i])
        mask = r1_new[idx]>0
        r1_new_mask = r1_new[idx[mask]]
        tmp = range(0,sum(r_max[np.asarray(group_hier[set_idx_][i])]) + 1, min(r1_new_mask))
 #       tmp = range(0,sum(r_max[np.asarray(group_hier[set_idx_][i])]) + min(r1_new[np.asarray(group_hier[set_idx_][i])]), min(r1_new[np.asarray(group_hier[set_idx_][i])]))
#        tmp = range(0,sum(r_max[np.asarray(group_hier[set_idx_][i])])+r1_new[np.asarray(group_hier[set_idx_][i][0])], r1_new[np.asarray(group_hier[set_idx_][i][0])])
        range_2 = range_2 + [tmp]

        # --- Bottom level ----
        range_3 = [x_range[i3] for i3 in group_hier[set_idx_][i]] # i3 : index of stage-3 pf stage-2 of [j]
        setx_iter_3_ = np.array(list(itertools.product(*range_3)))
        setx_iter_3_sum = np.sum(setx_iter_3_,axis=1)

        setx_iter_3_tmp = setx_iter_3_tmp + [setx_iter_3_]
        setx_iter_3_sum_ = setx_iter_3_sum_ + [setx_iter_3_sum]
        range_3_ = range_3_ + [range_3]
    return range_2, range_3_, setx_iter_3_sum_, setx_iter_3_tmp

def Single_level_group_gen(group_hier, set_idx_, r_max, r1_new, x_range):
    range_2 = []
    setx_iter_3_sum_ = []
    setx_iter_3_tmp = []
    range_3_ = []

    idx = np.asarray(group_hier[set_idx_][0])
    mask = r1_new[idx]>0
    r1_new_mask = r1_new[idx[mask]]

    #tmp = range(0,sum(r_max[np.asarray(group_hier[set_idx_][0])])+1, min(r1_new_mask))
    tmp = range(0,sum(r_max[np.asarray(group_hier[set_idx_][0])])+min(r1_new_mask), min(r1_new_mask))
    range_2 = [tmp]

    # --- Bottom level ----
    range_3 = [x_range[i3] for i3 in group_hier[set_idx_][0]] # i3 : index of stage-3 pf stage-2 of [j]
    setx_iter_3_ = np.array(list(itertools.product(*range_3)))
    setx_iter_3_sum = np.sum(setx_iter_3_,axis=1)

    setx_iter_3_tmp = setx_iter_3_tmp + [setx_iter_3_]
    setx_iter_3_sum_ = setx_iter_3_sum_ + [setx_iter_3_sum]
    range_3_ = range_3_ + [range_3]

    return range_2, range_3_, setx_iter_3_sum_, setx_iter_3_tmp 

def Folding_layers(set_idx_group, num_sub_group, group_hier, r_max, r1_new, x_range):

    range_2_tot = []
    setx_iter_2_tot = []
    setx_iter_2_sum_tot = []

    range_3_tot = []
    setx_iter_3_tot = []
    setx_iter_3_sum_tot = []

    # NOTE: To further accelerate the network compression, process this part in multi-threads.
    #---------------------------------------------------------------------
    for j in range(len(set_idx_group)) : # 2,4,7 (for a top group)

        # ---- Level-2 ----
        set_idx_ = set_idx_group[j]

        # Case 1 : Over 2-level sub-groups
        # -------------------------------
        if num_sub_group > 1:
            range_2, range_3_, setx_iter_3_sum_, setx_iter_3_tmp = Over_2_level_group_gen(group_hier, set_idx_, r_max, r1_new, x_range)
            setx_iter_2_tmp = np.array(list(itertools.product(*range_2)))

        # Case 2 : Single sub-group in a top-group
        # -----------------------------------------
        else :
            range_2, range_3_, setx_iter_3_sum_, setx_iter_3_tmp = Single_level_group_gen(group_hier, set_idx_, r_max, r1_new, x_range)
            setx_iter_2_tmp = range_2


        setx_iter_2_tot = setx_iter_2_tot + [setx_iter_2_tmp]
        setx_iter_2_sum_tot = setx_iter_2_sum_tot + [np.sum(setx_iter_2_tmp,axis=1)]
        range_2_tot = range_2_tot + [range_2]

        setx_iter_3_tot = setx_iter_3_tot + [setx_iter_3_tmp]
        setx_iter_3_sum_tot = setx_iter_3_sum_tot + [setx_iter_3_sum_]
        range_3_tot = range_3_tot + [range_3_]

    return range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot

# ===============================================
# ===============================================
# ===============================================


def Ac_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, num, Ct):
    R_set = Rmax[R_idx] - R_tmp
    eigen_sum = np.zeros(np.shape(R_tmp))
    acc_r_sum = np.zeros(np.shape(R_tmp))

    for i3 in range(np.shape(R_tmp)[1]) :
        ig = g_idx[i3]
        eigen_tmp = np.asarray(eigen_cumsum[ig][0][:])
        eigen_sum[:,i3] = (eigen_tmp[np.array(R_set[:,i3],dtype=int)-1]-eigen_cumsum[ig][0][0])/(eigen_cumsum[ig][0][-1]-eigen_cumsum[ig][0][0])
        acc_r_sum[:,i3] = interpolate.pchip_interpolate(R_norm[:,ig], A[:,ig], R_set[:,i3]/R[-1,ig].astype(float))

    #C_tmp = np.sum( R_set/R[-1][np.asarray(g_idx)], axis=1)

    C_tmp = R_set.astype(float).dot(a[np.asarray(g_idx)])/C_orig

    #C_max = R[-1][np.asarray(g_idx)].dot(a[np.asarray(g_idx)])
    #C_tmp = R_set.astype(float).dot(a[np.asarray(g_idx)])/C_max

    eigen_prod = np.multiply(np.prod(eigen_sum,axis=1),C_tmp)

    #eigen_prod = np.prod(eigen_sum,axis=1)*Ct

    eigen_prod = np.prod(acc_r_sum,axis=1) + eigen_prod

    eigen_prod2  = eigen_prod[eigen_prod.argsort()[::-1]]
    #tmp_num = min(min(max(5,int(len(eigen_prod)*0.01)),len(R_tmp)),num)
    tmp_num = min(len(R_tmp),num)
    tmp_val = eigen_prod2[tmp_num-1]
    tmp_idx = eigen_prod>=tmp_val
    R_ = R_tmp[tmp_idx]

    return R_


def Am_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, num):
    R_set = Rmax[R_idx] - R_tmp
    acc_r_sum = np.zeros(np.shape(R_tmp))

    for i3 in range(np.shape(R_tmp)[1]) :
        ig = g_idx[i3]
        acc_r_sum[:,i3] = interpolate.pchip_interpolate(R_norm[:,ig], A[:,ig], R_set[:,i3]/R[-1,ig].astype(float))

    #C_tmp = R_set.dot(a[np.asarray(g_idx)])/C_orig
    eigen_prod = np.prod(acc_r_sum,axis=1)

    eigen_prod2  = eigen_prod[eigen_prod.argsort()[::-1]]
    #tmp_num = min(min(max(5,int(len(eigen_prod)*0.01)),len(R_tmp)),num)
    #print 'tmp_num ',tmp_num, len(R_tmp), 
    tmp_num = min(len(R_tmp),num)
    #print 'tmp_num ',tmp_num, len(R_tmp)

    tmp_val = eigen_prod2[tmp_num-1]
    tmp_idx = eigen_prod>=tmp_val
    R_ = R_tmp[tmp_idx]

    return R_


def Ap_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm,  R, C_orig, a, num):
    R_set = Rmax[R_idx] - R_tmp
    R_set = R_set.astype(int)

    eigen_sum = np.zeros(np.shape(R_tmp))

    for i3 in range(np.shape(R_tmp)[1]) :
        ig = g_idx[i3]
        eigen_tmp = np.asarray(eigen_cumsum[ig][0][:])
        eigen_sum[:,i3] = (eigen_tmp[np.array(R_set[:,i3])-1]-eigen_cumsum[ig][0][0])/(eigen_cumsum[ig][0][-1]-eigen_cumsum[ig][0][0])

    #Cmax = R[-1].dot(a)/C_orig
    #C_tmp = R_set.dot(a[np.asarray(g_idx)])/C_orig/Cmax
    #eigen_prod = np.multiply(np.prod(eigen_sum,axis=1),C_tmp)

    eigen_prod = np.prod(eigen_sum,axis=1)

    eigen_prod2  = eigen_prod[eigen_prod.argsort()[::-1]]
    #tmp_num = min(min(max(5,int(len(eigen_prod)*0.01)),len(R_tmp)),num)
    tmp_num = min(len(eigen_prod),num)
    tmp_val = eigen_prod2[tmp_num-1]
    tmp_idx = eigen_prod>=tmp_val
    R_ = R_tmp[tmp_idx]

    return R_


def Hier_R_sel(r_max, Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, num, Ct):
    tmp = r_max[R_idx] - R_tmp
    tmp = np.prod(tmp,axis=1)
    R_tmp = R_tmp[tmp>=0]
    stop_flag = 0
    if len(R_tmp) == 0:
        stop_flag = 1
        R_ = []
        return R_, stop_flag

    elif len(R_tmp) == 1:
        R_ = R_tmp

    else:
        if Sel_metric == 0 :
            if len(R_tmp) > num:
                #R_ = Am_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, num)
                R_ = Ac_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, num, Ct)
            else : 
                R_ = R_tmp
        elif Sel_metric == 1:
            #R_ = Am_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, num)
            if len(R_tmp) > num:
                R_ = Ac_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm, A,  R, C_orig, a, num, Ct)
            else : 
                R_ = R_tmp

        elif Sel_metric == 2:
            R_ = Ap_hier(Rmax, R_idx, R_tmp, g_idx, eigen_cumsum, R_norm,  R, C_orig, a, num)

    return R_, stop_flag

def Ac_top(R_num, Rmax, Y_, eigen_cumsum, R_norm, A,  R, C_orig, a, L):
    eigen_sum = np.zeros(np.shape(Y_))
    acc_r_sum = np.zeros(np.shape(Y_))
    R_set = np.asarray(Rmax - Y_,dtype=int)
    for i in range(L):    
        eigen_tmp = np.asarray(eigen_cumsum[i][0][:])
        eigen_sum[:,i] = (eigen_tmp[np.array(R_set[:,i])-1]-eigen_cumsum[i][0][0])/(eigen_cumsum[i][0][-1]-eigen_cumsum[i][0][0])
        acc_r_sum[:,i] = interpolate.pchip_interpolate(R_norm[:,i], A[:,i], R_set[:,i]/R[-1,i].astype(float))

    C_tmp = R_set.astype(float).dot(a)/C_orig
    
    eigen_prod = np.multiply(np.prod(eigen_sum,axis=1),C_tmp)
    Am = np.prod(acc_r_sum,axis=1)
    eigen_prod = eigen_prod + Am
    eigen_prod2  = eigen_prod[eigen_prod.argsort()[::-1]]

    tmp_num = min(len(R_set),R_num)
    tmp_val = eigen_prod2[tmp_num-1]
    tmp_idx = eigen_prod>=tmp_val
    Y_ = Y_[tmp_idx]
    return Y_



def Am_top(R_num, Rmax, Y_, eigen_cumsum, R_norm, A,  R, C_orig, a, L):
    acc_r_sum = np.zeros(np.shape(Y_))
    R_set = np.asarray(Rmax - Y_,dtype=int)
    for i in range(L):    
        acc_r_sum[:,i] = interpolate.pchip_interpolate(R_norm[:,i], A[:,i], R_set[:,i]/R[-1,i].astype(float))
    C_tmp = R_set.dot(a)
    eigen_prod = np.prod(acc_r_sum,axis=1)

    eigen_prod2  = eigen_prod[eigen_prod.argsort()[::-1]]
    tmp_num = min(len(R_set),R_num)
    tmp_val = eigen_prod2[tmp_num-1]
    tmp_idx = eigen_prod>=tmp_val
    Y_ = Y_[tmp_idx]
    return Y_


def Ap_top(R_num, Rmax, Y_, eigen_cumsum, R_norm,  R, C_orig, a, L):
    eigen_sum = np.zeros(np.shape(Y_))
    R_set = np.asarray(Rmax - Y_,dtype=int)
    for i in range(L):    
        eigen_tmp = np.asarray(eigen_cumsum[i][0][:])
        eigen_sum[:,i] = (eigen_tmp[np.array(R_set[:,i])-1]-eigen_cumsum[i][0][0])/(eigen_cumsum[i][0][-1]-eigen_cumsum[i][0][0])

    eigen_prod = np.prod(eigen_sum,axis=1)
    eigen_prod2  = eigen_prod[eigen_prod.argsort()[::-1]]
    tmp_num = min(len(R_set),R_num)
    tmp_val = eigen_prod2[tmp_num-1]
    tmp_idx = eigen_prod>=tmp_val
    Y_ = Y_[tmp_idx]
    return Y_


# ===============================================
# ===============================================
# ===============================================

def Unfolding_group_2(set_idx_group, range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, Y, stop_flag, sel_num):
    setx_iter_2 = []
    setx_size_2 = []
    for j in range(len(set_idx_group)) : # j : index of stage-1

        # ---- stage 2 ----
        set_idx_ = set_idx_group[j]
        setx_iter_2_ = setx_iter_2_tot[j]
        setx_iter_2_sum = setx_iter_2_sum_tot[j]
        val_stage1 = Y[j] # for unique

        setx_iter_2_ = setx_iter_2_[setx_iter_2_sum == val_stage1] # effective set of stage-2
        if len(setx_iter_2_) == 0:
            stop_flag = 1
            break

        # random selection
        if len(setx_iter_2_) > sel_num:
            random_num = min(len(setx_iter_2_), sel_num)
            step = int(len(setx_iter_2_)/random_num) 
            setx_iter_2_ = setx_iter_2_[::step,:]

        size_2 = len(setx_iter_2_)
        setx_iter_2 = setx_iter_2 + [setx_iter_2_]
        setx_size_2 = setx_size_2 + [size_2]

    return setx_iter_2, setx_size_2, stop_flag



def Unfolding_bottom_group_3_of_3(group_hier, set_idx_, setx_iter_2_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, Ct):
    setx_iter_3_3 = []
    setx_size_3_3 = []
    for i in range(len(group_hier[set_idx_])) : # i : index of stage-2 of [j]  

        # --- stage 3 ----
        #range_3 = range_3_tot[i]
#        print 'group_hier[set_idx_]',group_hier[set_idx_]
        setx_iter_3_ = np.asarray(setx_iter_3_tot[i])
        setx_iter_3_sum = setx_iter_3_sum_tot[i]

        val_stage2 = setx_iter_2_group[i] # i-th fixed value of stage-2 set
        setx_iter_3_tmp = setx_iter_3_[setx_iter_3_sum == val_stage2] # effective set of stage-3
        if len(setx_iter_3_tmp) == 0:
            stop_flag = 1
            break

        setx_iter_3_, stop_flag = Hier_R_sel(r_max, Rmax, np.asarray(group_hier[set_idx_][i]), setx_iter_3_tmp, group_hier[set_idx_][i], eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, 1, Ct)
        if stop_flag == 1:
            break

        setx_iter_3_3 = setx_iter_3_3 + [setx_iter_3_]
        setx_size_3_3 = setx_size_3_3 + [len(setx_iter_3_)]

    return setx_iter_3_3, setx_size_3_3, stop_flag


def Unfolding_bottom_group_2_of_3(group_hier, set_idx_, setx_iter_2, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, stop_flag, Sel_metric, Ct):
    setx_size_3_2 = []
    setx_iter_3_2 = []

#    print 'before ',len(setx_iter_2)
#    xx = len(setx_iter_2)
#    setx_iter_2 = np.unique(np.asarray(setx_iter_2), axis=0)
#    print 'after ',len(setx_iter_2)
#    if (xx - len(setx_iter_2)) != 0:
#        aaa = input()

    
    # NOTE: To further accelerate the network compression, process this part in parallel.
    #---------------------------------------------------------------------
    for n3 in range(len(setx_iter_2)): # n3 : index of stage-2 sets of [j]
        setx_iter_2_group = setx_iter_2[n3] # a set of stage-2 of [j]

#        print 'setx_iter_2_group',setx_iter_2_group
#        aaa = input()

        setx_iter_3_3, setx_size_3_3, stop_flag = Unfolding_bottom_group_3_of_3(group_hier, set_idx_, setx_iter_2_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, Ct)
        if stop_flag == 1:
            sys.stdout.flush()
            stop_flag = 0
            continue

        tmp = np.asarray(list(itertools.product(*setx_iter_3_3))) #[setx_iter_3_3]
        tmp = tmp.reshape([len(tmp),-1])
        tmp2 = np.hstack(tmp.flat)
        tmp2 = tmp2.reshape([len(tmp),-1])



        setx_iter_3_2 = setx_iter_3_2 + tmp2.tolist() # group in n3 of j
        setx_size_3_2 = setx_size_3_2 + [np.prod(setx_size_3_3)] #[setx_size_3_3]

    return setx_iter_3_2, setx_size_3_2, stop_flag


def Unfolding_bottom_group_1_of_3(set_idx_group, group_init, group_hier, setx_iter_2,  range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, stop_flag, Sel_metric, Ct):
    setx_size_3_1 = []
    setx_iter_3_1 = []
    stop_flag = 0

    for j in range(len(set_idx_group)) : # j : index of stage-1

        set_idx_ = set_idx_group[j]
        setx_iter_3_2, setx_size_3_2, stop_flag = Unfolding_bottom_group_2_of_3(group_hier, set_idx_,setx_iter_2[j],  range_3_tot[j], setx_iter_3_tot[j], setx_iter_3_sum_tot[j], r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, stop_flag, Sel_metric, Ct)
        tmp = np.asarray(setx_iter_3_2)
        if len(tmp) == 0 :
            stop_flag = 1
            break
    
        tmp, stop_flag = Hier_R_sel(r_max, Rmax, group_init[set_idx_group[j]], tmp, group_init[set_idx_group[j]], eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, 1, Ct)
        if stop_flag == 1:
            break

        setx_iter_3_1 = setx_iter_3_1 + [tmp]
        setx_size_3_1 = setx_size_3_1 + [len(tmp)] 

    return setx_iter_3_1, setx_size_3_1, stop_flag

def Unfolding_single_group(set_idx_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, Y, group_hier, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, hier_num, Ct):
    setx_iter_3_3 = []
    setx_size_3_3 = []

    for j in range(len(set_idx_group)) : # j : index of stage-1

        set_idx_ = set_idx_group[j]
        range_3 = range_3_tot[j]
        setx_iter_3_ = np.asarray(setx_iter_3_tot[j])
        setx_iter_3_sum = setx_iter_3_sum_tot[j]

        val_stage2 = Y[set_idx_+1] # 
        #val_stage2 = Y[j] # for unique
        setx_iter_3_tmp = setx_iter_3_[setx_iter_3_sum == val_stage2] # effective set of stage-3
        if len(setx_iter_3_tmp) == 0:
            stop_flag = 1
            break

        setx_iter_3_, stop_flag = Hier_R_sel(r_max, Rmax, np.asarray(group_hier[set_idx_][0]), setx_iter_3_tmp, group_hier[set_idx_][0], eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, hier_num, Ct) #25
        if stop_flag == 1:
            break

        setx_iter_3_3 = setx_iter_3_3 + [setx_iter_3_]
        setx_size_3_3 = setx_size_3_3 + [len(setx_iter_3_)]

    return setx_iter_3_3, setx_size_3_3, stop_flag


def Unfolding_single_group(set_idx_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, Y, group_hier, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, hier_num, Ct):
    setx_iter_3_3 = []
    setx_size_3_3 = []

    for j in range(len(set_idx_group)) : # j : index of stage-1

        set_idx_ = set_idx_group[j]
        range_3 = range_3_tot[j]
        setx_iter_3_ = np.asarray(setx_iter_3_tot[j])
        setx_iter_3_sum = setx_iter_3_sum_tot[j]

        #val_stage2 = Y[set_idx_+1] # 
        val_stage2 = Y[j] # for unique
        setx_iter_3_tmp = setx_iter_3_[setx_iter_3_sum == val_stage2] # effective set of stage-3
        if len(setx_iter_3_tmp) == 0:
            stop_flag = 1
            break

        setx_iter_3_, stop_flag = Hier_R_sel(r_max, Rmax, np.asarray(group_hier[set_idx_][0]), setx_iter_3_tmp, group_hier[set_idx_][0], eigen_cumsum, R_norm, A,  R, C_orig, a, Sel_metric, hier_num, Ct) #25
        if stop_flag == 1:
            break

        setx_iter_3_3 = setx_iter_3_3 + [setx_iter_3_]
        setx_size_3_3 = setx_size_3_3 + [len(setx_iter_3_)]

    return setx_iter_3_3, setx_size_3_3, stop_flag

def Unfolding_layers(setx_iter_tot, L, group_init, set_idx_group, Y):
    Y1_ = np.zeros([len(setx_iter_tot)*len(Y),L])
    j = 0
    for i in range(len(group_init)) :
        if i in set_idx_group :
            tmp = setx_iter_tot[:,j]
            tmp = np.hstack(tmp.flat)
            tmp = tmp.reshape([len(setx_iter_tot),-1])
            tmp = np.tile(tmp,(len(Y),1))
            Y1_[:,group_init[i]] = tmp
            j += 1
        else :
            tmp = np.repeat(Y[:,i+1],len(setx_iter_tot),axis=0).reshape([len(Y1_),1])
            Y1_[:,group_init[i]] = tmp

    return Y1_

def Concatenation_sets(n, idx, Y1_, Rmax, Y_, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Y, Sel_metric):
    if idx == 0:
        Y_ = Y1_
    else:
        Y_ = np.concatenate((Y_,Y1_), axis=0)
        if len(Y_) > 600000 :
            if Sel_metric == 0:
                Y_ = Ac_top(10000, Rmax, Y_, eigen_cumsum, R_norm, A,  R, C_orig, c, L)
            elif Sel_metric == 1:
                Y_ = Am_top(10000, Rmax, Y_, eigen_cumsum, R_norm, A,  R, C_orig, c, L)
            elif Sel_metric == 2:
                Y_ = Ap_top(10000, Rmax, Y_, eigen_cumsum, R_norm, R, C_orig, c, L)

    return Y_

def Make_sets_over_2_level_sub(Y, uY, uIdx, group_init, group_hier, set_idx_group, range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Sel_metric, Ct, sel_num):
    # NOTE: To further accelerate the network compression, process this part in parallel.
    #---------------------------------------------------------------------
    stop_flag = 0
    Y_ = []
    for n in range(len(uY)):
        stop_flag = 0
        # unfolding level-2 groups
        setx_iter_2, setx_size_2, stop_flag = Unfolding_group_2(set_idx_group, range_2_tot, setx_iter_2_tot, setx_iter_2_sum_tot, uY[n], stop_flag, sel_num)
        if stop_flag == 1:
            stop_flag = 0
            continue

        # unfolding level-3 groups
        setx_iter_3_1, setx_size_3_1, stop_flag = Unfolding_bottom_group_1_of_3(set_idx_group, group_init, group_hier, setx_iter_2, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, stop_flag, Sel_metric, Ct)
        if stop_flag == 1 :
            stop_flag = 0
            continue
        
        num_sub_sets = np.prod(setx_size_3_1)
        if num_sub_sets == 0 :
            stop_flag = 0
            sys.stdout.flush()
            continue
        
        # final unfolding sets
        setx_iter_tot = np.array(list(itertools.product(*setx_iter_3_1)))
        if len(setx_iter_tot) == 0 :
            stop_flag = 0
            continue

        # Unfolding layers
        Y_ = Unfolding_layers(setx_iter_tot, L, group_init, set_idx_group, Y)

    return Y_


def Make_sets_single_level_sub(Y, uY, uIdx, group_init, group_hier, set_idx_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, L, Sel_metric, hier_num, Ct):
    # NOTE: To further accelerate the network compression, process this part in parallel.
    #---------------------------------------------------------------------
    stop_flag = 0
    Y_ = []
    k = 0
    for n in range(len(uY)):
        stop_flag = 0
        setx_iter_3_3, setx_size_3_3, stop_flag  = Unfolding_single_group(set_idx_group, range_3_tot, setx_iter_3_tot, setx_iter_3_sum_tot, uY[n], group_hier, r_max, Rmax, eigen_cumsum, R_norm, A,  R, C_orig, c, Sel_metric, hier_num, Ct)
        if stop_flag == 1 :
            stop_flag = 0
            continue

        setx_iter_tot = np.array(list(itertools.product(*setx_iter_3_3)))
        if len(setx_iter_tot) == 0 :
            stop_flag = 0
            continue

        # Unfolding layers
        Y_ = Unfolding_layers(setx_iter_tot, L, group_init, set_idx_group, Y)

    return Y_

