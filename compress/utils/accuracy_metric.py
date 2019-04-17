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


def Calculate_Am(Rmax, R_txt, eigenvalue, eigen_cumsum, group, Sel_comp):

    # load network specification
    W_orig, C_orig, FC_cost, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C = Net_spec(Sel_comp)
    a = Net[:,7]/Net[:,1]
    w = Net[:,6]/Net[:,1]

    acc_r_sum = np.empty(np.shape(R_txt),dtype=float)

    # Cost calculation - PCA Energy
    Test_C = R_txt*a

    for i in range(L):
        # PCA energy - only conv
        acc_r_sum[:,i] = interpolate.pchip_interpolate(R_norm[:,i], A[:,i], R_txt[:,i]/R[-1,i].astype(float))

    Am = np.prod(acc_r_sum,axis=1)

    return Am

def Calculate_Ap(Rmax, R_txt, eigenvalue, eigen_cumsum, group, Sel_comp):

    # load network specification
    W_orig, C_orig, FC_cost, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C = Net_spec(Sel_comp)
    a = Net[:,7]/Net[:,1]
    w = Net[:,6]/Net[:,1]

    eigen_sum = np.empty(np.shape(R_txt),dtype=float)
    eigen_prod = np.empty(np.shape(R_txt)[0],dtype=np.float64)

    for i in range(L):
        eigen_tmp = np.asarray(eigen_cumsum[i][0][:])
        eigen_sum[:,i] = (eigen_tmp[np.array(R_txt[:,i])-1]-eigen_cumsum[i][0][0])/(eigen_cumsum[i][0][-1]-eigen_cumsum[i][0][0])

    Ap = np.prod(eigen_sum,axis=1)

    return Ap
 

def Calculate_Ac_matlab(Rmax, R_txt, eigenvalue, eigen_cumsum, group, Sel_comp):

    np.savetxt('tmp_/R_txt.txt',R_txt.ravel())
    # load network specification
    W_orig, C_orig, FC_cost, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C = Net_spec(Sel_comp)
    a = Net[:,7]/Net[:,1]
    w = Net[:,6]/Net[:,1]

    eigen_sum = np.empty(np.shape(R_txt),dtype=float)
    eigen_prod = np.empty(np.shape(R_txt)[0],dtype=np.float64)

    for i in range(L):
        # PCA energy - only conv
        eigen_tmp = np.asarray(eigen_cumsum[i][0][:])
        eigen_sum[:,i] = (eigen_tmp[np.array(R_txt[:,i])-1]-eigen_cumsum[i][0][0])/(eigen_cumsum[i][0][-1]-eigen_cumsum[i][0][0])

    Test_C2 = R_txt.astype(float).dot(a)/C_orig
    Ap = np.multiply(np.prod(eigen_sum,axis=1),Test_C2) # EC 

    command = 'addpath({}); feature_Ac({},{},{}); exit;'.format("'../../utils'",np.shape(R_txt)[0], np.shape(R_txt)[1], Sel_comp)
    print command
    subprocess.call(['matlab', '-nodisplay', '-nodesktop', '-nosplash', '-r', command])
    out = np.loadtxt('tmp_/MATLAB_feature.txt')
    Am = out[:,0]

    Ac = Ap + Am

    return Ac

