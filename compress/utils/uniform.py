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

def Uniform_comp(ratio, Sel_comp):
    W_orig, C_orig, VGG16_fc, Net, W_norm, Wmax, C_norm, Cmax, R_norm, L, A, R, W, C = Net_spec(Sel_comp)
    R_set = np.round(Net[:,1]*ratio)
    a = Net[:,7]/Net[:,1]
    w = Net[:,6]/Net[:,1]


    C_acc = (R_set.dot(a))/C_orig
    W_acc = (R_set.dot(w))/W_orig

    return R_set, C_acc, W_acc
