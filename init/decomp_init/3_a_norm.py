
import numpy as np
import json
import os
import os.path as osp
import sys
import time
import itertools
import google.protobuf as pb
#import caffe
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pprint import pprint
#from caffe.proto.caffe_pb2 import NetParameter, LayerParameter


import subprocess
from scipy import interpolate

np.set_printoptions(threshold=np.nan)


#from __future__ import print_function
#sys.stdout.write('.')
#sys.stdout.flush()
#from pchip_interpolate import *


sys.path.insert(0, osp.join('./.'))


def load_config(config_file):
    with open(config_file, 'r') as fp:
        conf = json.load(fp)
    return conf


def json_dump(data, i):

    json_write = './nin/conf/config.json.auto' + '_{}'.format(i)
    with open(json_write, 'w') as outfile:
            json.dump(data, outfile)

def read_file_rankset3(path):
    results = []
    with open(path) as inputfile:
        for line in  inputfile:
	    results.append(line.strip().split(' '))

    return results

               
def main(args):     

    Sel_acc = args.acc_sel # 1 : top-1 / 5 : top-5
    print 'Current Acc type : ', Sel_acc

    filename_norm = '../../base_models/{}/base/R_list.txt'.format(args.model)
    tmp = read_file_rankset3(filename_norm)
    tmp = np.array(tmp).astype(int)
    p_num = len(tmp)

    filename_acc = '../decomp_val/stage1_{}.txt'.format(args.model)
    filename = '../../base_models/{}/base/A_norm.txt'.format(args.model)
    f = open(filename, 'w')

    A = read_file_rankset3(filename_acc)
    A = np.array(A).astype(float)
    if Sel_acc == '1':
        acc_top = 2
    elif Sel_acc == '5':
        acc_top = 3
    acc_max = A[-1,acc_top]
    A = A[:-1,acc_top]
    A = A.reshape([-1, p_num-1]).T
    A = np.insert(A,len(A),acc_max,axis=0)
    A = A/acc_max

    for i in range(len(A)):
        for j in range(np.shape(A)[1]):
            f.write('{} '.format(A[i][j]))
        f.write('\n')
    f.close()

    print '[Done] A_norm.txt'

    # Show the graph  ---------------
    import matplotlib.pyplot as plt
    plt.figure()
    A = np.insert(A,0,0,axis=0)
    for i in range(np.shape(A)[1]):
        plt.plot(range(len(A)), A[:,i])
    plt.axis([0,len(A)-1,0,1])
    plt.show()

 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--acc_sel')


    args = parser.parse_args()
    main(args)

