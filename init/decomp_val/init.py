
import numpy as np
import json
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from pprint import pprint


import google.protobuf as pb
import google.protobuf.text_format as pbt

current_path = os.getcwd()
CAFFE_ROOT = current_path+'/caffe/'

if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))

sys.path.insert(0, osp.join(current_path))
import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter

from decompose_layer_fc import *
from operator import itemgetter
import operator

def load_config(config_file):
    with open(config_file, 'r') as fp:
        conf = json.load(fp)
    return conf


def json_dump(data, i):
    json_write = './nin/conf/config.json.auto' + '_{}'.format(i)
    with open(json_write, 'w') as outfile:
            json.dump(data, outfile)


def load_file(input_file):
    with open(input_file, 'r') as fp:
        net = NetParameter()
        pbt.Parse(fp.read(), net)

    return net

def read_file_txt(path):
    results = []
    with open(path) as inputfile:
        for line in  inputfile:
	    results.append(line.strip().split(' '))

    return results


##########################################################################################
def main(args):

    # ===========
    gpu_idx = 0
    # ==========

    # Start Net
    start_train_val = args.priv_train_val
    start_deploy = args.priv_deploy
    start_weights = args.priv_weights

    # load the Rmax net
    orig_net = caffe.Net(args.orig_train_val, args.orig_weights, caffe.TEST)

    # load the deploy file
    net = load_file(args.orig_train_val)
    priv_net = load_file(start_train_val)

    # load the vbmf weight 
    priv_param = caffe.Net(start_train_val, start_weights, caffe.TEST)

    # load configure file
    priv_conf = load_config(args.priv_config) 
    eigen_conf = load_config('{}/eigenvalue.conf'.format(args.type)) 

    # load candidate sets
    net_type = args.type
    filename_set = '{}/comp_{}/{}.txt'.format(net_type,net_type,net_type)
    set_conf = read_file_txt(filename_set)
 
    # initial list_set
    check_set = set_conf
    check_set = np.array(check_set).astype(int)    
    check_set.tolist()

    layer_dic = eigen_conf["layer"]
    sorted_dic = sorted(layer_dic.items(), key=operator.itemgetter(1))
    layer_list = []
    for j in range(len(sorted_dic)):
        layer_list.append(sorted_dic[j][0])

    acc_loss_i = {}
    acc = {}
    loss = {}
        
    i = gpu_idx+1
    acc, loss = decompose_layer_fc(i-1, orig_net, priv_conf, net, priv_param, args.orig_train_val, args.orig_weights, start_train_val, start_weights, i, args.log, args.type, check_set, layer_list)
    filename = 'stage1_{}.txt'.format(args.type)
    f = open(filename, 'w')
    for k in range(len(acc)):
        acc_loss_i[k] = (i, acc[k], loss[k])
        f.write('{} {} {} {}\n'.format(i, k, acc[k], loss[k]))

    f.close()

if __name__ == '__main__':
    parser = ArgumentParser(description="Low-rank approximation")
    parser.add_argument('--orig_deploy', 
        help="Prototxt of the original deploy net")
    parser.add_argument('--orig_train_val', 
        help="Prototxt of the original train_val net")
    parser.add_argument('--orig_weights',
        help="Caffemodel of the original net")

    parser.add_argument('--priv_config', 
        help="JSON config file specifying the initial Rank , CaCb from VBMF")
    parser.add_argument('--rset_config', 
        help="JSON config file specifying the initial Rank , CaCb from VBMF")

    parser.add_argument('--priv_deploy',
        help="Path to the deploy prototxt of the low-rank approximated net")
    parser.add_argument('--priv_train_val',
        help="Path to the train_val prototxt of the low-rank approximated net")
    parser.add_argument('--priv_weights',
        help="Path to the caffemodel of the low-rank approximated net")
    parser.add_argument('--priv_solver',
        help="Path to the solver of the low-rank approximated net")

    parser.add_argument('--inter_deploy',
        help="Path to the deploy prototxt of the low-rank approximated net")
    parser.add_argument('--inter_train_val',
        help="Path to the train_val prototxt of the low-rank approximated net")
    parser.add_argument('--inter_weights',
        help="Path to the caffemodel of the low-rank approximated net")


    parser.add_argument('--log',
        help="Path to the solver of the low-rank approximated net")
    parser.add_argument('--type',
        help="Path to the solver of the low-rank approximated net")
    parser.add_argument('--start',
        help="Path to the solver of the low-rank approximated net")



    args = parser.parse_args()
    main(args)



