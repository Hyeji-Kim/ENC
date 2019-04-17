
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


import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import operator
from operator import itemgetter

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


#########################################################################################
    ## Calculate the Initial Rank and CaCb of each Matrix from VBMF
    ## 1st decompose
    ## fine tuning 1000 iter
    ## check the accuracy and loss
        ## -> load the accuacy and loss at 1000 iter. (refer graph py code)
    ## determine stop or continue wiht acc. loss
        ## threshold : about 2.36 loss
        ## -> if continue : reduce rank by using CaCb
##########################################################################################
def main(args):

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
    eigen_conf = load_config('base/eigenvalue.conf')


    # load candidate sets
    txt_idx = int(args.txt_idx)
    filename_set = 'tmp_/final_result_now_{}.txt'.format(txt_idx)
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
        
    i = txt_idx+1
    acc, loss = decompose_layer_fc(i-1, orig_net, priv_conf, net, priv_param, args.orig_train_val, args.orig_weights, start_train_val, start_weights, i, args.log, args.type, check_set, layer_list)

    filename = 'tmp_/stage1_{}_{}.txt'.format(args.type,txt_idx)
    f = open(filename, 'w')
    for k in range(len(acc)):
        acc_loss_i[k] = (i, acc[k], loss[k])
        f.write('{} {} {} {}\n'.format(i, k, acc[k], loss[k]))

    f.close()


if __name__ == '__main__':
    parser = ArgumentParser(description="Low-rank approximation")
    parser.add_argument('--orig_deploy') 
    parser.add_argument('--orig_train_val') 
    parser.add_argument('--orig_weights')

    parser.add_argument('--priv_config') 
    parser.add_argument('--priv_deploy')
    parser.add_argument('--priv_train_val')
    parser.add_argument('--priv_weights')
    parser.add_argument('--priv_solver')

    parser.add_argument('--inter_deploy')
    parser.add_argument('--inter_train_val')
    parser.add_argument('--inter_weights')


    parser.add_argument('--log')
    parser.add_argument('--type')
    parser.add_argument('--start')
    parser.add_argument('--txt_idx')
    parser.add_argument('--acc_type')

    args = parser.parse_args()

    #path_decomp = current_path+'/../../../init/decomp_{}'.format(args.acc_type)
    path_decomp = os.path.realpath('../../../init/decomp_{}'.format(args.acc_type))

    sys.path.insert(0, osp.join(path_decomp))
    from decompose_layer_fc import *

    main(args)

