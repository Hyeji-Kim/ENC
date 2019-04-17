
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

#sys.path.insert(0, osp.join('/home/hjkim/ssd'))
import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter

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

def main(args):

    # load the deploy file
    net = load_file(args.orig_prototxt)
    orig_net = caffe.Net(args.orig_prototxt, args.orig_weight, caffe.TEST)

    sel_only_conv = args.sel_conv

    j = 0

    filename = '../../base_models/{}/config.json.rmax'.format(args.net_type)
    f = open(filename, 'w')
    f.write('{\n\t"vbmf": {\n')
    f.write('\t\t"cv": {\n')
    
    layer_dic = {}
    rmax_dic = {}

    for layer in net.layer:
        if layer.type == 'Convolution' :
            a=layer.name
            D=layer.convolution_param.kernel_size[0] if len(np.shape(layer.convolution_param.kernel_size))>0 else layer.convolution_param.kernel_size
            N=layer.convolution_param.num_output
            D1, D2, D3, D4 = orig_net.blobs[layer.name].data.shape

            bottom = layer.bottom[0]
#            print(bottom, bottom[0:4])

            if(len(orig_net.params[layer.name]) == 2):
                W, b = [p.data for p in orig_net.params[layer.name]]
            else:
                W = orig_net.params[layer.name][0].data

            # Get the shapes
            g = layer.convolution_param.group
            Ng = N // g
            N, C, D, D = np.asarray(W.shape,dtype=float)

            if bottom == 'data' :
                Rmax = C*N/g/(C+N/g/D/D) #C*N/(C+N/D/D)
            else :
                Rmax = C*N/g*D/(C+N/g) #C*N*D/(C+N)

            if 'proj' in layer.name :
                continue

            layer_dic[j] = [layer.name, int(round(Rmax))]
            j += 1


    for i in range(len(layer_dic)):
        if i == len(layer_dic)-1:
           f.write('\t\t\t"{}": {}\n'.format(layer_dic[i][0],layer_dic[i][1]))
        else:
           f.write('\t\t\t"{}": {},\n'.format(layer_dic[i][0],layer_dic[i][1]))


    layer_dic = {}
    j = 0
    f.write('\t\t},\n\t\t"fc": {\n')
    for layer in net.layer:
        if sel_only_conv == 0 and layer.type == 'InnerProduct' :
            a=layer.name
            D=1
            N=layer.inner_product_param.num_output

            if(len(orig_net.params[layer.name]) == 2):
                W, b = [p.data for p in orig_net.params[layer.name]]
            else:
                W = orig_net.params[layer.name][0].data

            # Get the shapes
            N, C = np.asarray(W.shape,dtype=float)
            Rmax = C*N/(C+N)

            layer_dic[j] = [layer.name, int(round(Rmax))]
            j += 1

    if sel_only_conv == 0:
        for i in range(len(layer_dic)):
            if i == len(layer_dic)-1:
               f.write('\t\t\t"{}": {}\n'.format(layer_dic[i][0],layer_dic[i][1]))
            else:
               f.write('\t\t\t"{}": {},\n'.format(layer_dic[i][0],layer_dic[i][1]))

    f.write('\t\t}\n\t}\n}')

    print '\n============================'
    print '  configure file gen [done]'
    print '============================'
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--orig_prototxt')
    parser.add_argument('--orig_weight')
    parser.add_argument('--sel_conv')
    parser.add_argument('--net_type')

    args = parser.parse_args()
    main(args)



