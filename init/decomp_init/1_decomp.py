import numpy as np
import json
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from pprint import pprint
import scipy
import operator

import google.protobuf as pb
import google.protobuf.text_format as pbt

current_path = os.getcwd()
CAFFE_ROOT = current_path+'/caffe/'
if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
import caffe.proto.caffe_pb2
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter #V1LayerParameter


#from google import protobuf

def load_config(config_file):
    with open(config_file, 'r') as fp:
        conf = json.load(fp)
    return conf


def fc_decomp_2d(fc, K):
    def _create_new(name):
        new_ = LayerParameter()
        new_.CopyFrom(fc)
        new_.name = name
        return new_
    fc_param = fc.inner_product_param
    # vertical
    v = _create_new(fc.name + '_v')
    del(v.top[:])
    v.top.extend([v.name])
    print('v.name = {}'.format(v.name))
    if v.param:
        v.param[1].lr_mult = 0
    v_param = v.inner_product_param
    v_param.num_output = K
    # horizontal
    h = _create_new(fc.name + '_h')
    del(h.bottom[:])
    h.bottom.extend(v.top)
    h_param = h.inner_product_param
    return v, h


def cv_spatial(conv, K):
    def _create_new(name):
        new_ = LayerParameter()
        new_.CopyFrom(conv)
        new_.name = name
        new_.convolution_param.ClearField('kernel_size')
        new_.convolution_param.ClearField('pad')
        new_.convolution_param.ClearField('stride')
        return new_
    conv_param = conv.convolution_param
    # vertical
    print(conv.name)
    v = _create_new(conv.name + '_v')
    del(v.top[:])
    v.top.extend([v.name])
    if np.shape(v.param)[0]>1:
        v.param[1].lr_mult = 0
    v_param = v.convolution_param
    v_param.num_output = K
    v_param.kernel_h = conv_param.kernel_size[0] if len(np.shape(conv_param.kernel_size)) > 0 else conv_param.kernel_size
    v_param.kernel_w = 1
    if conv_param.pad :
        v_param.pad_h = conv_param.pad[0] if len(np.shape(conv_param.pad)) > 0 else conv_param.pad
    else :
        v_param.pad_h = 0
    v_param.pad_w = 0

    if conv_param.stride:
        v_param.stride_h = conv_param.stride[0] if len(np.shape(conv_param.stride)) > 0 else conv_param.stride
        v_param.stride_w = conv_param.stride[0] if len(np.shape(conv_param.stride)) > 0 else conv_param.stride
    else :
        v_param.stride_h = 1
        v_param.stride_w = 1

    # horizontal
    h = _create_new(conv.name + '_h')
    del(h.bottom[:])
    h.bottom.extend(v.top)
    h_param = h.convolution_param
    h_param.kernel_h = 1
    h_param.kernel_w = conv_param.kernel_size[0] if len(np.shape(conv_param.kernel_size)) > 0 else conv_param.kernel_size
    h_param.pad_h = 0
    if conv_param.pad : 
        h_param.pad_w = conv_param.pad[0] if len(np.shape(conv_param.pad)) > 0 else conv_param.pad
    else:
        h_param.pad_w = 0

    if conv_param.stride:
        h_param.stride_w = 1 #conv_param.stride[0]
        h_param.stride_h = 1
    else:
        h_param.stride_w = 1
        h_param.stride_h = 1

    return v, h


def cv_channel(conv, K):
    def _create_new(name):
        new_ = LayerParameter()
        new_.CopyFrom(conv)
        new_.name = name
        new_.convolution_param.ClearField('kernel_size')
        new_.convolution_param.ClearField('pad')
        new_.convolution_param.ClearField('stride')
        return new_
    conv_param = conv.convolution_param
    # vertical
    print(conv.name)
    v = _create_new(conv.name + '_v')
    del(v.top[:])
    v.top.extend([v.name])
    if np.shape(v.param)[0]>1:
        v.param[1].lr_mult = 0
    v_param = v.convolution_param
    v_param.num_output = K
    v_param.kernel_h = conv_param.kernel_size[0] if len(np.shape(conv_param.kernel_size)) > 0 else conv_param.kernel_size
    v_param.kernel_w = conv_param.kernel_size[0] if len(np.shape(conv_param.kernel_size)) > 0 else conv_param.kernel_size
    if conv_param.pad :
        v_param.pad_h = conv_param.pad[0] if len(np.shape(conv_param.pad)) > 0 else conv_param.pad
        v_param.pad_w = conv_param.pad[0] if len(np.shape(conv_param.pad)) > 0 else conv_param.pad
    else:
        v_param.pad_h = 0
        v_param.pad_w = 0

    if conv_param.stride:
        v_param.stride_h = conv_param.stride[0] if len(np.shape(conv_param.stride)) > 0 else conv_param.stride
        v_param.stride_w = conv_param.stride[0] if len(np.shape(conv_param.stride)) > 0 else conv_param.stride
    else:
        v_param.stride_h = 1
        v_param.stride_w = 1

    # horizontal
    h = _create_new(conv.name + '_h')
    del(h.bottom[:])
    h.bottom.extend(v.top)
    h_param = h.convolution_param
    h_param.kernel_h = 1
    h_param.kernel_w = 1
    h_param.pad_h = 0
    h_param.pad_w = 0
    h_param.stride_h = 1
    h_param.stride_w = 1

    return v, h

def make_decomp_file(input_file, conf, output_file, sim_type, weight, output_weight, args):
    with open(input_file, 'r') as fp:
        net = NetParameter()
        pbt.Parse(fp.read(), net)

    filename = '../../base_models/{}/eigenvalue.conf'.format(args.net_type)
    f = open(filename, 'w')
    f.write('{\n\t"layer":{\n')

    i = 1
    idx = 0
    new_layers = []
    layer_dic = {}
    for layer in net.layer:
        if layer.name in conf["vbmf"]["cv"].keys():
            print('layer.name = {}'.format(layer.name))
            layer_dic[layer.name] = idx
            bottom = layer.bottom[0]
            k = (conf["vbmf"]["cv"][layer.name])
            if bottom == 'data' :
                a, b = cv_channel(layer, k)
                new_layers.extend([a, b])
            else:
                g = layer.convolution_param.group
                a, b = cv_spatial(layer, k*g)
                new_layers.extend([a, b])
            idx += 1
        else: 
            if layer.name in conf["vbmf"]["fc"].keys():
                layer_dic[layer.name] = idx
                k = conf["vbmf"]["fc"][layer.name]
                a, b = fc_decomp_2d(layer, k)
                new_layers.extend([a, b])
            else:
                new_layers.append(layer)
                continue

    new_net = NetParameter()
    new_net.CopyFrom(net)
    del(new_net.layer[:])
    new_net.layer.extend(new_layers)

    # File Write - eigenvalue.conf
    sorted_dic = sorted(layer_dic.items(), key=operator.itemgetter(1))
    for j in range(len(sorted_dic)-1):
        f.write('"{}": {},\n'.format(sorted_dic[j][0], sorted_dic[j][1]))
    f.write('"{}": {}\n {}'.format(sorted_dic[len(sorted_dic)-1][0], sorted_dic[len(sorted_dic)-1][1],'},'))
    f.close()

    # File Write - prototxt
    out = os.path.splitext(os.path.basename(output_file))[0] + '_{}.prototxt'.format(i)
    out_dir = os.path.dirname(output_file)
    filename = '{}/{}'.format(out_dir, out)
    with open(filename, 'w') as fp:
        fp.write(pb.text_format.MessageToString(new_net))
    print '[Total] Wrote compressed prototxt to: {:s}'.format(filename)

    if sim_type == 'weight':
        out = os.path.splitext(os.path.basename(output_weight))[0] + '_{}.caffemodel'.format(i)
        out_dir = os.path.dirname(output_weight)
        out_weight = '{}/{}'.format(out_dir, out)
        decomp_weights(net.layer, input_file, weight, conf, filename, out_weight, i, args)



def decomp_weights(orig_layer, orig_model, orig_weights, conf, lowrank_model, lowrank_weights,  scaling,args):
    orig_net = caffe.Net(orig_model, orig_weights, caffe.TEST)
    lowrank_net = caffe.Net(lowrank_model, orig_weights, caffe.TEST)

    filename = '../../base_models/{}/eigenvalue.conf'.format(args.net_type)
    f = open(filename, 'a+')
    f.write('\t"eigen":{\n')

    eigen_dic = {}

    for layer_name in conf["vbmf"]["cv"]:
        for layer in orig_layer:
            if layer.name == layer_name:
                orig_param = layer.convolution_param
                bottom = layer.bottom[0]
                continue

        if bottom == 'data' :
            if(len(orig_net.params[layer_name]) == 2):
                W, b = [p.data for p in orig_net.params[layer_name]]
                v_weights, v_bias = [p.data for p in lowrank_net.params[layer_name + '_v']]
                h_weights, h_bias = [p.data for p in lowrank_net.params[layer_name + '_h']]
                v_bias[...] = 0
                h_bias[...] = b.copy()
            else:
                W = orig_net.params[layer_name][0].data
                v_weights = lowrank_net.params[layer_name + '_v'][0].data
                h_weights = lowrank_net.params[layer_name + '_h'][0].data

            # Get the shapes
            num_groups = v_weights.shape[0] // h_weights.shape[1]
            N, C, D, D = W.shape
            N = N // num_groups
            K = h_weights.shape[1] // num_groups
            print('{} {}, K : {}'.format(layer_name, W.shape, K))
#            if num_groups > 1:
#                f.write('"{}" : [[ \n'.format(layer_name))
#            else:
#                f.write('"{}" : [ \n'.format(layer_name))
            tmp_S = np.empty([num_groups, K], dtype=float)
            for g in xrange(num_groups):
                W_ = W[N*g:N*(g+1)].reshape((N, C*D*D))
                U, S, V = scipy.linalg.svd(W_ , full_matrices=False, lapack_driver='gesvd')
#                U, S, V = scipy.linalg.svd(W_ , full_matrices=False)
#                for i in range(K-1):
#                    f.write('{},\n'.format(S[i]))
#                if g == num_groups-1:
#                    f.write('{}]],\n'.format(S[K-1]))
#                else:
#                    f.write('{}],\n[\n'.format(S[K-1]))
#                f.write('\n')

                h = U[:, :K] * np.sqrt(S[:K])
                h = h[:, :K].reshape((N, K, 1, 1))
                np.copyto(h_weights[N*g:N*(g+1)], h)

                v = V[:K, :] * np.sqrt(S)[:K, np.newaxis]
                v = v.reshape((K, C, D, D))
                np.copyto(v_weights[K*g:K*(g+1)], v)

                tmp_S[g] = S[:K]

            eigen_dic[layer_name] = tmp_S

        else:
            if(len(orig_net.params[layer_name]) == 2):
                W, b = [p.data for p in orig_net.params[layer_name]]
                v_weights, v_bias = [p.data for p in lowrank_net.params[layer_name + '_v']]
                h_weights, h_bias = [p.data for p in lowrank_net.params[layer_name + '_h']]
                v_bias[...] = 0
                h_bias[...] = b.copy()
            else:
                W = orig_net.params[layer_name][0].data
                v_weights = lowrank_net.params[layer_name + '_v'][0].data
                h_weights = lowrank_net.params[layer_name + '_h'][0].data


            # Get the shapes
            num_groups = orig_param.group
            N, C, D, D = W.shape
            N = N// num_groups
            K = v_weights.shape[0]//num_groups
            print('{} {}, K = {}, group = {}'.format(layer_name, W.shape, K*num_groups, num_groups))
#            if num_groups > 1:
#                f.write('"{}" : [[ \n'.format(layer_name))
#            else:
#                f.write('"{}" : [ \n'.format(layer_name))
            tmp_S = np.empty([num_groups, K], dtype=float)
            for g in xrange(num_groups):
                W_ = W[N*g:N*(g+1)].transpose(1, 2, 3, 0).reshape((C*D, D*N))
                U, S, V = scipy.linalg.svd(W_ , full_matrices=False, lapack_driver='gesvd')
#                U, S, V = scipy.linalg.svd(W_ , full_matrices=False)
#                for i in range(K-1):
#                    f.write('{},\n'.format(S[i]))
#                if g == num_groups-1:
#                    f.write('{}]],\n'.format(S[K-1]))
#                else:
#                    f.write('{}],\n[\n'.format(S[K-1]))
#                f.write('\n')

                h = V[:K, :] * np.sqrt(S)[:K, np.newaxis]
                h = h.reshape((K, 1, D, N)).transpose(3, 0, 1, 2)
                np.copyto(h_weights[N*g:N*(g+1)], h)

                v = U[:, :K] * np.sqrt(S[:K])
                v = v[:, :K].reshape((C, D, 1, K)).transpose(3, 0, 1, 2)
                np.copyto(v_weights[K*g:K*(g+1)], v)

                tmp_S[g] = S[:K]

            eigen_dic[layer_name] = tmp_S

    for layer_name in conf["vbmf"]["fc"]:
        if(len(orig_net.params[layer_name]) == 2):
            W, b = [p.data for p in orig_net.params[layer_name]]
            v_weights, v_bias = [p.data for p in lowrank_net.params[layer_name + '_v']]
            h_weights, h_bias = [p.data for p in lowrank_net.params[layer_name + '_h']]
            v_bias[...] = 0
            h_bias[...] = b.copy()
        else:
            W = orig_net.params[layer_name][0].data
            v_weights = lowrank_net.params[layer_name + '_v'][0].data
            h_weights = lowrank_net.params[layer_name + '_h'][0].data

        # Get the shapes
        N, C = W.shape
        #K = hp_weights.shape[1]
        K = h_weights.shape[1]
        num_groups = v_weights.shape[0] // h_weights.shape[1]
        N = N // num_groups

        # SVD approximation
        print('num_groups = {}'.format(num_groups))
        print('{} {}, K = {}, group = {}'.format(layer_name, W.shape, K*num_groups, num_groups))
#        f.write('"{}" : [ \n'.format(layer_name))
        W_ = W
        U, S, V = scipy.linalg.svd(W_ , full_matrices=False, lapack_driver='gesvd')
#        for i in range(K-1):
#            f.write('{},\n'.format(S[i]))
#        f.write('{}],\n'.format(S[K-1]))
#        f.write('\n')

        eigen_dic[layer_name] = S[:K]
        v = np.dot(np.diag(S[:K]), V[:K,:])
        v_weights[0:K] = v.copy()
        h = U[:, :K]
        h_weights[0:N] = h.copy()

    # File Write - eigenvalue.conf
    sorted_dic = sorted(eigen_dic.items(), key=operator.itemgetter(0))
    for j in range(len(sorted_dic)):
        layer_name = sorted_dic[j][0]
        S_ = sorted_dic[j][1]
        K = np.shape(S_)[1]
        num_g = np.shape(S_)[0] 

        if num_g > 1:
            f.write('"{}" : [[ \n'.format(layer_name))
        else:
            f.write('"{}" : [ \n'.format(layer_name))

        for g in xrange(num_g):
            S = S_[g]
            for i in range(K-1):
                f.write('{},\n'.format(S[i]))
            if j == len(sorted_dic)-1:
                if num_g > 1 and g == num_g-1:
                    f.write('{}]]\n'.format(S[K-1]))
                elif num_g > 1 and g < num_g-1: 
                    f.write('{}]],\n[\n'.format(S[K-1]))
                else:
                    f.write('{}]\n'.format(S[K-1]))
            else:
                if num_g > 1 and g == num_g-1:
                    f.write('{}]],\n'.format(S[K-1]))
                elif num_g > 1 and g < num_g-1: 
                    f.write('{}]],\n[\n'.format(S[K-1]))
                else:
                    f.write('{}],\n'.format(S[K-1]))
            f.write('\n')


#            if num_groups > 1:
#                f.write('"{}" : [[ \n'.format(layer_name))
#            else:
#                f.write('"{}" : [ \n'.format(layer_name))
#                for i in range(K-1):
#                    f.write('{},\n'.format(S[i]))
#                if g == num_groups-1:
#                    f.write('{}]],\n'.format(S[K-1]))
#                else:
#                    f.write('{}],\n[\n'.format(S[K-1]))
#                f.write('\n')


    f.write('}\n}')
    f.close()
    lowrank_net.save(lowrank_weights)
    print '[Total] Wrote compressed weight to: {:s}'.format(lowrank_weights)

    out = os.path.splitext(os.path.basename(lowrank_weights))[0] 
    out_dir = os.path.dirname(lowrank_weights)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


def main(args):
    conf = load_config(args.config)
    print('START PROTOTXT')
    make_decomp_file(args.train, conf, args.save_train, 'file', args.weights, args.save_weights, args)
    print('START COMP')
    make_decomp_file(args.model, conf, args.save_model, 'weight', args.weights, args.save_weights, args)


if __name__ == '__main__':
    parser = ArgumentParser(description="Low-rank approximation")
    parser.add_argument('--model', required=True,
        help="Prototxt of the original deploy net")
    parser.add_argument('--train', required=True,
        help="Prototxt of the original train_val net")
    parser.add_argument('--config', required=True,
        help="JSON config file specifying the low-rank approximation")
    parser.add_argument('--weights', required=True,
        help="Caffemodel of the original net")

    parser.add_argument('--save_model', required=True,
        help="Path to the deploy prototxt of the low-rank approximated net")
    parser.add_argument('--save_train', required=True,
        help="Path to the train_val prototxt of the low-rank approximated net")
    parser.add_argument('--save_weights', required=True,
        help="Path to the caffemodel of the low-rank approximated net")

    parser.add_argument('--net_type', required=True,
        help="network model")

    args = parser.parse_args()
    main(args)



