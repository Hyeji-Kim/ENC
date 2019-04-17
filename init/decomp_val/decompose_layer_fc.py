import numpy as np
import json
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from pprint import pprint
import google.protobuf as pb
import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter

current_path = os.getcwd()
CAFFE_ROOT = current_path+'/caffe/'
def decompose_layer_fc(txt_idx, orig_net, priv_conf, net, priv_net, orig_file, orig_weight, output_file, output_weight, i, log, net_type, check_set, layer_list):

    filename = 'stage1_buf_{}.txt'.format(txt_idx)
    new_conf = priv_conf
    acc = {}
    loss = {}

    for num in range(len(check_set)):
        new_layers = []
        buf_f = open(filename, 'a+')

        for layer in net.layer:

            if layer.name in priv_conf["vbmf"]["cv"].keys():
                idx = layer_list.index(layer.name)
                g = layer.convolution_param.group
                bottom = layer.bottom[0]
                weight_a = priv_net.params[layer.name + '_v'][0].data
                weight_b = priv_net.params[layer.name + '_h'][0].data
                k = check_set[num][idx]*g
                if bottom == 'data':
                    a, b = cv_channel(layer, k)
                else :
                    a, b = cv_spatial(layer, k)
                new_layers.extend([a, b])
                new_conf["vbmf"]["cv"][layer.name] = k


            elif layer.name in priv_conf["vbmf"]["fc"].keys():
                idx = layer_list.index(layer.name)
                weight_a = priv_net.params[layer.name + '_v'][0].data
                weight_b = priv_net.params[layer.name + '_h'][0].data
                k = check_set[num][idx]
                a, b = fc_decomp_2d(layer, k)
                new_layers.extend([a, b])
                new_conf["vbmf"]["fc"][layer.name] = k

            else:
                new_layers.append(layer)
                continue

        new_net = NetParameter()
        new_net.CopyFrom(net)
        del(new_net.layer[:])
        new_net.layer.extend(new_layers)

        # File Write
        out = os.path.splitext(os.path.basename(output_file))[0] + '_{}_{}.prototxt'.format(i, num)
        out_dir = os.path.dirname(output_file)
        comp_file = '{}/{}'.format(out_dir, out )
        with open(comp_file, 'w') as fp:
            fp.write(pb.text_format.MessageToString(new_net))
        #print '[Layer] Wrote compressed prototxt to: {:s}'.format(comp_file)

        out = os.path.splitext(os.path.basename(output_weight))[0] + '_{}_{}.caffemodel'.format(i, num)
        out_dir = os.path.dirname(output_weight)
        comp_weight = '{}/{}'.format(out_dir, out)

        # Decomposition 
        decomp_weights(orig_net, num, net.layer, orig_file, orig_weight, new_conf, comp_file, comp_weight, priv_net, check_set, layer_list)

        # Accuarcy & Loss test 
        out = os.path.splitext(os.path.basename(log))[0] + '_{}_{}.log'.format(i, num)
        out_dir = os.path.dirname(log)
        comp_log = '{}/{}'.format(out_dir, out)

        tmp_acc_loss = call_caffe_test(comp_file, comp_weight, comp_log, txt_idx, net_type)

        acc[num] = tmp_acc_loss[0][0]
        loss[num] = tmp_acc_loss[0][1]

        buf_f.write('{} {} {} {}\n'.format(i, num, acc[num], loss[num]))
        buf_f.close()
        os.system('./remove_model.sh {} {}'.format(txt_idx+1,net_type))

    return acc, loss


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


def load_weights(orig_net, layer_name, lowrank_net, priv_net):
    b = 0
    v_bias = 0
    h_bias = 0
    vp_bias = 0
    hp_bias = 0
    if(len(orig_net.params[layer_name]) == 2):
        W, b = [p.data for p in orig_net.params[layer_name]]
        v_weights, v_bias = [p.data for p in lowrank_net.params[layer_name + '_v']]
        h_weights, h_bias = [p.data for p in lowrank_net.params[layer_name + '_h']]
        vp_weights, vp_bias = [p.data for p in priv_net.params[layer_name + '_v']]
        hp_weights, hp_bias = [p.data for p in priv_net.params[layer_name + '_h']]
        v_bias[...] = 0
        h_bias[...] = b.copy()
    else:
        W = orig_net.params[layer_name][0].data
        v_weights = lowrank_net.params[layer_name + '_v'][0].data
        h_weights = lowrank_net.params[layer_name + '_h'][0].data
        vp_weights = priv_net.params[layer_name + '_v'][0].data
        hp_weights = priv_net.params[layer_name + '_h'][0].data

    return W, b, v_weights, v_bias, h_weights, h_bias, vp_weights, vp_bias, hp_weights, hp_bias

##########################################################
# In this process, the network model is decomposed. 
# --------------------------------------------------------
def decomp_weights(orig_net, num, orig_layer, orig_model, orig_weights, conf, lowrank_model, lowrank_weights, priv_net, check_set, layer_list):

    lowrank_net = caffe.Net(lowrank_model, orig_weights, caffe.TEST) # empty space for reduce rank model

    for layer_name in conf["vbmf"]["cv"]:

        if not layer_name:
            print('cv layer is empty :(')
            break

        for layer in orig_layer:
            if layer.name == layer_name:
                orig_param = layer.convolution_param
                bottom = layer.bottom[0]
                print layer.name, layer_name
                continue

        idx = layer_list.index(layer_name)
        W, b, v_weights, v_bias, h_weights, h_bias, vp_weights, vp_bias, hp_weights, hp_bias = load_weights(orig_net, layer_name, lowrank_net, priv_net)

        # Get the shapes
        num_groups = orig_param.group
        N, C, D, D = W.shape
        N = N// num_groups
        k = v_weights.shape[0]//num_groups
        K = vp_weights.shape[0]//num_groups
        print('{} {}, K = {}, k = {}, group = {}'.format(layer_name, W.shape, K*num_groups, k*num_groups, num_groups))

        if bottom == 'data' : 
            for g in xrange(num_groups):
                v = vp_weights[K*g:K*(g+1)].reshape((K,C*D*D))
                v = v[:k,:]
                v = v.reshape((k, C, D, D))
                np.copyto(v_weights[k*g:k*(g+1)], v)

                h = hp_weights[N*g:N*(g+1)].reshape((N, K))
                h = h[:,:k]
                h = h.reshape((N, k, 1, 1))
                np.copyto(h_weights[N*g:N*(g+1)], h)

        else :
            for g in xrange(num_groups):
                v = vp_weights[K*g:K*(g+1)].transpose(1, 2, 3, 0).reshape((C*D, K))
                v = v[:,:k]
                v = v.reshape((C, D, 1, k)).transpose(3, 0, 1, 2)
                np.copyto(v_weights[k*g:k*(g+1)], v)

                h = hp_weights[N*g:N*(g+1)].transpose(1, 2, 3, 0).reshape((K, D*N))
                h = h[:k,:]
                h = h.reshape((k, 1, D, N)).transpose(3, 0, 1, 2)
                np.copyto(h_weights[N*g:N*(g+1)], h)


    for layer_name in conf["vbmf"]["fc"]:

        if not layer_name:
            print('fc layer is empty :(')
            break

        idx = layer_list.index(layer_name)
        W, b, v_weights, v_bias, h_weights, h_bias, vp_weights, vp_bias, hp_weights, hp_bias = load_weights(orig_net, layer_name, lowrank_net, priv_net)

        # Get the shapes
        N, C = W.shape
        num_groups = v_weights.shape[0] // h_weights.shape[1]
        N = N // num_groups
        k = v_weights.shape[0]//num_groups
        K = vp_weights.shape[0]//num_groups
        #K = h_weights.shape[1]

        print('{} {}, K = {}, k = {}, group = {}'.format(layer_name, W.shape, K*num_groups, k*num_groups, num_groups))
        # SVD approximation
        np.copyto(v_weights[:k], vp_weights[:k])
        np.copyto(h_weights[:N], hp_weights[:,:k])


    lowrank_net.save(lowrank_weights)
    #print '[Total] Wrote compressed weight to: {:s}'.format(lowrank_weights)
    out = os.path.splitext(os.path.basename(lowrank_weights))[0] 
    out_dir = os.path.dirname(lowrank_weights)


def call_caffe_test(prototxt, weight, log, txt_idx, net_type):

    # call caffe and test  (path must be absolute)
    os.system('./acc_loss_check.sh {} {} {} {} {}'.format(prototxt, weight, log, txt_idx, net_type))

    # read log file and take acc, loss
    results = []
    with open(log) as inputfile:
        for line in inputfile:
            results.append(line.strip().split('  '))

    return results


