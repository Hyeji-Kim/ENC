
import numpy as np
import json
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from pprint import pprint

import google.protobuf as pb
import google.protobuf.text_format as pbt
import VBMF


current_path = os.getcwd()
CAFFE_ROOT = current_path+'/caffe/'
if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))

sys.path.insert(0, osp.join('/home/hjkim/ssd'))
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

def EVBMF_layer(W, layer_name):

    sigamp2 = 2
    noiseamp2 = 1
    # matrix factorization
    L,M = W.shape
    if L > M:
        W = W.T
    U1, S1, V1, post1 = VBMF.EVBMF(W)
    cacb   = post1["cacb"][0]
    sigma2 = post1["sigma2"]
    U1, S1, V1, post1 = VBMF.VBMF(W, cacb, sigma2)

    return S1.shape[0]

def Cost_orig(W, layer, orig_net):
    g = layer.convolution_param.group
    N, C, D, D = np.asarray(W.shape,dtype=float)
    D1, D2, D3, D4 = orig_net.blobs[layer.name].data.shape
    param = (N/g*C*D*D)*g
    oper =  param*D3*D4
    return oper, param

def Cost_spatial(W, layer, orig_net, k):
    g = layer.convolution_param.group
    N, C, D, D = np.asarray(W.shape,dtype=float)
    D1, D2, D3, D4 = orig_net.blobs[layer.name].data.shape
    param = (N/g*D + D*C)*g*k
    oper =  param*D3*D4
    return oper, param

def Cost_channel(W, layer, orig_net, k):
    g = layer.convolution_param.group
    N, C, D, D = np.asarray(W.shape,dtype=float)
    D1, D2, D3, D4 = orig_net.blobs[layer.name].data.shape
    param = (N/g + C*D*D)*g*k
    oper =  param*D3*D4
    return oper, param


def oper_param_cal(P, net, orig_net, args, sel_conv):

    filename = '../../base_models/{}/base/Net_def.txt'.format(args.model)
    f4 = open(filename, 'w')

    operation = 0
    operation_c = 0
    operation_f = 0
    param = 0
    param_c = 0
    param_f = 0



    C_norm = np.empty([len(P),200],dtype=float)
    W_norm = np.empty([len(P),200],dtype=float)
    R_norm = np.empty([len(P),200],dtype=float)
    for i in range(len(P)):
        j = 0
        orig_param = 0
        orig_param_c = 0
        orig_param_f = 0
        orig_oper = 0
        orig_oper_c = 0
        orig_oper_f = 0

        for layer in net.layer:

            if layer.type == 'Convolution' :
                
                a=layer.name
                D1, D2, D3, D4 = orig_net.blobs[layer.name].data.shape
                stride = layer.convolution_param.stride[0] if len(np.shape(layer.convolution_param.stride))>0 else 1
                #if len(stride) == 0 :
                #    stride = 1

                bottom = layer.bottom[0]

                if(len(orig_net.params[layer.name]) == 2):
                    W, b = [p.data for p in orig_net.params[layer.name]]
                else:
                    W = orig_net.params[layer.name][0].data

                # Get the shapes
                g = layer.convolution_param.group
                N, C, D, D = np.asarray(W.shape,dtype=float)

                oper_layer, param_layer = Cost_orig(W, layer, orig_net)
                orig_oper += oper_layer 
                orig_oper_c += oper_layer
                orig_param += param_layer
                orig_param_c += param_layer
#                print('{} {}'.format(oper_layer,param_layer))

                k = P[i][j]
                #k = k - k%g
                if 'proj' in layer.name :
                    oper_layer, param_layer = Cost_orig(W, layer, orig_net)
                    operation += oper_layer
                    operation_c += oper_layer
                    param += param_layer
                    param_c += param_layer
                    #print 'Proj Complexity : ',oper_layer, param_layer
                    continue

                if bottom == 'data' :
                    oper_layer, param_layer = Cost_channel(W, layer, orig_net, k)

                else :
                    oper_layer, param_layer = Cost_spatial(W, layer, orig_net, k)

#                C(1)   R(2) T(3) g(4) Fm(5) K(6) Weight(7)     FLOPs(8)
                operation += oper_layer
                operation_c += oper_layer
                param += param_layer
                param_c += param_layer
                C_norm[i][j] = oper_layer
                W_norm[i][j] = param_layer

                f4.write('{} {} {} {} {} {} {} {} {}\n'.format(int(C),k,int(N),g,D3,int(D),int(param_layer),int(oper_layer),stride))
#                print('"{}" : {}, {}, {}, {},'.format(a,k,N,C,D))
                j += 1

            elif (layer.type == 'InnerProduct') and args.conv == 0 :
                a=layer.name
                D=1
                D3=1
                stride=1
                g=1

                if(len(orig_net.params[layer.name]) == 2):
                    W, b = [p.data for p in orig_net.params[layer.name]]
                else:
                    W = orig_net.params[layer.name][0].data

                # Get the shapes
                N, C = np.asarray(W.shape,dtype=float)

                orig_oper += (N*C)
                orig_oper_f += (N*C)
                orig_param += (N*C)
                orig_param_f += (N*C)

                k = P[i][j]
                oper_layer = (N*k) + (k*C)
                param_layer = (N*k) + (k*C)
                operation += oper_layer
                operation_f += oper_layer
                param += param_layer
                param_f += param_layer

                C_norm[i][j] = oper_layer
                W_norm[i][j] = param_layer

                j += 1

                f4.write('{} {} {} {} {} {} {} {} {}\n'.format(int(C),k,int(N),g,D3,D,int(param_layer),int(oper_layer),stride))
#                print('"{}" : {}, {}, {}, {},'.format(a,k,N,C,D))

    C_norm = C_norm[:,:j]
    W_norm = W_norm[:,:j]
    R_norm = R_norm[:,:j]

    f4.close()
    print '[Done] Net_def.txt'
#    print('\ntotal Operations(all, conv, fc) = {} {} {}'.format(orig_oper, orig_oper_c, orig_oper_f))
#    print('total Params (all, conv, fc)= {} {} {}\n'.format(orig_param, orig_param_c, orig_param_f))

#    print('Reduced Operations(all, conv, fc) = {} {} {}'.format(operation, operation_c, operation_f))
#    print('Reduced Params(all, conv, fc) = {} {} {}\n'.format(param, param_c, param_f))


    return C_norm, W_norm, orig_oper, orig_param

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
    
    args.conv = 1 # 1 : only conv / 0 : all

    # load the deploy file
    net = load_file(args.orig_prototxt)
    orig_net = caffe.Net(args.orig_prototxt, args.orig_weight, caffe.TEST)
    conf_rmax = load_config('../../base_models/{}/config.json.rmax'.format(args.model))

    filename = '../../base_models/{}/base/R_list.txt'.format(args.model)
    f = open(filename, 'w')

    Rmax = np.empty([200,],dtype=float)
    vbmf = np.empty([200,],dtype=float)

    # make the rank list ---------------------------
    i = 0
    j = 0
    print '\ncalculating a rank using VBMF :'
    for layer in net.layer:
        if layer.type == 'Convolution' :
            a=layer.name
            g = layer.convolution_param.group
            bottom = layer.bottom[0]

            if(len(orig_net.params[layer.name]) == 2):
                W, b = [p.data for p in orig_net.params[layer.name]]
            else:
                W = orig_net.params[layer.name][0].data
            N,C,D,D = np.asarray(W.shape,dtype=float)
            if 'proj' in layer.name :
                continue
            print '-- layer = ',layer.name
            tmp_vbmf = np.empty(g)
            if bottom == 'data':
                for k in range(g):
                    W_ = W[int(N)*k:int(N)*(k+1)].reshape((int(N), int(C*D*D)))
                    tmp_vbmf[k] = EVBMF_layer(W_, layer.name)
            else:
                for k in range(g):
                    W_ = W[int(N)*k:int(N)*(k+1)].transpose(1,2,3,0).reshape((int(C*D), int(D*N)))
                    tmp_vbmf[k] = EVBMF_layer(W_, layer.name)
            vbmf[i] = max(tmp_vbmf)
            Rmax[i] = conf_rmax["vbmf"]["cv"][layer.name]
            i += 1
  #          f.write('{} '.format(a))

        elif (layer.type == 'InnerProduct') and (args.conv == 0):
            a=layer.name
            if(len(orig_net.params[layer.name]) == 2):
                W, b = [p.data for p in orig_net.params[layer.name]]
            else:
                W = orig_net.params[layer.name][0].data

            # Get the shapes
            N, C = np.asarray(W.shape,dtype=float)
            Rmax[i] = conf_rmax["vbmf"]["fc"][layer.name]
            vbmf[i] = EVBMF_layer(W, layer.name)
            i += 1
  #              f.write('{} '.format(a))

    L = np.copy(i)
    Rmax = Rmax[:i]
    vbmf = vbmf[:i]
    #f.write('\n')
    Rmax = np.around(Rmax)

    P = np.empty([5,len(Rmax)],dtype=float)
    P[4] = Rmax
    P[3] = np.around(vbmf/2)
    P[2] = np.around(vbmf/4)
    P[0] = np.around(Rmax/10)
    P[1] = np.around((P[2]+P[0])/2)

    for i in range(len(P)):
        for j in range(np.shape(P)[1]):
            P[i][j] = max(P[i][j],P[0,j])
            f.write('{} '.format(int(round(max(P[i][j],1),0))))
        f.write('\n')
    f.close()
    print '[Done] R_list.txt'
    P = np.array([[int(round(max(P[i][j],1),0)) for j in range(np.shape(P)[1])] for i in range(len(P))])

    filename = '../../base_models/{}/base/Rset.txt'.format(args.model)
    frset = open(filename, 'w')

    set_num = L*(len(P)-1)+1
    Rset = np.asarray([Rmax.astype(int)]*set_num)
    for i in range(L):
        Rset[int(i*(len(P)-1)):int((i+1)*(len(P)-1)),i] = P[:-1,i]

    for i in range(len(Rset)):
        for j in range(L):
            frset.write('{} '.format(int(Rset[i,j])))
        frset.write('\n')
    frset.close()
    print '[Done] Rset.txt'

    # calculate the cost from the list  ---------------------------

    filename = '../../base_models/{}/base/C_norm.txt'.format(args.model)
    f1 = open(filename, 'w')
    filename = '../../base_models/{}/base/W_norm.txt'.format(args.model)
    f2 = open(filename, 'w')
    filename = '../../base_models/{}/base/R_norm.txt'.format(args.model)
    f3 = open(filename, 'w')


    sel_conv = args.conv
    C_norm, W_norm, orig_oper, orig_param = oper_param_cal(P, net, orig_net, args, sel_conv)

    C_norm = C_norm/orig_oper
    W_norm = W_norm/orig_param
    R_norm = [P[i]/Rmax for i in range(len(P))]
    for i in range(len(P)):
        for j in range(np.shape(P)[1]) :
            f1.write('{} '.format(C_norm[i][j]))
            f2.write('{} '.format(W_norm[i][j]))
            f3.write('{} '.format(R_norm[i][j]))

        f1.write('\n')
        f2.write('\n')
        f3.write('\n')

    f1.close()
    f2.close()
    f3.close()
    print '[Done] C_norm.txt'
    print '[Done] W_norm.txt'
    print '[Done] R_norm.txt'

    # calculation for Rmax ----------------
    #P = P[-1,:].reshape([1,np.shape(P)[1]])
    #print 'last Rset = ',P
    #sel_conv = 1 # write with fc layer
    #C_norm, W_norm, orig_oper, orig_param = oper_param_cal(P, net, orig_net, args, sel_conv)
    # C(1)   R(2) T(3) g(4) Fm(5) K(6) Weight(7)     FLOPs(8)

if __name__ == '__main__':
    parser = ArgumentParser(description="Low-rank approximation")

    parser.add_argument('--orig_prototxt')
    parser.add_argument('--orig_weight')
    parser.add_argument('--config')
    parser.add_argument('--rank')
    parser.add_argument('--conv')
    parser.add_argument('--model')

    args = parser.parse_args()
    main(args)



