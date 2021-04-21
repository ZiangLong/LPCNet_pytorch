import numpy as np
import torch
import torch.nn as nn
from lpcnet import *
from torch.nn import Embedding, Linear, Conv1d, GRU, Module


max_rnn_neurons = 1
max_conv_inputs = 1
max_mdense_tmp = 1


CNAME = {
    'emb_pcm'       :   'emb_sig',
    'emb_pitch'     :   'emb_pitch',
    'conv1'         :   'feature_conv1',
    'conv2'         :   'feature_conv2',
    'dense1'        :   'feature_dense1',
    'dense2'        :   'feature_dense2',
    'gru1'          :   'gru_a',
    'gru2'          :   'gru_b',
    'md'            :   'dual_fc',
}

# helper
def printVector(f, vector, name, dtype='float'):
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const {} {}[{}] = {{\n   '.format(dtype, name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(v[i]))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break;
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    return;

# helper
def printSparseVector(f, A, name):
    N = A.shape[0]
    W = np.zeros((0,))
    diag = np.concatenate([np.diag(A[:,:N]), np.diag(A[:,N:2*N]), np.diag(A[:,2*N:])])
    A[:,:N] = A[:,:N] - np.diag(np.diag(A[:,:N]))
    A[:,N:2*N] = A[:,N:2*N] - np.diag(np.diag(A[:,N:2*N]))
    A[:,2*N:] = A[:,2*N:] - np.diag(np.diag(A[:,2*N:]))
    printVector(f, diag, name + '_diag')
    idx = np.zeros((0,), dtype='int')
    for i in range(3*N//16):
        pos = idx.shape[0]
        idx = np.append(idx, -1)
        nb_nonzero = 0
        for j in range(N):
            if np.sum(np.abs(A[j, i*16:(i+1)*16])) > 1e-10:
                nb_nonzero = nb_nonzero + 1
                idx = np.append(idx, j)
                W = np.concatenate([W, A[j, i*16:(i+1)*16]])
        idx[pos] = nb_nonzero
    printVector(f, W, name)
    #idx = np.tile(np.concatenate([np.array([N]), np.arange(N)]), 3*N//16)
    printVector(f, idx, name + '_idx', dtype='int')
    return;

# default
def dump_layer_ignore(self, f, hf, key):
    print("ignoring layer " + self.name + " of type " + self.__class__.__name__)
    return False
Module.dump_layer = dump_layer_ignore

# Sparse GRU, only use once
def dump_sparse_gru(self, f, hf, key):
    global max_rnn_neurons
    name = 'sparse_' + CNAME[key]
    print("printing layer " + name + " of type sparse " + self.__class__.__name__)
    W1 = self.weight_ih_l0.data.transpose(1,0).detach().numpy()
    W2 = self.weight_hh_l0.data.transpose(1,0).detach().numpy()
    b1 = self.bias_ih_l0.data.unsqueeze(0)
    b2 = self.bias_hh_l0.data.unsqueeze(0)
    b  = torch.cat((b1, b2), dim=0).detach().numpy()
    printSparseVector(f, W2, name + '_recurrent_weights')
    printVector(f, b, name + '_bias')
    activation = 'TANH'
    reset_after = 1
    neurons = W.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const SparseGRULayer {} = {{\n   {}_bias,\n   {}_recurrent_weights_diag,\n   {}_recurrent_weights,\n   {}_recurrent_weights_idx,\n   {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, name, W.shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), W.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), W.shape[1]//3))
    hf.write('extern const SparseGRULayer {};\n\n'.format(name));
    return True

# GRU
def dump_gru_layer(self, f, hf, key):
    global max_rnn_neurons
    name = CNAME[key]
    print("printing layer " + name + " of type " + self.__class__.__name__)
    W1 = self.weight_ih_l0.data.transpose(1,0).detach().numpy()
    W2 = self.weight_hh_l0.data.transpose(1,0).detach().numpy()
    b1 = self.bias_ih_l0.data.unsqueeze(0)
    b2 = self.bias_hh_l0.data.unsqueeze(0)
    b  = torch.cat((b1, b2), dim=0).detach().numpy()
    printVector(f, W1, name + '_weights')
    printVector(f, W2, name + '_recurrent_weights')
    printVector(f, b, name + '_bias')
    activation = 'TANH'
    reset_after = 1
    neurons = W1.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, W1.shape[0], W1.shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), W1.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), W1.shape[1]//3))
    hf.write('extern const GRULayer {};\n\n'.format(name))
    return True
GRU.dump_layer = dump_gru_layer

def dump_dense_layer_impl(name, weights, bias, activation, f, hf):
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));

def dump_dense_layer(self, f, hf, key):
    name = CNAME[key]
    print("printing layer " + name + " of type " + self.__class__.__name__)
    w, b = self.weight.data.transpose(1, 0), self.bias.data
    w, b = w.detach().numpy(), b.detach().numpy()
    dump_dense_layer_impl(name, w, b, 'TANH', f, hf)
    return False

Linear.dump_layer = dump_dense_layer

def dump_mdense_layer(self, f, hf, key):
    global max_mdense_tmp
    name = CNAME[key]
    print("printing layer " + name + " of type " + self.__class__.__name__)
    w1, w2 = self.fc1.weight.data, self.fc2.weight.data
    W = torch.cat((w1.unsqueeze(2), w2.unsqueeze(2)), dim=2).detach().numpy()
    b1, b2 = self.fc1.bias.data, self.fc2.bias.data
    b = torch.cat((b1.unsqueeze(1), b2.unsqueeze(1)), dim=1).detach().numpy()
    gamma1, gamma2 = self.gamma1.data, self.gamma2.data
    gamma = torch.cat((gamma1.unsqueeze(1), gamma2.unsqueeze(1)), dim=1).detach().numpy()
    printVector(f, np.transpose(W, (1, 2, 0)), name + '_weights')
    printVector(f, np.transpose(b, (1, 0)), name + '_bias')
    printVector(f, np.transpose(gamma, (1, 0)), name + '_factor')
    activation = 'TANH'
    max_mdense_tmp = max(max_mdense_tmp, W.shape[0] * W.shape[2])
    f.write('const MDenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_factor,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, name, W.shape[1], W.shape[0], W.shape[2], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), W.shape[0]))
    hf.write('extern const MDenseLayer {};\n\n'.format(name));
    return False
MDense.dump_layer = dump_mdense_layer

def dump_conv1d_layer(self, f, hf, key):
    global max_conv_inputs
    name = CNAME[key]
    print("printing layer " + name + " of type " + self.__class__.__name__)
    W = self.weight.data.transpose(2, 0)
    b = self.bias.data
    W, b = W.detach().numpy(), b.detach().numpy()
    printVector(f, W, name + '_weights')
    printVector(f, b, name + '_bias')
    activation = 'TANH'
    max_conv_inputs = max(max_conv_inputs, W.shape[1]*W.shape[0])
    f.write('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, W.shape[1], W.shape[0], W.shape[2], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), W.shape[2]))
    hf.write('#define {}_STATE_SIZE ({}*{})\n'.format(name.upper(), W.shape[1], (W.shape[0]-1)))
    hf.write('#define {}_DELAY {}\n'.format(name.upper(), (W.shape[0]-1)//2))
    hf.write('extern const Conv1DLayer {};\n\n'.format(name));
    return True
Conv1d.dump_layer = dump_conv1d_layer

# helper
def dump_embedding_layer_impl(name, weights, f, hf):
    printVector(f, weights, name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));

def dump_embedding_layer(self, f, hf, key):
    name = CNAME[key]
    print("printing layer " + name + " of type " + self.__class__.__name__)
    W = self.weight.data.detach().numpy()
    dump_embedding_layer_impl(name, W, f, hf)
    return False
Embedding.dump_layer = dump_embedding_layer


model = torch.nn.DataParallel(LPCNet())
model.load_state_dict(torch.load('./ckpts/120.pkl', map_location=torch.device('cpu')))
model = model.cpu()
net = model.module

cfile = 'nnet_data.c'
hfile = 'nnet_data.h'


f = open(cfile, 'w')
hf = open(hfile, 'w')


f.write('/*This file is automatically generated from a Keras model*/\n\n')
f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "nnet.h"\n#include "{}"\n\n'.format(hfile))

hf.write('/*This file is automatically generated from a Keras model*/\n\n')
hf.write('#ifndef RNN_DATA_H\n#define RNN_DATA_H\n\n#include "nnet.h"\n\n')

E = net.emb_pcm.weight.data.detach().numpy()
W = net.gru1.weight_ih_l0.data.transpose(1, 0)[:embed_size, :].detach().numpy()
dump_embedding_layer_impl('gru_a_embed_sig', np.dot(E, W), f, hf)
W = net.gru1.weight_ih_l0.data.transpose(1, 0)[embed_size:2*embed_size,:].detach().numpy()
dump_embedding_layer_impl('gru_a_embed_pred', np.dot(E, W), f, hf)
W = net.gru1.weight_ih_l0.transpose(1, 0)[2*embed_size:3*embed_size,:].detach().numpy()
dump_embedding_layer_impl('gru_a_embed_exc', np.dot(E, W), f, hf)
b1, b2 = net.gru1.bias_ih_l0.data, net.gru1.bias_hh_l0.data
b = torch.cat((b1.unsqueeze(0), b2.unsqueeze(0)), dim=0).detach().numpy()
dump_dense_layer_impl('gru_a_dense_feature', W, b, 'LINEAR', f, hf)


layer_list = []
for key in net._modules:
    if net._modules[key].dump_layer(f, hf, key):
        layer_list.append(key)

dump_sparse_gru(net.gru1, f, hf, 'gru1')

hf.write('#define MAX_RNN_NEURONS {}\n\n'.format(max_rnn_neurons))
hf.write('#define MAX_CONV_INPUTS {}\n\n'.format(max_conv_inputs))
hf.write('#define MAX_MDENSE_TMP {}\n\n'.format(max_mdense_tmp))


hf.write('typedef struct {\n')
for i, key in enumerate(layer_list):
    name = CNAME[key]
    hf.write('  float {}_state[{}_STATE_SIZE];\n'.format(name, name.upper())) 
hf.write('} NNetState;\n')

hf.write('\n\n#endif\n')

f.close()
hf.close()
