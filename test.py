import torch
from lpcnet import LPCNet
import numpy as np
from ulaw import *
from tqdm import tqdm
import sys
import argparse
from scipy import io
from scipy.io import wavfile

parser = argparse.ArgumentParser(description='Test LPCNet')
parser.add_argument('--load', default='./ckpts/120.pkl', type=str, help='file of state dict')
parser.add_argument('--feat', default='../test_features.f32', type=str, help='input feature')
parser.add_argument('--file', default='./test.wav', type=str, help='output wav file')

args = parser.parse_args()

feature_file = args.feat
out_file = args.file

model.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
model = torch.nn.DataParallel(LPCNet(mode='test'))
model = model.cpu()


frame_size = 160
nb_features = 55
nb_used_features = 38

features = np.fromfile(feature_file, dtype='float32')
features = np.resize(features, (-1, nb_features))
nb_frames = 1
feature_chunk_size = features.shape[0]
pcm_chunk_size = frame_size*feature_chunk_size

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features[:,:,18:36] = 0
#periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')

periods = torch.Tensor((.1 + 50*features[:,:,36:37]+100)).type(torch.LongTensor)

order = 16

#pcm = np.zeros((nb_frames*pcm_chunk_size, ))
#fexc = np.zeros((1, 1, 3), dtype='int16')+128
pcm    = np.zeros(nb_frames*pcm_chunk_size)
fexc   = np.zeros((1, 1, 3), dtype='int16')+128
#state1 = np.zeros((1, model.rnn_units1), dtype='float32')
#state2 = np.zeros((1, model.rnn_units2), dtype='float32')
state1 = torch.zeros((1, 1, model.module.rnn_units1))
state2 = torch.zeros((1, 1, model.module.rnn_units2))

mem = 0
coef = 0.85

#fout = open(out_file, 'wb')

skip = order + 1
res = []
with torch.no_grad():
    for c in range(0, nb_frames):
        feat  = torch.FloatTensor(features[c:c+1, :, :nb_used_features])
        pitch = periods[c:c+1, :, :]
        cfeat = model.module.encode(feat, pitch)
        for fr in tqdm(range(0, feature_chunk_size)):
            f = c*feature_chunk_size + fr
            a = features[c, fr, nb_features-order:]
            for i in range(skip, frame_size):
                t = f*frame_size + i
                # float
                pred = -sum(a * pcm[t-1 : t-1-order:-1])
                fexc[0, 0, 1] = lin2ulaw(pred)
                p, state1, state2 = model.module.decode(torch.LongTensor(fexc), torch.Tensor(cfeat[:, fr:fr+1, :]), state1, state2, frame_size=1)
                p = p.softmax(dim=-1).numpy()
                p *= np.power(p, np.maximum(0, 1.5*features[c, fr, 37] - .5))
                p = p/(1e-18 + np.sum(p))
                p = np.maximum(p-0.002, 0).astype('float64')
                p = p/(1e-8 + np.sum(p))
                fexc[0, 0, 2] = np.argmax(np.random.multinomial(1, p[0,0,:], 1))
                pcm[t] = pred + ulaw2lin(fexc[0, 0, 2])
                fexc[0, 0, 0] = lin2ulaw(pcm[t])
                mem = coef*mem + pcm[t]
                res.append(round(mem))
            skip = 0

wavfile.write(out_file, rate=16000, data=np.array(res, dtype='int16'))