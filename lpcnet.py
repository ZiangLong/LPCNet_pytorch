import torch
import torch.nn as nn
import math

frame_size = 160
pcm_bits = 8
embed_size = 128
pcm_levels = 2**pcm_bits


def _init_linear_conv_(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)

def _init_gru_(layer):
    nn.init.xavier_uniform_(layer.weight_ih_l0)
    nn.init.orthogonal_(layer.weight_hh_l0)
    nn.init.zeros_(layer.bias_ih_l0)
    nn.init.zeros_(layer.bias_hh_l0)

class MDense(nn.Module):
    def __init__(self, in_dim=16, out_dim=256):
        super(MDense, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.gamma1 = nn.Parameter(torch.ones(out_dim), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(out_dim), requires_grad=True)
        _init_linear_conv_(self.fc1)
        _init_linear_conv_(self.fc2)
    def forward(self, x):
        y1 = self.fc1(x).tanh()
        y2 = self.fc2(x).tanh()
        p  = self.gamma1 * y1 + self.gamma2 * y2
        return p

class LPCNet(nn.Module):
    def __init__(self, rnn_units1=384, rnn_units2=16, nb_used_features=38, mode='prepad'):
        super(LPCNet, self).__init__()
        self.rnn_units1 = rnn_units1
        self.rnn_units2 = rnn_units2
        padding = 0 if mode=='prepad' else 1
        self.emb_pcm    = nn.Embedding(256, 128)
        self.emb_pitch  = nn.Embedding(256, 64)
        self.conv1      = nn.Conv1d(102, 128, 3, padding=padding)
        self.conv2      = nn.Conv1d(128, 128, 3, padding=padding)
        self.dense1     = nn.Linear(128, 128)
        self.dense2     = nn.Linear(128, 128)
        self.gru1       = nn.GRU(input_size=512, hidden_size=rnn_units1, batch_first=True)
        self.gru2       = nn.GRU(input_size=512, hidden_size=rnn_units2, batch_first=True)
        self.md         = MDense(in_dim=16, out_dim=256)
        self.pcminit(self.emb_pcm)
        nn.init.uniform_(self.emb_pitch.weight, a=-0.05, b=0.05)
        _init_linear_conv_(self.conv1)
        _init_linear_conv_(self.conv2)
        _init_linear_conv_(self.dense1)
        _init_linear_conv_(self.dense2)
        _init_gru_(self.gru1)
        _init_gru_(self.gru2)

    def pcminit(self, layer):
        w = layer.state_dict()['weight']
        shape = w.shape
        num_rows, num_cols = shape
        flat_shape = (num_rows, num_cols)
        p = torch.rand(shape).add_(-0.5).mul_(1.7321 * 2)
        r = torch.arange(-.5*num_rows+.5,.5*num_rows-.4).mul(math.sqrt(12)/num_rows).reshape(num_rows, 1)
        w[:] = p + r

    def encode(self, feat, pitch):
        # (bs, 15/19, 1) --> (bs, 15/19, 1, 64) --> (bs, (15/19), 64)
        pitch  = self.emb_pitch(pitch).squeeze(2)
        # (bs, (15/19), 38+64) --> (bs, 102, 15/19)
        feat   = torch.cat((feat, pitch), dim=2).permute(0, 2, 1)
        # (bs, 102, 15/19) --> (bs, 128, 15/17)
        feat   = self.conv1(feat).tanh()
        # (bs, 128, 15/17) --> (bs, 128, 15) --> (bs, 15, 128)
        feat   = self.conv2(feat).tanh().permute(0, 2, 1)
        # (bs, 15, 128) --> (bs, 15, 128)
        feat   = self.dense2(self.dense1(feat).tanh()).tanh()
        return feat

    def decode(self, pcm, feat, hid1, hid2, frame_size=160):
        # bs, cs = batch size, chunk size
        bs, cs = pcm.shape[0], pcm.shape[1]
        # (bs, 2400, 3) --> (bs, 2400, 3, 128) --> (bs, 2400, 384)
        pcm    = self.emb_pcm(pcm).reshape(bs, cs, -1)
        # (bs, 15, 128) --> (bs, 2400, 128)
        rfeat  = torch.repeat_interleave(feat, frame_size, dim=1) if frame_size > 1 else feat
        # (bs, 2400, 512)
        rnn_in = torch.cat((pcm, rfeat), dim=-1)
        # (bs, 2400, 512) --> (bs, 2400, 384)
        self.gru1.flatten_parameters()
        hid1   = hid1.to(rnn_in)
        out, hid1 = self.gru1(rnn_in, hid1)
        # (bs, 2400, 384+128)
        rnn_in = torch.cat((out, rfeat), dim=-1)
        # (bs, 2400, 16)
        self.gru2.flatten_parameters()
        hid2   = hid2.to(rnn_in)
        out, hid2 = self.gru2(rnn_in, hid2)
        prob   = self.md(out)
        return prob, hid1, hid2

    def forward(self, pcm, feat, pitch):
        # bs, cs = batch size, chunk size
        bs, cs = pcm.shape[0], pcm.shape[1]
        feat = self.encode(feat, pitch)
        # (1, bs, 384)
        zeros1 = torch.zeros(1, bs, self.rnn_units1)
        # (1, bs, 16)
        zeros2 = torch.zeros(1, bs, self.rnn_units2)
        prob, hid1, hid2 = self.decode(pcm, feat, zeros1, zeros2)
        return prob
