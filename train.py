import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

t_start = 2000
t_end   = 40000
interval= 400
final_density = (0.05, 0.05, 0.2)

def sparsity(net, i):
    if i < t_start:
        return
    if (i - t_start) % interval != 0 and i < t_end:
        return
    p = net.module.gru1.state_dict()['weight_hh_l0'].transpose(1, 0)
    nb = p.shape[1] // p.shape[0]
    N = p.shape[0]
    for k in range(nb):
        density = final_density[k]
        if i < t_end:
            r = 1.0 - (i - t_start) / (t_end - t_start)
            density = 1 - (1 - final_density[k]) * (1-r*r*r)
        A = p[:, k*N:(k+1)*N]
        A = A - torch.diag(torch.diag(A))
        L = torch.reshape(A, (N, N//16, 16))
        S = torch.sum(L*L, axis=-1)
        SS, _ = torch.sort(torch.reshape(S, (-1,)))
        thresh = SS[round(N*N//16*(1-density))]
        mask = (S>=thresh).float()
        mask = torch.repeat_interleave(mask, 16, axis=1)
        mask.add_(torch.eye(N).cuda()).clamp_(max=1)
        p[:, k*N:(k+1)*N] = p[:, k*N:(k+1)*N]*mask
    net.module.gru1.state_dict()['weight_hh_l0'][:] = p.transpose(1, 0)

def train(net, dataloader, loss, lr=0.001, epochs=120):
    opt = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99), eps=1e-7)
    iteration = 0
    for epoch in range(1, epochs+1):
        print('Epoch:\t'+str(epoch)+'/'+str(epochs))
        total_loss = []
        for i, (pcm, feat, pitch, target) in enumerate(tqdm(dataloader), 1):
            iteration += 1
            for pg in opt.param_groups:
                pg['lr'] = lr / (1. + 5e-5 * iteration)
            pcm    = pcm.type(torch.LongTensor).cuda()
            feat   = feat.cuda()
            pitch  = pitch.type(torch.LongTensor).cuda()
            target = target.type(torch.LongTensor).reshape(-1).cuda()
            prob   = net(pcm, feat, pitch).reshape(-1, 256)
            L      = loss(prob, target)
            opt.zero_grad()
            L.backward()
            opt.step()
            sparsity(net, iteration)
            total_loss.append(L.item())
        avg_loss = sum(total_loss)/len(total_loss)
        print('\nEpoch Loss %.4f' % avg_loss)
        torch.save(net.state_dict(), './ckpts/%03d.pkl' % epoch)
            

            
