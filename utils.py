from torch.utils.data import Dataset, DataLoader
from lpcnet import frame_size
import numpy as np

batch_size = 64
nb_epochs = 120
nb_features = 55
nb_used_features = 38
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

class LPCData(Dataset):
    def __init__(self, in_data, features, periods, out_exc):
        self.in_data  = in_data
        self.features = features
        self.periods  = periods
        self.out_exc  = out_exc
    def __getitem__(self, i):
        return (self.in_data[i], self.features[i], self.periods[i], self.out_exc[i])

    def __len__(self):
        return self.in_data.shape[0]

def get_data(pcm_file, feature_file):
    data = np.fromfile(pcm_file, dtype='uint8')
    nb_frames = len(data)//(4*pcm_chunk_size)
    features = np.fromfile(feature_file, dtype='float32')
    data = data[:nb_frames*4*pcm_chunk_size]
    features = features[:nb_frames*feature_chunk_size*nb_features]
    features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))
    sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
    pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
    in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
    out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))
    #print("ulaw std = ", np.std(out_exc))
    features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
    features = features[:, :, :nb_used_features]
    features[:,:,18:36] = 0
    fpad1 = np.concatenate([features[0:1, 0:2, :], features[:-1, -2:, :]], axis=0)
    fpad2 = np.concatenate([features[1:, :2, :], features[0:1, -2:, :]], axis=0)
    features = np.concatenate([fpad1, features, fpad2], axis=1)
    periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')
    #periods = np.minimum(periods, 255)
    in_data = np.concatenate([sig, pred, in_exc], axis=-1)
    dataset = LPCData(in_data, features, periods, out_exc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    del data, sig, pred, in_exc
    return dataloader
