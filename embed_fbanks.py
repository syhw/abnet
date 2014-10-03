"""python embed_fbanks.py timit_dtw_train_small_fbank7_ab_net_adadelta.pickle npz7_train npz_emb7_train
"""

import cPickle, sys, glob
import numpy as np
from nnet_archs import ABNeuralNet, DropoutABNeuralNet

NFEATURES = 40

with open(sys.argv[1], 'rb') as f:
    nnet = cPickle.load(f)

NFRAMES = nnet.layers_ins[0] / NFEATURES

in_fldr = sys.argv[2].rstrip('/') + '/'
out_fldr = sys.argv[3].rstrip('/') + '/'

transform = nnet.transform_x1()
tmp = np.load('mean_std_3.npz')
mean = np.tile(tmp['mean'], NFRAMES)
std = np.tile(tmp['std'], NFRAMES)

# TODO maybe normalize embedded features ???
for fname in glob.iglob(in_fldr + "*.npz"):
    npz = np.load(fname)
    X = np.asarray((npz['features'] - mean) / std, dtype='float32')
    embedded = transform(X)
    np.savez(out_fldr + fname.split('/')[-1],
            features=embedded,
            time=npz['time'])

