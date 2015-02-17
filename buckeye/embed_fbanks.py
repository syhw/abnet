"""python embed_fbanks.py nnet.pickle npz11_test npz_emb11_wrd_test npz_emb11_spkr_test
"""

import cPickle, sys, glob
import numpy as np
from nnet_archs import ABNeuralNet2Outputs

NFEATURES = 40

with open(sys.argv[1], 'rb') as f:
    nnet = cPickle.load(f)

NFRAMES = nnet.layers_ins[0] / NFEATURES

in_fldr = sys.argv[2].rstrip('/') + '/'
out_fldr1 = sys.argv[3].rstrip('/') + '/'
out_fldr2 = sys.argv[4].rstrip('/') + '/'

transform = nnet.transform_x1()
tmp = np.load('mean_std_spkr_word.npz')
mean = np.tile(tmp['mean'], NFRAMES)
std = np.tile(tmp['std'], NFRAMES)

# TODO maybe normalize embedded features ???
for fname in glob.iglob(in_fldr + "*.npz"):
    npz = np.load(fname)
    X = np.asarray((npz['features'] - mean) / std, dtype='float32')
    emb_wrd, emb_spkr = transform(X)
    np.savez(out_fldr1 + fname.split('/')[-1],
            features=emb_wrd,
            time=npz['time'])
    np.savez(out_fldr2 + fname.split('/')[-1],
            features=emb_spkr,
            time=npz['time'])
