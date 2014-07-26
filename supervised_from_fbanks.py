"""python supervised_from_fbanks.py supervised_timit_fbank11_nnet_adadelta.pickle npz11_train npz_sup11_train
"""

import cPickle, sys, glob
import numpy as np
from nnet_archs import NeuralNet
import theano
from theano import tensor as T

NFEATURES = 40

with open(sys.argv[1], 'rb') as f:
    nnet = cPickle.load(f)

NFRAMES = nnet.layers_ins[0] / NFEATURES

in_fldr = sys.argv[2].rstrip('/') + '/'
out_fldr = sys.argv[3].rstrip('/') + '/'

batch_x = T.fmatrix('batch_x')
transform = theano.function(inputs=[theano.Param(batch_x)],
        outputs=nnet.layers[-1].p_y_given_x,
        updates={},
        givens={nnet.x: batch_x})

tmp = np.load('mean_std.npz')
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

