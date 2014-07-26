"""python stack_fbanks.py npz7_train/*.npy
"""

import sys
import numpy as np

NFRAMES = 7
b_a = (NFRAMES - 1) / 2
FRAMES_PER_SEC = 100  # features frames per second
FEATURES_RATE = 1. / FRAMES_PER_SEC

for fname in sys.argv[1:]:
    fbanks = np.load(fname)
    fbanks_s = np.zeros((fbanks.shape[0], fbanks.shape[1] * NFRAMES),
            dtype='float32')
    for i in xrange(b_a + 1):
        fbanks_s[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                (max(0, (b_a - i) * fbanks.shape[1]),
                    max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                'constant', constant_values=(0, 0))
    for i in xrange(b_a + 1, fbanks.shape[0] - b_a):
        fbanks_s[i] = fbanks[i - b_a:i + b_a + 1].flatten()
    for i in xrange(fbanks.shape[0] - b_a - 1, fbanks.shape[0]):
        fbanks_s[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                (max(0, (b_a - i) * fbanks.shape[1]),
                    max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                'constant', constant_values=(0, 0))
    time_table = np.zeros(fbanks_s.shape[0])
    for i in xrange(time_table.shape[0]):
        time_table[i] = float(i) / FRAMES_PER_SEC + FEATURES_RATE / 2
    np.savez(fname.split('.')[0] + '.npz',
            features=fbanks_s,
            time=time_table)

