"""python pca_fbanks_npz.py npy_for_pca/*.npy
"""

import sys
import numpy as np


for NFRAMES in [1, 7, 11]:
    b_a = (NFRAMES - 1) / 2
    FRAMES_PER_SEC = 100  # features frames per second
    FEATURES_RATE = 1. / FRAMES_PER_SEC

    all_stacked_fbanks = []
    for fname in sys.argv[1:]:
        fbanks = np.load(fname)
        if NFRAMES > 1:
            fbanks7 = np.zeros((fbanks.shape[0], fbanks.shape[1] * NFRAMES),
                    dtype='float32')
            for i in xrange(b_a + 1):
                fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                        (max(0, (b_a - i) * fbanks.shape[1]),
                            max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                        'constant', constant_values=(0, 0))
            for i in xrange(b_a + 1, fbanks.shape[0] - b_a):
                fbanks7[i] = fbanks[i - b_a:i + b_a + 1].flatten()
            for i in xrange(fbanks.shape[0] - b_a - 1, fbanks.shape[0]):
                fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                        (max(0, (b_a - i) * fbanks.shape[1]),
                            max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                        'constant', constant_values=(0, 0))
            all_stacked_fbanks.append(fbanks7)
        else:
            all_stacked_fbanks.append(fbanks)

    from sklearn.decomposition import TruncatedSVD

    pca_stacked = TruncatedSVD(n_components=NFRAMES*39)
    pca_stacked.fit(np.concatenate(all_stacked_fbanks, axis=0))

    for fname in sys.argv[1:]:
        fbanks = np.load(fname)
        if NFRAMES > 1:
            fbanks7 = np.zeros((fbanks.shape[0], fbanks.shape[1] * NFRAMES),
                    dtype='float32')
            for i in xrange(b_a + 1):
                fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                        (max(0, (b_a - i) * fbanks.shape[1]),
                            max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                        'constant', constant_values=(0, 0))
            for i in xrange(b_a + 1, fbanks.shape[0] - b_a):
                fbanks7[i] = fbanks[i - b_a:i + b_a + 1].flatten()
            for i in xrange(fbanks.shape[0] - b_a - 1, fbanks.shape[0]):
                fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                        (max(0, (b_a - i) * fbanks.shape[1]),
                            max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                        'constant', constant_values=(0, 0))
            time_table = np.zeros(fbanks7.shape[0])
            for i in xrange(time_table.shape[0]):
                time_table[i] = float(i) / FRAMES_PER_SEC + FEATURES_RATE / 2
            np.savez('pca_fbanks_'+str(NFRAMES)+'/'+fname.replace('_fbanks', '').split('/')[-1].split('.')[0] + '.npz',
                    features=pca_stacked.transform(fbanks7),
                    time=time_table)
        else:
            time_table = np.zeros(fbanks.shape[0])
            for i in xrange(time_table.shape[0]):
                time_table[i] = float(i) / FRAMES_PER_SEC + FEATURES_RATE / 2
            np.savez('pca_fbanks_'+str(NFRAMES)+'/'+fname.replace('_fbanks', '').split('/')[-1].split('.')[0] + '.npz',
                    features=pca_stacked.transform(fbanks),
                    time=time_table)

