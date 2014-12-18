import joblib, glob, os
from joblib import Parallel, delayed
from itertools import izip
from functools import partial
from collections import defaultdict
import numpy as np
from multiprocessing import cpu_count
from dtw import DTW
from spectral import Spectral
from scipy.io import wavfile

FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40 # number of filterbanks to use
basedir = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/buckeye_modified_split_devtest/"

def do_fbank(fname):
    srate, sound = wavfile.read(fname)
    fbanks = Spectral(nfilt=N_FBANKS,    # nb of filters in mel bank
                 alpha=0.97,             # pre-emphasis
                 do_dct=False,           # we do not want MFCCs
                 fs=srate,               # sampling rate
                 frate=FBANKS_RATE,      # frame rate
                 wlen=FBANKS_WINDOW,     # window length
                 nfft=1024,              # length of dft
                 do_deltas=False,       # speed
                 do_deltasdeltas=False  # acceleration
                 )
    fb = fbanks.transform(sound)
    print "did:", fname
    #print fbnk.shape
    return fb



if __name__ == "__main__":
    for dset in ['test', 'dev']:
        for bdir, _, files in os.walk(basedir + 'wrd/' + dset + '/'):
            for fname in files:
                if fname[-4:] != '.wrd':
                    continue
                wrdfname = bdir + fname
                wavfname = wrdfname.replace('wrd', 'wav')
                with open(wavfname[:-3] + 'npy', 'wb') as wf:
                    np.save(wf, do_fbank(wavfname))


