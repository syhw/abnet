from dtw import DTW
import numpy as np
import joblib, glob, os, sys
from spectral import Spectral
from scipy.io import wavfile
MAX_LENGTH_WORDS = 6     # in phones
MIN_LENGTH_WORDS = 6     # in phones
MIN_FRAMES = 5           # in speech frames
#FBANKS_TIME_STEP = 0.01  # in seconds
FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40 # number of filterbanks to use

bdir = '/fhgfs/bootphon/scratch/gsynnaeve/zerospeech/english_wavs/'


class Memoize:
    """Memoize(fn) 
    Will only work on functions with non-mutable arguments
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args):
        if not self.memo.has_key(args):
            self.memo[args] = self.fn(*args)
        return self.memo[args]


@Memoize
def do_fbank(fname):
    fn = bdir + fname + '.wav'
    try:
        with open(fn[:-3] + 'npy', 'rb') as rfb:
            fb = np.load(rfb)
    except IOError:
        srate, sound = wavfile.read(fn)
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
        fb = np.array(fbanks.transform(sound), dtype='float32')
    print "did:", fn
    #print fb.shape
    return fb

pairs = []
same_spkrs = 0
diff_spkrs = 0
with open("SysE.zs") as rf:
    cword = ''
    fs = []
    for line in rf:
        l = line.rstrip('\n')
        if l == '':
            continue
        if "Class" in line:
            cword = l.split()[1]
            fs = []
        else:
            fname, start, end = l.split()
            start = int(float(start) * FBANKS_RATE)
            end = int(float(end) * FBANKS_RATE)
            tmp = do_fbank(fname)[start:end+1]
            for (fname2, tmp2) in fs:
                dtw = DTW(tmp, tmp2, return_alignment=1)
                spkr1 = fname[:3]
                spkr2 = fname2[:3]
                if spkr1 == spkr2:
                    same_spkrs += 1
                else:
                    diff_spkrs += 1
                pairs.append((cword, spkr1, spkr2, tmp, tmp2, dtw[0], dtw[-1][1], dtw[-1][2]))
            fs.append((fname, tmp))
joblib.dump(pairs, "from_aren.joblib",
        compress=3, cache_size=512)
print "ratio same spkrs / all:", float(same_spkrs) / (same_spkrs + diff_spkrs)
