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

MIN_LENGTH_WORDS = 7     # in characters
MIN_FRAMES = 5           # in speech frames
#FBANKS_TIME_STEP = 0.01  # in seconds
FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40 # number of filterbanks to use
RATIO_SAME = 0.25

bdir = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/buckeye_modified_split_devtest/"


def do_dtw_pair(p1, p2):
    dtw = DTW(p1[2], p2[2], return_alignment=1)
    # word, talkerX, talkerY, x, y, cost_dtw, dtw_x_to_y_mapping, dtw_y_to_x_mapping
    return p1[0], p1[1], p2[1], p1[2], p2[2], dtw[0], dtw[-1][1], dtw[-1][2]


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
    #print fb.shape
    return fb


@Memoize
def extract_features(fname, word, talker, s, e, before_after=2):
    sf = s * FBANKS_RATE
    ef = e * FBANKS_RATE
    fb = do_fbank(fname)
    before = max(0, sf - before_after)
    after = min(ef + before_after, fb.shape[0])
    return (word, talker, fb[before:after])


if __name__ == "__main__":
    #for dset in ['test']:
    for dset in ['test', 'dev']:
        words = defaultdict(lambda: [])
        for bd, _, files in os.walk(bdir + 'wrd/' + dset + '/'):
            for fname in files:
                if fname[-4:] != '.wrd':
                    continue
                wrdfname = bd + fname
                wavfname = wrdfname.replace('wrd', 'wav')
                with open(wrdfname) as rf:
                    for line in rf:
                        s, e, w = line.rstrip('\n').split()
                        s, e = float(s), float(e)
                        w = w.strip().lower()
                        words[w].append((wavfname, s, e, wavfname.split('/')[-1][:3]))  # the first 3 characters define the speaker

        print len(words)
        words = dict(filter(lambda (w,_): len(w)>=MIN_LENGTH_WORDS, words.iteritems()))

        output_name = "BUCKEYE_" + dset
        pairs = []
        diff_spkr = 1
        same_spkr = 1
        s_same_spkr = 1
        s_diff_spkr = 1
        s_np_same_spkr = 1
        s_np_diff_spkr = 1

        for word, tokens in words.iteritems():
            for i, t1 in enumerate(tokens):
                for j, t2 in enumerate(tokens):
                    if i >= j:
                        continue
                    if t1[-1] != t2[-1]:
                        diff_spkr += 1
                        if s_same_spkr * 1. / (s_diff_spkr + s_same_spkr) > RATIO_SAME:
                            f1 = extract_features(t1[0], word, t1[-1],
                                    t1[1], t1[2])
                            f2 = extract_features(t2[0], word, t2[-1],
                                    t2[1], t2[2])
                            if (f1[-1].shape[0] > MIN_FRAMES and
                                    f2[-1].shape[0] > MIN_FRAMES):
                                s_diff_spkr += 1
                                pairs.append((f1, f2))
                    else:
                        same_spkr += 1
                        f1 = extract_features(t1[0], word, t1[-1],
                                t1[1], t1[2])
                        f2 = extract_features(t2[0], word, t2[-1],
                                t2[1], t2[2])
                        if (f1[-1].shape[0] > MIN_FRAMES and
                                f2[-1].shape[0] > MIN_FRAMES):
                            s_same_spkr += 1
                            pairs.append((f1, f2))

        print "ratio same speakers / all (on word pairs):",
        print same_spkr * 1. / (same_spkr + diff_spkr)
        print "ratio same speakers / all (on SAMPLED word pairs):",
        print s_same_spkr * 1. / (s_same_spkr + s_diff_spkr)
        print "same spkrs:", s_same_spkr
        print "diff skprs:", s_diff_spkr
        same_words = Parallel(n_jobs=cpu_count()-3)(delayed(do_dtw_pair)
                (sp[0], sp[1]) for sp in pairs)
        joblib.dump(same_words, output_name + ".joblib",
        #        compress=5, cache_size=512)
                compress=3, cache_size=512)

