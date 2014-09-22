import joblib, glob
from joblib import Parallel, delayed
from itertools import izip
from functools import partial
from collections import defaultdict
import numpy as np
from multiprocessing import cpu_count
from dtw import DTW


MIN_LENGTH_WORDS = 11     # in characters # TODO 6
MIN_FRAMES = 5           # in speech frames
#FBANKS_TIME_STEP = 0.01  # in seconds
FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40 # number of filterbanks to use

wav_dirname = '/fhgfs/bootphon/scratch/gsynnaeve/LUCID/wav_native/wav_to_process/'
tokens_fname = '/fhgfs/bootphon/scratch/gsynnaeve/LUCID/lucid_native.tokens.txt'


def strip_split(l):
    return l.rstrip('\n').strip().split('\t')


def parse_header(h):
    return strip_split(h.replace('#', ''))


def parse_line(l, keys=None):
    if keys != None:
        keys = keys
    else:
        keys = ('fname', 'onset', 'offset', 'word', 'talker', 'task',
                'cond', 'length')
    return dict(izip(keys, strip_split(l)))


def select_words(l, header):
    tmp = parse_line(l, header)
    if int(tmp['word_length']) >= MIN_LENGTH_WORDS:
        return tmp
    return None


def do_it_dirty(l):
    tmp = strip_split(l)
    assert(len(tmp) == 8) 
    if int(tmp[-1]) >= MIN_LENGTH_WORDS:
        return tmp
    return None


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
def open_fbank(fname):
    fbankfname = glob.glob(wav_dirname.rstrip('/') + '/' + fname.split('.')[0] + "*_fbanks.npy")[0]
    try:
        fb = np.load(fbankfname)
        print "opened:", fbankfname
    except IOError:
        print "missing fbank for", fbankfname
    return fb


@Memoize
def extract_features(fname, word, talker, s, e, before_after=2):
    sf = s * FBANKS_RATE
    ef = e * FBANKS_RATE
    fb = open_fbank(fname)
    before = max(0, sf - before_after)
    after = min(ef + before_after, fb.shape[0])
    return (word, talker, fb[before:after])


if __name__ == "__main__":
    with open(tokens_fname) as f:
        header = parse_header(f.readline())
        #s_w = partial(select_words, header=header)
        #res = filter(lambda x: x != None, map(s_w, (l for l in f)))
        res = filter(lambda x: x != None, map(do_it_dirty, (l for l in f)))

    output_name = "LUCID"

    words = defaultdict(lambda: [])
    #speakers = defaultdict(lambda: [])
    for l in res:
        words[l[3].strip().lower()].append((l[0], float(l[1]), float(l[2]),
            l[4]))
        #speakers[l[4]].append((l[0], float(l[1]), float(l[2]), l[3]))

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
                    if s_same_spkr * 1. / (s_diff_spkr + s_same_spkr) > 0.4999:
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
    print s_same_spkr
    print s_diff_spkr
    same_words = Parallel(n_jobs=cpu_count()-3)(delayed(do_dtw_pair)
            (sp[0], sp[1]) for sp in pairs)
    joblib.dump(same_words, output_name + ".joblib",
    #        compress=5, cache_size=512)
            compress=3, cache_size=512)

