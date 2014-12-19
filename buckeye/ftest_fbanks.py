import cPickle, sys, glob, os
from collections import defaultdict
import numpy as np

NFRAMES_PER_SEC = 100
NFRAMES = 11
NFBANKS = 40
bdir = sys.argv[1].rstrip('/') + '/'
phndir = "buckeye_modified_split_devtest/phn/*/"


def parse(fn):
    try:
        with open(fn) as rf:
            li = map(lambda x: x.rstrip('\n').split(), rf.readlines())
            li = map(lambda x: (float(x[0]), float(x[1]), x[2]), li)
            #print fn, "found"
            return li
    except IOError:
        print fn, "not found => skipping it"
        return []


dir = bdir + 'npz11_test/'

spkrs = defaultdict(lambda: [np.zeros(NFRAMES*NFBANKS), np.zeros(NFRAMES*NFBANKS), 0])  # {'sid':[NFRAMES*NFBANKS (mean, mean**2, length)]}
phns = defaultdict(lambda: [np.zeros(NFRAMES*NFBANKS), np.zeros(NFRAMES*NFBANKS), 0])
all = [np.zeros(NFRAMES*NFBANKS), np.zeros(NFRAMES*NFBANKS), 0] # [NFRAMES*NFBANKS (mean, mean**2, length)]

for fname in glob.iglob(dir + "*.npz"):
    npz = np.load(fname)
    npz_fn = fname.split('/')[-1]
    phn_fn = [x for x in glob.iglob(phndir + npz_fn.split('.')[0] + '.phn')][0]
    phones = parse(phn_fn)
    spkr = npz_fn[:3]
    t = npz['features']
    t2 = t**2
    s = np.sum(t, axis=0) # (length, NFRAMES*NFBANKS)
    s2 = np.sum(t2, axis=0)
    l = t.shape[0]
    spkrs[spkr][0] += s
    spkrs[spkr][1] += s2
    spkrs[spkr][2] += l
    for start, end, phn in phones:
        tt = t[start*NFRAMES_PER_SEC:end*NFRAMES_PER_SEC]
        tt2 = t2[start*NFRAMES_PER_SEC:end*NFRAMES_PER_SEC]
        phns[phn][0] += np.sum(tt, axis=0)
        phns[phn][1] += np.sum(tt2, axis=0)
        phns[phn][2] += tt.shape[0]
    all[0] += s
    all[1] += s2
    all[2] += l
N = all[2]
means = all[0] / N
# SPEAKERS
print spkrs.keys()
K = len(spkrs)
assert(K > 1)
means_spkrs = {}
vars_spkrs = {}
for spkr, (s, s2, l) in spkrs.iteritems():
    m = s/l
    means_spkrs[spkr] = m
    vars_spkrs[spkr] = s2/l - m**2
between_vars = np.zeros(NFRAMES*NFBANKS)
for spkr, m in means_spkrs.iteritems():
    between_vars += spkrs[spkr][2] * (m - means)**2 / (K - 1)
within_vars = np.zeros(NFRAMES*NFBANKS)
for v in vars_spkrs.itervalues():
    within_vars += v / (N - K)
F = between_vars / within_vars
with open(bdir + 'F_spkrs_fbanks11.npy', 'wb') as f:
    F.dump(f)
# PHONES
print phns.keys()
K = len(phns)
assert(K > 1)
means_phns = {}
vars_phns = {}
for phn, (s, s2, l) in phns.iteritems():
    m = s/l
    means_phns[phn] = m
    vars_phns[phn] = s2/l - m**2
between_vars = np.zeros(NFRAMES*NFBANKS)
for phn, m in means_phns.iteritems():
    between_vars += phns[phn][2] * (m - means)**2 / (K - 1)
within_vars = np.zeros(NFRAMES*NFBANKS)
for v in vars_phns.itervalues():
    within_vars += v / (N - K)
F = between_vars / within_vars
with open(bdir + 'F_phns_fbanks11.npy', 'wb') as f:
    F.dump(f)
