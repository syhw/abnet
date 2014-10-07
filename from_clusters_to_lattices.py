"""python from_clusters_to_lattices.py clustering11_train_40 lattices
"""

"""
Header fields
VERSION=%s     V  o  Lattice specification adhered to
UTTERANCE=%s   U  o  Utterance identifier
SUBLAT=%s
acscale=%f
tscale=%f
base=%f
lmname=%s
lmscale=%f
wdpenalty=%f
S  o  Sub-lattice name
o  Scaling factor for acoustic likelihoods
o  Scaling factor for times (default 1.0, i.e.\ seconds)
o  LogBase for Likelihoods (0.0 not logs, default base e)
o  Name of Language model
o  Scaling factor for language model
o  Word insertion penalty
Lattice Size fields
NODES=%d       N  c  Number of nodes in lattice
LINKS=%d       L  c  Number of links in lattice
Node Fields
I=%d                 Node identifier.  Starts node information
time=%f        t  o  Time from start of utterance (in seconds)
WORD=%s        W wc  Word (If lattice labels nodes rather that links)
L=%s             wc  Substitute named sub-lattice for this node
var=%d         v wo  Pronunciation variant number
s=%s           s  o  Semantic Tag
Link Fields
J=%d                 Link identifier.  Starts link information
START=%d       S  c  Start node number (of the link)
END=%d         E  c  End node number (of the link)
WORD=%s        W wc  Word (If lattice labels links rather that nodes)
var=%d         v wo  Pronunciation variant number
div=%s         d wo  Segmentation (modelname, duration, likelihood) triples
acoustic=%f    a wo  Acoustic likelihood of link
language=%f    l  o  General language model likelihood of link
r=%f           r  o  Pronunciation probability
"""

import cPickle, sys, glob, os
import numpy as np

in_fldr = sys.argv[1].rstrip('/') + '/'
out_fldr = sys.argv[2].rstrip('/') + '/'
try:
    os.mkdir(out_fldr)
except:
    pass


EPS_PROB = 1.E-3  # 1.E-6 maybe?


def dists_to_probs(d):
    return d / d.sum(1)[:,None]


def filter_low_probs(d):
    tmp = d[:]
    tmp[tmp < EPS_PROB] = 0.
    return tmp


def find_order_changing_frames(d):
    s = np.argsort(d, axis=1)
    prev_l = s[0]
    frames = [0]
    for i, l in enumerate(s[1:]):
        if np.all(prev_l == l):
            pass
        else:
            frames.append(i+1)





for fname in glob.iglob(in_fldr + "*.npz"):
    npz = np.load(fname)
    dists = npz['features']
    times = npz['time']
    utt = fname.split('/')[-1].split('.')[0]
    lat_fname = utt + '.lat'
    nbnodes = 0
    nblinks = 0

    with open(out_fldr + lat_fname, 'w') as wf:
        wf.write("VERSION=1.0\n")
        wf.write("UTTERANCE=" + utt + "\n")
        wf.write("lmname=...\n")
        wf.write("lmscale=X.XX  wdpenalty=X.XX\n")
        wf.write("acscale=X.XX\n")
        wf.write("vocab=...\n")
        wf.write("hmms=...\n")
        wf.write("N=" + str(nbnodes) + " L=" + str(nblinks) + "\n")
        wf.write("I=0    t=0.00  W=!NULL            \n")
        #I=1    t=0.13  W=<s>                 v=1  
        #J=125284 S=6571 E=6572 a=-241.90   l=-5.950  d=:silE,0.06:
        
