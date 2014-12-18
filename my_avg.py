import sys
import numpy as np

# TODO ponderer par le produit de la frequence des 2 phonemes (A et B)
# DU BUCKEYE (pour phone.*)

with open(sys.argv[1]) as f:
    lines = f.readlines()[1:]
    lines = map(lambda x: (int(x.split('\t')[-1]), float(x.split('\t')[-2])), lines)
    l = np.array(lines)
    l = l.transpose()
    #s = np.sum(l[0] * l[1]) / np.sum(l[0])  # TODO check
    s = np.sum(l[1]) / l[1].shape[0]
    print "avg for", sys.argv[1], "=", s


