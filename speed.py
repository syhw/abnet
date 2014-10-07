import numpy as np
import joblib, time
from itertools import imap, izip

data_same = joblib.load('dtw_words_train.joblib')

t = time.time()
same_spkr = 0
for i, tup in enumerate(data_same):
    if tup[1] == tup[2]:
        same_spkr += 1
print same_spkr
print time.time()-t


t = time.time()
print np.sum(np.array(zip(*data_same)[1]) == np.array(zip(*data_same)[2]))
print time.time()-t


t = time.time()
print np.sum(list(imap(lambda tup: 1 if tup[1]==tup[2] else 0, data_same)))
print time.time()-t


t = time.time()
print np.sum(map(lambda tup: 1 if tup[1]==tup[2] else 0, data_same))
print time.time()-t
