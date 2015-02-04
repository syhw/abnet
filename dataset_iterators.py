MIN_FRAMES_PER_SENTENCE = 26
BATCH_SIZE = 100
import numpy, theano
from collections import defaultdict
import random, joblib, math, sys
from multiprocessing import cpu_count
from itertools import izip
from random import shuffle


def pad(x, nf, ma=0):
    """ pad x for nf frames with margin ma. """
    ba = (nf - 1) / 2  # before/after
    if ma:
        ret = numpy.zeros((x.shape[0] - 2 * ma, x.shape[1] * nf),
                dtype=theano.config.floatX)
        if ba <= ma:
            for j in xrange(ret.shape[0]):
                ret[j] = x[j:j + 2*ma + 1].flatten()
        else:
            for j in xrange(ret.shape[0]):
                ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                        (max(0, (ba - j) * x.shape[1]),
                            max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                        'constant', constant_values=(0, 0))
        return ret
    else:
        ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                dtype=theano.config.floatX)
        for j in xrange(x.shape[0]):
            ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                    (max(0, (ba - j) * x.shape[1]),
                        max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0, 0))
        return ret


from dtw import DTW
def do_dtw(x1, x2):
    dtw = DTW(x1, x2, return_alignment=1)
    return dtw[0], dtw[-1][1], dtw[-1][2]


class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)

    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])


class DatasetSentencesIterator(object):
    """ An iterator on sentences of the dataset. """

    def __init__(self, x, y, phn_to_st, nframes=1, batch_size=None):
        # batch_size is ignored
        self._x = x
        self._y = numpy.asarray(y)
        self._start_end = [[0]]
        self._nframes = nframes
        self._memoized_x = defaultdict(lambda: {})
        i = 0
        for i, s in enumerate(self._y == phn_to_st['!ENTER[2]']):
            if s and i - self._start_end[-1][0] > MIN_FRAMES_PER_SENTENCE:
                self._start_end[-1].append(i)
                self._start_end.append([i])
        self._start_end[-1].append(i+1)

    def _stackpad(self, start, end):
        """ Method because of the memoization. """
        if start in self._memoized_x and end in self._memoized_x[start]:
            return self._memoized_x[start][end]
        x = self._x[start:end]
        nf = self._nframes
        ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                dtype=theano.config.floatX)
        ba = (nf - 1) / 2  # before/after
        for i in xrange(x.shape[0]):
            ret[i] = numpy.pad(x[max(0, i - ba):i + ba +1].flatten(),
                    (max(0, (ba - i) * x.shape[1]),
                        max(0, ((i + ba + 1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0, 0))
        self._memoized_x[start][end] = ret
        return ret

    def __iter__(self):
        for start, end in self._start_end:
            if self._nframes > 1:
                yield self._stackpad(start, end), self._y[start:end]
            else:
                yield self._x[start:end], self._y[start:end]


class DatasetSentencesIteratorPhnSpkr(DatasetSentencesIterator):
    """ An iterator on sentences of the dataset, specialized for datasets
    with both phones and speakers in y labels. """
    def __init__(self, x, y, phn_to_st, nframes=1, batch_size=None):
        super(DatasetSentencesIteratorPhnSpkr, self).__init__(x, y[0], phn_to_st, nframes, batch_size)
        self._y_spkr = numpy.asarray(y[1])

    def __iter__(self):
        for start, end in self._start_end:
            if self._nframes > 1:
                yield self._stackpad(start, end), self._y[start:end], self._y_spkr[start:end]
            else:
                yield self._x[start:end], self._y[start:end], self._y_spkr[start:end]


class DatasetBatchIteratorPhn(object):
    def __init__(self, x, y, phn_to_st, nframes=1, batch_size=None):
        pass
        # TODO (see timit_tools/DBN one)


class DatasetABIterator(object):
    """ An iterator over pairs x1/x2 that can be diff/same (y=0/1). """
    def __init__(self, x1, x2, y, batch_size=BATCH_SIZE):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        assert(self.x1.shape[0] == self.x2.shape[0])
        assert(self.x1.shape[0] == self.y.shape[0])
        self.batch_size = batch_size

    def __iter__(self):
        n_samples = self.x1.shape[0]
        for i in xrange((n_samples + self.batch_size - 1)
                        / self.batch_size):
            yield ((self.x1[i*self.batch_size:(i+1)*self.batch_size],
                self.x2[i*self.batch_size:(i+1)*self.batch_size]),
                self.y[i*self.batch_size:(i+1)*self.batch_size])


class DatasetABSamplingIteratorFromLabels(object):
    """ An iterator that samples over pairs x1/x2
    that can be diff/same (y=0/1). """
    def __init__(self, x, y, n_samples=10, batch_size=BATCH_SIZE):
        assert((batch_size % 2) == 0)
        assert(batch_size >= 2*n_samples)  # 2* for same+diff
        self.y_set = numpy.unique(y)
        self.x = x
        self.y = y
        self.y_indices = dict(izip([y_ind for y_ind in self.y_set],
            [numpy.where(self.y==y_ind)[0] for y_ind in self.y_set]))
        self.not_y_indices = dict(izip([y_ind for y_ind in self.y_set],
            [numpy.where(self.y!=y_ind)[0] for y_ind in self.y_set]))
        self.n_samples = n_samples
        self.batch_size = batch_size
        print >> sys.stderr, "finished initializing the iterator"

    def __iter__(self):
        n_items_per_batch = ((self.batch_size/2)/self.n_samples)
        for i in xrange((self.x.shape[0]+n_items_per_batch-1)
                / n_items_per_batch):
            todo_x = self.x[i*n_items_per_batch:(i+1)*n_items_per_batch]
            todo_y = self.y[i*n_items_per_batch:(i+1)*n_items_per_batch]
            tmp_x1 = []
            tmp_x2 = []
            tmp_y = numpy.zeros(self.batch_size, dtype='int32')
            tmp_y[::2] = 1
            for x, y in izip(todo_x, todo_y):  # slow
                same_x = self.x[numpy.random.choice(self.y_indices[y],
                        size=self.n_samples, replace=False)]  # can do A=A
                while same_x == x:
                    same_x = self.x[numpy.random.choice(self.y_indices[y],
                            size=self.n_samples, replace=False)]  # can do A=A
                diff_x = self.x[numpy.random.choice(self.not_y_indices[y],
                        size=self.n_samples, replace=False)]
                for x2_same, x2_diff in izip(same_x, diff_x): # slow
                    tmp_x1.append(x)
                    tmp_x2.append(x2_same)
                    tmp_x1.append(x)
                    tmp_x2.append(x2_diff)
            yield ((numpy.array(tmp_x1), numpy.array(tmp_x2)), tmp_y)


class DatasetAB2OSamplingIteratorFromLabels(object):
    """ An iterator that samples over pairs x1/x2
    that can be diff/same over one (y1=0/1). """
    def __init__(self, x, y1, y2, n_samples=10, batch_size=BATCH_SIZE):
        assert((batch_size % 4) == 0)
        assert(batch_size >= 4*n_samples)  # 4* for same and diff combinations
        self.x = x
        self.y1_set = numpy.unique(y1)
        self.y1 = y1
        self.y1_indices = dict(izip([y_ind for y_ind in self.y1_set],
            [numpy.where(self.y1==y_ind)[0] for y_ind in self.y1_set]))
        self.not_y1_indices = dict(izip([y_ind for y_ind in self.y1_set],
            [numpy.where(self.y1!=y_ind)[0] for y_ind in self.y1_set]))
        self.y2_set = numpy.unique(y2)
        self.y2 = y2
        self.y2_indices = dict(izip([y_ind for y_ind in self.y2_set],
            [numpy.where(self.y2==y_ind)[0] for y_ind in self.y2_set]))
        self.not_y2_indices = dict(izip([y_ind for y_ind in self.y2_set],
            [numpy.where(self.y2!=y_ind)[0] for y_ind in self.y2_set]))
        self.same_same_i = {}
        self.same_diff_i = {}
        self.diff_same_i = {}
        self.diff_diff_i = {}
        from itertools import product
        for y1, y2 in product(self.y1_set, self.y2_set):
            self.same_same_i[(y1, y2)] = numpy.intersect1d(self.y1_indices[y1],
                    self.y2_indices[y2], assume_unique=True)
            self.same_diff_i[(y1, y2)] = numpy.setdiff1d(self.y1_indices[y1],
                    self.same_same_i[(y1, y2)], assume_unique=True)
            self.diff_same_i[(y1, y2)] = numpy.setdiff1d(self.y2_indices[y2],
                    self.same_same_i[(y1, y2)], assume_unique=True)
            #self.diff_same_i[(y1, y2)] = numpy.intersect1d(self.not_y1_indices[y1],
            #        self.y2_indices[y2], assume_unique=True)
            self.diff_diff_i[(y1, y2)] = numpy.intersect1d(self.not_y1_indices[y1],
                    self.not_y2_indices[y2], assume_unique=True)
            #print y1, y2
            #print self.same_same_i[(y1, y2)].shape
            #print self.same_diff_i[(y1, y2)].shape
            #print self.diff_same_i[(y1, y2)].shape
            #print self.diff_diff_i[(y1, y2)].shape
        self.n_samples = n_samples
        self.batch_size = batch_size
        assert(self.x.shape[0] == len(self.y1) == len(self.y2))
        print >> sys.stderr, "finished initializing the iterator"

    def __iter__(self):
        n_items_per_batch = ((self.batch_size/4)/self.n_samples)
        for i in xrange((self.x.shape[0]+n_items_per_batch-1)
                / n_items_per_batch):
            todo_x = self.x[i*n_items_per_batch:(i+1)*n_items_per_batch]
            todo_y1 = self.y1[i*n_items_per_batch:(i+1)*n_items_per_batch]
            todo_y2 = self.y2[i*n_items_per_batch:(i+1)*n_items_per_batch]
            tmp_x1 = []
            tmp_x2 = []
            tmp_y1 = numpy.zeros(todo_x.shape[0]*4*self.n_samples,
                    dtype='int32')
            tmp_y1[::4] = 1
            tmp_y1[1::4] = 1
            tmp_y2 = numpy.zeros(todo_x.shape[0]*4*self.n_samples,
                    dtype='int32')
            tmp_y2[::2] = 1
            # TODO that tmp_y1 and tmp_y2 are arguments to the iterator
            for x, y1, y2 in izip(todo_x, todo_y1, todo_y2):  # slow
                replace = False
                if self.same_same_i[(y1, y2)].shape[0] < self.n_samples:
                    replace = True
                x2_same_same = self.x[numpy.random.choice(
                    self.same_same_i[(y1, y2)],
                    size=self.n_samples, replace=replace)]
                x2_same_diff = self.x[numpy.random.choice(
                    self.same_diff_i[(y1, y2)],
                    size=self.n_samples, replace=False)]
                x2_diff_same = self.x[numpy.random.choice(
                    self.diff_same_i[(y1, y2)],
                    size=self.n_samples, replace=False)]
                x2_diff_diff = self.x[numpy.random.choice(
                    self.diff_diff_i[(y1, y2)],
                    size=self.n_samples, replace=False)]
                for x2, x3, x4, x5 in izip(x2_same_same, x2_same_diff, x2_diff_same, x2_diff_diff): # slow
                    tmp_x1.append(x)
                    tmp_x2.append(x2)
                    tmp_x1.append(x)
                    tmp_x2.append(x3)
                    tmp_x1.append(x)
                    tmp_x2.append(x4)
                    tmp_x1.append(x)
                    tmp_x2.append(x5)
            yield ((numpy.array(tmp_x1), numpy.array(tmp_x2)), (tmp_y1, tmp_y2))


class DatasetDTWIterator(object):
    """ An iterator over dynamic time warped words of the dataset. """

    def __init__(self, x1, x2, y, nframes=1, batch_size=1, marginf=0):
        # x1 and x2 are tuples or arrays that are [nframes, nfeatures]
        self._x1 = x1
        self._x2 = x2
        self._y = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        # self._y says if frames in x1 and x2 are same (1) or different (0)
        for ii, yy in enumerate(y):
            self._y[ii][:] = yy
        self._nframes = nframes
        self._nwords = batch_size
        self._margin = marginf
        # marginf says if we pad taking a number of frames as margin
        self._x1_mem = []
        self._x2_mem = []
        self._y_mem = []

    def _memoize(self, i):
        """ Computes the corresponding x1/x2/y for the given i depending on the
        self._nframes (stacking x1/x2 features for self._nframes), and
        self._nwords (number of words per mini-batch).
        """
        ind = i/self._nwords
        if ind < len(self._x1_mem) and ind < len(self._x2_mem):
            return [[self._x1_mem[ind], self._x2_mem[ind]], self._y_mem[ind]]

        nf = self._nframes
        def local_pad(x):  # TODO replace with pad global function
            if nf <= 1:
                return x
            if self._margin:
                ma = self._margin
                ba = (nf - 1) / 2  # before/after
                if x.shape[0] - 2*ma <= 0:
                    print >> sys.stderr, "shape[0]:", x.shape[0]
                    print >> sys.stderr, "ma:", ma
                if x.shape[1] * nf <= 0:
                    print >> sys.stderr, "shape[1]:", x.shape[1]
                    print >> sys.stderr, "nf:", nf
                ret = numpy.zeros((max(0, x.shape[0] - 2 * ma),
                    x.shape[1] * nf),
                    dtype=theano.config.floatX)
                if ba <= ma:
                    for j in xrange(ret.shape[0]):
                        ret[j] = x[j + ma - ba:j + ma + ba + 1].flatten()
                else:
                    for j in xrange(ret.shape[0]):
                        ret[j] = numpy.pad(x[max(0, j - ba + ma):j + ba + ma + 1].flatten(),
                                (max(0, (ba - j - ma) * x.shape[1]),
                                    max(0, ((j + ba + ma + 1) - x.shape[0]) * x.shape[1])),
                                'constant', constant_values=(0, 0))
                return ret
            else:
                ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                        dtype=theano.config.floatX)
                ba = (nf - 1) / 2  # before/after
                for j in xrange(x.shape[0]):
                    ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                            (max(0, (ba - j) * x.shape[1]),
                                max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                            'constant', constant_values=(0, 0))
                return ret
        
        def cut_y(y):
            ma = self._margin
            if nf <= 1 or ma == 0:
                return numpy.asarray(y, dtype='int8')
            ret = numpy.zeros(max(0, (y.shape[0] - 2 * ma)), dtype='int8')
            for j in xrange(ret.shape[0]):
                ret[j] = y[j+ma]
            return ret

        x1_padded = [local_pad(self._x1[i+k]) for k 
                in xrange(self._nwords) if i+k < len(self._x1)]
        x2_padded = [local_pad(self._x2[i+k]) for k
                in xrange(self._nwords) if i+k < len(self._x2)]
        assert x1_padded[0].shape[0] == x2_padded[0].shape[0]
        y_padded = [cut_y(self._y[i+k]) for k in
            xrange(self._nwords) if i+k < len(self._y)]
        assert x1_padded[0].shape[0] == len(y_padded[0])
        self._x1_mem.append(numpy.concatenate(x1_padded))
        self._x2_mem.append(numpy.concatenate(x2_padded))
        self._y_mem.append(numpy.concatenate(y_padded))
        return [[self._x1_mem[ind], self._x2_mem[ind]], self._y_mem[ind]]

    def __iter__(self):
        for i in xrange(0, len(self._y), self._nwords):
            yield self._memoize(i)


class DatasetDTWWrdSpkrIterator(DatasetDTWIterator):
    """ TODO """

    def __init__(self, data_same, normalize=True, min_max_scale=False,
            scale_f1=None, scale_f2=None,
            nframes=1, batch_size=1, marginf=0, only_same=False,
            cache_to_disk=False):
        self.print_mean_DTW_costs(data_same)
        self.ratio_same = 0.5  # init
        self.ratio_same = self.compute_ratio_speakers(data_same)
        self._nframes = nframes
        print "nframes:", self._nframes

        (self._x1, self._x2, self._y_word, self._y_spkr,
                self._scale_f1, self._scale_f2) = self.prep_data(data_same,
                        normalize, min_max_scale, scale_f1, scale_f2)

        self._y1 = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        self._y2 = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        # self._y1 says if frames in x1 and x2 belong to the same (1) word or not (0)
        # self._y2 says if frames in x1 and x2 were said by the same (1) speaker or not(0)
        for ii, yy in enumerate(self._y_word):
            self._y1[ii][:] = yy
        for ii, yy in enumerate(self._y_spkr):
            self._y2[ii][:] = yy
        self._nwords = batch_size
        self._margin = marginf
        # marginf says if we pad taking a number of frames as margin
        self._x1_mem = []
        self._x2_mem = []
        self._y1_mem = []
        self._y2_mem = []
        self.cache_to_disk = cache_to_disk
        if self.cache_to_disk:
            from joblib import Memory
            self.mem = Memory(cachedir='joblib_cache', verbose=0)

    def _memoize(self, i):
        """ Computes the corresponding x1/x2/y1/y2 for the given i 
        depending on the self._nframes (stacking x1/x2 features for
        self._nframes), and self._nwords (number of words per mini-batch).
        """
        ind = i/self._nwords
        if ind < len(self._x1_mem) and ind < len(self._x2_mem):
            return [[self._x1_mem[ind], self._x2_mem[ind]],
                    [self._y1_mem[ind], self._y2_mem[ind]]]

        nf = self._nframes
        def local_pad(x):  # TODO replace with pad global function
            if nf <= 1:
                return x
            if self._margin:
                ma = self._margin
                ba = (nf - 1) / 2  # before/after
                if x.shape[0] - 2*ma <= 0:
                    print >> sys.stderr, "shape[0]:", x.shape[0]
                    print >> sys.stderr, "ma:", ma
                if x.shape[1] * nf <= 0:
                    print >> sys.stderr, "shape[1]:", x.shape[1]
                    print >> sys.stderr, "nf:", nf
                ret = numpy.zeros((max(0, x.shape[0] - 2 * ma),
                    x.shape[1] * nf),
                    dtype=theano.config.floatX)
                if ba <= ma:
                    for j in xrange(ret.shape[0]):
                        ret[j] = x[j + ma - ba:j + ma + ba + 1].flatten()
                else:
                    for j in xrange(ret.shape[0]):
                        ret[j] = numpy.pad(x[max(0, j - ba + ma):j + ba + ma + 1].flatten(),
                                (max(0, (ba - j - ma) * x.shape[1]),
                                    max(0, ((j + ba + ma + 1) - x.shape[0]) * x.shape[1])),
                                'constant', constant_values=(0, 0))
                return ret
            else:
                ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                        dtype=theano.config.floatX)
                ba = (nf - 1) / 2  # before/after
                for j in xrange(x.shape[0]):
                    ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                            (max(0, (ba - j) * x.shape[1]),
                                max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                            'constant', constant_values=(0, 0))
                return ret
        
        def cut_y(y):
            ma = self._margin
            if nf <= 1 or ma == 0:
                return numpy.asarray(y, dtype='int8')
            ret = numpy.zeros(max(0, (y.shape[0] - 2 * ma)), dtype='int8')
            for j in xrange(ret.shape[0]):
                ret[j] = y[j+ma]
            return ret

        x1_padded = [local_pad(self._x1[i+k]) for k 
                in xrange(self._nwords) if i+k < len(self._x1)]
        x2_padded = [local_pad(self._x2[i+k]) for k
                in xrange(self._nwords) if i+k < len(self._x2)]
        assert x1_padded[0].shape[0] == x2_padded[0].shape[0]
        y1_padded = [cut_y(self._y1[i+k]) for k in
            xrange(self._nwords) if i+k < len(self._y1)]
        y2_padded = [cut_y(self._y2[i+k]) for k in
            xrange(self._nwords) if i+k < len(self._y2)]
        assert x1_padded[0].shape[0] == len(y1_padded[0])
        assert x1_padded[0].shape[0] == len(y2_padded[0])
        xx1 = numpy.concatenate(x1_padded)
        xx2 = numpy.concatenate(x2_padded)
        yy1 = numpy.concatenate(y1_padded)
        yy2 = numpy.concatenate(y2_padded)
        if not self.cache_to_disk:
            self._x1_mem.append(xx1)
            self._x2_mem.append(xx2)
            self._y1_mem.append(yy1)
            self._y2_mem.append(yy2)
        #return [[self._x1_mem[ind], self._x2_mem[ind]],
        #        [self._y1_mem[ind], self._y2_mem[ind]]]
        return [[xx1, xx2],
                [yy1, yy2]]

    def __iter__(self):
        memo = self._memoize
        if self.cache_to_disk:
            memo = self.mem.cache(self._memoize)
        for i in xrange(0, len(self._y_word), self._nwords):
            yield memo(i)

    def print_mean_DTW_costs(self, data_same):
        dtw_costs = numpy.array(zip(*data_same)[5])
        print "mean DTW cost", numpy.mean(dtw_costs), "std dev", numpy.std(dtw_costs)
        words_frames = numpy.array([fb.shape[0] for fb in zip(*data_same)[3]])
        print "mean word length in frames", numpy.mean(words_frames), "std dev", numpy.std(words_frames)
        print "mean DTW cost per frame", numpy.mean(dtw_costs/words_frames), "std dev", numpy.std(dtw_costs/words_frames)

    def compute_ratio_speakers(self, data_same):
        same_spkr = 0
        for i, tup in enumerate(data_same):
            if tup[1] == tup[2]:
                same_spkr += 1
        ratio = same_spkr * 1. / len(data_same)
        print "ratio same spkr / all for same:", ratio
        return ratio

    def prep_data(self, data_same, normalize=True, min_max_scale=False,
            scale_f1=None, scale_f2=None,
            balanced_spkr=True):
        #data_same = [(word_label, talker1, talker2, fbanks1, fbanks2, DTW_cost, DTW_1to2, DTW_2to1)]
        data_diff = []
        ldata_same = len(data_same)-1
        y_spkrs_same = []
        y_spkrs_diff = []
        SAMPLE_DIFF_WORDS = True  # TODO that's for debug purposes, needs to run on CPU
        if SAMPLE_DIFF_WORDS:
            print "Now sampling the pairs of different words..."
        else:
            print "Now writing y_spkrs labels for same words..."
        for i, ds in enumerate(data_same):
            if ds[1] == ds[2]:
                #print "same spkr same word"
                y_spkrs_same.append(1)
            else:
                y_spkrs_same.append(0)
            if SAMPLE_DIFF_WORDS:
                word_1 = random.randint(0, ldata_same)
                word_1_type = data_same[word_1][0]
                word_2 = random.randint(0, ldata_same)
                while data_same[word_2][0] == word_1_type:
                    word_2 = random.randint(0, ldata_same)
                if balanced_spkr:
                    ratio = numpy.mean(y_spkrs_diff)
                    spkr1_a = data_same[word_1][1]
                    spkr1_b = data_same[word_1][2]
                    spkr2_a = data_same[word_2][1]
                    spkr2_b = data_same[word_2][2]
                    ratio_balancing = False
                    while ratio < (self.ratio_same - 0.001) and (
                            spkr1_a != spkr2_a and spkr1_a != spkr2_b and
                            spkr1_b != spkr2_a and spkr1_b != spkr2_b):
                        word_2 = random.randint(0, ldata_same)
                        ratio_balancing = True
                        spkr1_a = data_same[word_1][1]
                        spkr1_b = data_same[word_1][2]
                        spkr2_a = data_same[word_2][1]
                        spkr2_b = data_same[word_2][2]
                    if ratio_balancing:
                        if spkr1_a == spkr2_a:
                            wt1 = 0
                            wt2 = 0
                        elif spkr1_a == spkr2_b:
                            wt1 = 0
                            wt2 = 1
                        elif spkr1_b == spkr2_a:
                            wt1 = 1
                            wt2 = 0
                        elif spkr1_b == spkr2_b:
                            wt1 = 1
                            wt2 = 1
                    else:
                        wt1 = random.randint(0, 1)  # random filterbank
                        wt2 = random.randint(0, 1)  # random filterbank
                    
                else:
                    wt1 = random.randint(0, 1)  # random filterbank
                    wt2 = random.randint(0, 1)  # random filterbank
                spkr1 = data_same[word_1][1+wt1]
                spkr2 = data_same[word_2][1+wt2]
                p1 = data_same[word_1][3+wt1]
                p2 = data_same[word_2][3+wt2]
                r1 = p1[:min(len(p1), len(p2))]
                r2 = p2[:min(len(p1), len(p2))]
                data_diff.append((r1, r2))
                #if spkr1[0] == spkr2[0]:  # TODO TODO speaker sex/genre
                if spkr1 == spkr2:
                    #print "same spkr diff word"
                    y_spkrs_diff.append(1)
                else:
                    y_spkrs_diff.append(0)
        if SAMPLE_DIFF_WORDS:
            ratio = numpy.mean(y_spkrs_diff)
            print "ratio same spkr / all for diff:", ratio

        x_arr_same = numpy.r_[numpy.concatenate([e[3] for e in data_same]),
            numpy.concatenate([e[4] for e in data_same])]
        print x_arr_same.shape
        if SAMPLE_DIFF_WORDS:
            x_arr_diff = numpy.r_[numpy.concatenate([e[0] for e in data_diff]),
                    numpy.concatenate([e[1] for e in data_diff])]
            print x_arr_diff.shape
        else:
            x_arr_diff = None

        if normalize:
            # Normalizing
            if scale_f1 == None or scale_f2 == None:
                if x_arr_diff != None:
                    x_arr_all = numpy.concatenate([x_arr_same, x_arr_diff])
                else:
                    x_arr_all = x_arr_same
                scale_f1 = numpy.mean(x_arr_all, 0)
                scale_f2 = numpy.std(x_arr_all, 0)
                numpy.savez("mean_std_spkr_word.npz", mean=scale_f1, std=scale_f2)

            x_same = [((e[3][e[-2]] - scale_f1) / scale_f2,
                (e[4][e[-1]] - scale_f1) / scale_f2)
                    for e in data_same]
        elif min_max_scale:
            # Min-max scaling
            if scale_f1 == None or scale_f2 == None:
                if x_arr_diff != None:
                    x_arr_all = numpy.concatenate([x_arr_same, x_arr_diff])
                else:
                    x_arr_all = x_arr_same
                scale_f1 = x_arr_all.min(axis=0)
                scale_f2 = x_arr_all.max(axis=0)
                numpy.savez("min_max_spkr_word.npz", min=scale_f1, max=scale_f2)

            x_same = [((e[3][e[-2]] - scale_f1) / 10*(scale_f2 - scale_f1),
                (e[4][e[-1]] - scale_f1) / 10*(scale_f2 - scale_f1))
                    for e in data_same]
        else:
            x_same = [(e[3][e[-2]], e[4][e[-1]]) for e in data_same]
        zipped = zip(x_same, y_spkrs_same)
        shuffle(zipped)
        x_same, y_sprks_same = zip(*zipped)
        y_same = [[1 for _ in xrange(len(e[0]))] for e in x_same]
        y_same_spkr = [[y_spkrs_same[i] for _ in xrange(len(e[0]))] for i, e
                in enumerate(x_same)]
        assert(len(y_same) == len(y_same_spkr))

        if SAMPLE_DIFF_WORDS:
            if normalize:
                x_diff = [((e[0] - scale_f1) / scale_f2,
                    (e[1] - scale_f1) / scale_f2)
                        for e in data_diff]
            elif min_max_scale:
                x_diff = [((e[0] - scale_f1) / 10*(scale_f2 - scale_f1),
                    (e[1] - scale_f1) / 10*(scale_f2 - scale_f1))
                        for e in data_diff]
            else:
                x_diff = [(e[0], e[1]) for e in data_diff]
            y_diff = [[0 for _ in xrange(len(e[0]))] for e in x_diff]
            y_diff_spkr = [[y_spkrs_diff[i] for _ in xrange(len(e[0]))] for i, e
                    in enumerate(x_diff)]
            assert(len(y_diff) == len(y_diff_spkr))

            y_word = [j for i in zip(y_same, y_diff) for j in i]
            y_spkr = [j for i in zip(y_same_spkr, y_diff_spkr) for j in i]
            x = [j for i in zip(x_same, x_diff) for j in i]
            x1, x2 = zip(*x)
        else:
            x1, x2 = zip(*x_same)
            y_word = y_same
            y_spkr = y_same_spkr
        #print x1[0]
        #print x2[0]
        #print y_word[0]
        #print y_spkr[0]

        assert x1[0].shape[0] == x2[0].shape[0]
        assert x1[0].shape[1] == x2[0].shape[1]
        assert len(x1) == len(x2)
        assert len(x1) == len(y_word)
        assert len(x1) == len(y_spkr)
        assert len(x1[0]) == len(x2[0])
        assert len(x1[0]) == len(y_spkr[0])
        assert len(x1[0]) == len(y_word[0])
        self._scale_f1 = scale_f1
        self._scale_f2 = scale_f2

        return x1, x2, y_word, y_spkr, scale_f1, scale_f2



class DatasetDTReWIterator(DatasetDTWIterator):
    """ TODO """

    def __init__(self, data_same, mean, std, nframes=1, batch_size=1, marginf=0, only_same=False):
        dtw_costs = zip(*data_same)[5]
        self._orig_x1s = zip(*data_same)[3]
        self._orig_x2s = zip(*data_same)[4]
        self._words_frames = numpy.asarray([fb.shape[0] for fb in self._orig_x1s])
        self.print_mean_DTW_costs(dtw_costs)

        self._mean = mean
        self._std = std
        self._nframes = nframes
        self._nwords = batch_size
        self._margin = marginf
        self._only_same = only_same
        # marginf says if we pad taking a number of frames as margin

        same_spkr = 0
        for i, tup in enumerate(data_same):
            if tup[1] == tup[2]:
                same_spkr += 1
        ratio = same_spkr * 1. / len(data_same)
        print "ratio same spkr / all for same:", ratio
        data_diff = []
        ldata_same = len(data_same)-1
        same_spkr_diff = 0
        for i in xrange(len(data_same)):
            word_1 = random.randint(0, ldata_same)
            word_1_type = data_same[word_1][0]
            word_2 = random.randint(0, ldata_same)
            while data_same[word_2][0] == word_1_type:
                word_2 = random.randint(0, ldata_same)

            wt1 = random.randint(0, 1)
            wt2 = random.randint(0, 1)
            if data_same[word_1][1+wt1] == data_same[word_2][1+wt2]:
                same_spkr_diff += 1
            p1 = data_same[word_1][3+wt1]
            p2 = data_same[word_2][3+wt2]
            r1 = p1[:min(len(p1), len(p2))]
            r2 = p2[:min(len(p1), len(p2))]
            data_diff.append((r1, r2))
        ratio = same_spkr_diff * 1. / len(data_diff)
        print "ratio same spkr / all for diff:", ratio

        self._data_same = zip(zip(*data_same)[3], zip(*data_same)[4],
                zip(*data_same)[-2], zip(*data_same)[-1])
        self._data_diff = data_diff

        self.remix()

        if self._nframes > 1:
            # pad the orig_xes1/2 once and for all
            self._orig_x1s = joblib.Parallel(n_jobs=cpu_count()-3)(
                    joblib.delayed(pad)(x, self._nframes, self._margin)
                    for x in self._orig_x1s)
            self._orig_x2s = joblib.Parallel(n_jobs=cpu_count()-3)(
                    joblib.delayed(pad)(x, self._nframes, self._margin)
                    for x in self._orig_x2s)


    def remix(self):
        x_same = [((e[0][e[-2]] - self._mean) / self._std, (e[1][e[-1]] - self._mean) / self._std)
                for e in self._data_same]
        y_same = [[1 for _ in xrange(len(e[0]))] for i, e in enumerate(x_same)]
        if not self._only_same:
            x_diff = [((e[0] - self._mean) / self._std, (e[1] - self._mean) / self._std)
                    for e in self._data_diff]
            random.shuffle(x_diff)
            y_diff = [[0 for _ in xrange(len(e[0]))] for i, e in enumerate(x_diff)]
            y = [j for i in zip(y_same, y_diff) for j in i]
            x = [j for i in zip(x_same, x_diff) for j in i]
        else:
            x = x_same
            y = y_same
        x1, x2 = zip(*x)
        # x1 and x2 are tuples or arrays that are [nframes, nfeatures]
        self._x1 = x1
        self._x2 = x2
        self._y = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        # self._y says if frames in x1 and x2 are same (1) or different (0)
        for ii, yy in enumerate(y):
            self._y[ii][:] = yy
        self._x1_mem = []
        self._x2_mem = []
        self._y_mem = []

    def recompute_DTW(self, transform_f):
        from itertools import izip
        xes1 = map(transform_f, self._orig_x1s)
        xes2 = map(transform_f, self._orig_x2s)
        res = joblib.Parallel(n_jobs=cpu_count()-3)(joblib.delayed(do_dtw)
                (x1, x2) for x1, x2 in izip(xes1, xes2))
        dtw_costs = zip(*res)[0]
        self.print_mean_DTW_costs(dtw_costs)
        ds = zip(*self._data_same)
        rs = zip(*res)
        data_same_00shapes = self._data_same[0][0].shape
        data_same_01shapes = self._data_same[0][1].shape
        print data_same_00shapes
        print data_same_01shapes
        self._data_same = zip(ds[0], ds[1], rs[-2], rs[-1])
        data_same_00shapes = self._data_same[0][0].shape
        data_same_01shapes = self._data_same[0][1].shape
        print data_same_00shapes
        print data_same_01shapes
        self._margin = 0  # TODO CORRECT THAT IF NEEDED
        self.remix()


    def print_mean_DTW_costs(self, dtw_costs):
        print "mean DTW cost", numpy.mean(dtw_costs), "std dev", numpy.std(dtw_costs)
        print "mean word length in frames", numpy.mean(self._words_frames), "std dev", numpy.std(self._words_frames)
        print "mean DTW cost per frame", numpy.mean(dtw_costs/self._words_frames), "std dev", numpy.std(dtw_costs/self._words_frames)
