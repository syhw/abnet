"""Runs an ABnet (Siamese network) on brain data

Usage:
    run_exp_eeg.py [--dataset-path=path]
    [--batch-size=100]
    [--init-lr=0.001] [--epochs=100] 
    [--trainer-type=adadelta] 
    [--prefix-output-fname=my_prefix_42] [--debug-print=0] 
    [--debug-time] [--debug-plot=0]


Options:
    -h --help                  Show this screen
    --version                  Show version
    --dataset-path=str         A valid path to the dataset
    default is timit
    --batch-size=int           Batch size, used only by the batch iterator
    default is 100 (unused for "sentences" iterator type)
    --init-lr=float            Initial learning rate for SGD
    default is 0.001 (that is very low intentionally)
    --epochs=int               Max number of epochs (always early stopping)
    default is 100
    --trainer-type=str         "SGD" | "adagrad" | "adadelta"
    default is "adadelta"
    --prefix-output-fname=str  An additional prefix to the output file name
    default is "" (empty string)
    --debug-print=int          Level of debug printing. 0: nothing, 1: network
    default is 0               2: epochs/iters related
    --debug-time               Flag that activates timing epoch duration
    default is False, using it makes it True
    --debug-plot=int           Level of debug plotting, 1: costs
    default is 0               >= 2: gradients & updates
"""

import socket, docopt, cPickle, time, sys, os
import numpy
import matplotlib
matplotlib.use('Agg')
try:
    import prettyplotlib as ppl
except:
    print >> sys.stderr, "you should install prettyplotlib"
import matplotlib.pyplot as plt
import joblib
import random
from numpy.random import shuffle

from prep_timit import load_data
from layers import Linear, ReLU, SigmoidLayer, SoftPlus
from classifiers import LogisticRegression
from nnet_archs import ABNeuralNet2Outputs
#from nnet_archs import DropoutABNeuralNet2Outputs # TODO

DEBUG = False
DIM_EMBEDDING = 100


class DatasetEEGIterator(object):
    def __init__(self, data, normalize=False, min_max_scale=False,
            scale_f1=None, scale_f2=None,
            batch_size=1, only_same=False):
        self._scale_f1 = scale_f1
        self._scale_f2 = scale_f2
        self._data = data

    def __iter__(self):
        # TODO batch_size
        # TODO equilibrate same/different subjects/conditions cross-all
        for i, l1 in enumerate(self._data):
            for l2 in self._data[i+1:]:
                y1 = (l1[1] == l2[1])  # condition
                y2 = (l1[0] == l2[0])  # subject
                yield [[[l1[2:]], [l2[2:]]],
                       [[y1], [y2]]]


class DatasetEEGCachedIterator(DatasetEEGIterator):
    def __init__(self, data, normalize=False, min_max_scale=False,
            scale_f1=None, scale_f2=None,
            batch_size=1, only_same=False):
        super(DatasetEEGCachedIterator, self).__init__(data, normalize,
                min_max_scale, scale_f1, scale_f2, batch_size, only_same)
        self.batch_size = batch_size
        self._x1 = []
        self._x2 = []
        self._y1 = []
        self._y2 = []
        min_ratio = 0.25  # TODO
        same_c = 0.01
        diff_c = 0.01
        same_s = 0.01
        diff_s = 0.01
        for i, l1 in enumerate(self._data):
            for l2 in self._data[i+1:]:
                # TODO BALANCE SMARTLY HERE
                y1 = int(l1[1] == l2[1])  # condition
                y2 = int(l1[0] == l2[0])  # subject
                add_it = False
                #if y1 == 0 and y2 == 0: TODO
                if y1 == 0 or y2 == 0:
                    if same_c / (same_c + diff_c) > min_ratio and same_s / (same_s + diff_s) > min_ratio:
                        add_it = True
                else:
                    # TODO
                    add_it = True
                if add_it:
                    self._x1.append(l1[2:])
                    self._x2.append(l2[2:])
                    self._y1.append(y1)
                    self._y2.append(y2)
                    same_c += y1
                    same_s += y2
                    diff_c += 1-y1
                    diff_s += 1-y2
        self._x1 = numpy.asarray(self._x1, dtype='float32')
        self._x2 = numpy.asarray(self._x2, dtype='float32')
        self._y1 = numpy.asarray(self._y1, dtype='int32')
        self._y2 = numpy.asarray(self._y2, dtype='int32')
#        print self._x1
#        print self._y1
#        print self._x1.shape
#        print self._y1.shape
#        print self._x1.dtype
#        print self._y1.dtype

    def __iter__(self):
        bs = self.batch_size
        for i in xrange(0, self._x1.shape[0], bs):
            yield [[self._x1[i:i+bs], self._x2[i:i+bs]],
                   [self._y1[i:i+bs], self._y2[i:i+bs]]]


def print_mean_weights_biases(params):
    for layer_ind, param in enumerate(params):
        filler = "weight"
        if layer_ind % 2:
            filler = "bias"
        print("layer %i mean %s values %f and std devs %f" % (layer_ind/2, 
            filler, numpy.mean(param.eval()), numpy.std(param.eval())))


def plot_costs(cost):
    # TODO
    pass


def rolling_avg_pgu(iteration, pgu, l):
    # (iteration * pgu + l) / (iteration + 1)
    assert len(l) == len(pgu)
    ll = len(l)/3
    params, gparams, updates = l[:ll], l[ll:-ll], l[-ll:]
    mpars, mgpars, mupds = pgu[:ll], pgu[ll:-ll], pgu[-ll:]
    ii = iteration + 1
    return [(iteration * mpars[k] + p) / ii for k, p in enumerate(params)] +\
            [(iteration * mgpars[k] + g) / ii for k, g in enumerate(gparams)] +\
            [(iteration * mupds[k] + u) / ii for k, u in enumerate(updates)]


def plot_params_gradients_updates(n, l):
    # TODO currently works only with THEANO_FLAGS="device=cpu" (not working on
    #CudaNDArrays)
    def plot_helper(li, ti, p):
        if ppl == None:
            print >> sys.stderr, "cannot plot this without prettyplotlib"
            return
        fig, ax = plt.subplots(1)
        if li % 2:
            title = "biases" + ti
            ppl.bar(ax, numpy.arange(p.shape[0]), p) # TODO with plt
        else:
            title = "weights" + ti
            ppl.pcolormesh(fig, ax, p) # TODO with plt
        plt.title(title)
        plt.savefig(title + ".png")
        #ppl.show()
        plt.close()
    ll = len(l)/3
    params, gparams, updates = l[:ll], l[ll:-ll], l[-ll:]
    if DEBUG:
        print "params"
        print params
        print "===================="
        print "gparams"  # TODO find out why not CudaNDArray here
        print gparams
        print "===================="
        print "updates"  # TODO find out why not CudaNDArray here
        print updates
    title_iter =  "_%04i" % n
    for layer_ind, param in enumerate(params):
        title = "_for_layer_" + str(layer_ind/3) + title_iter
        plot_helper(layer_ind, title, param)
    for layer_ind, gparam in enumerate(gparams):
        title = "_gradients_for_layer_" + str(layer_ind/3) + title_iter
        plot_helper(layer_ind, title, gparam)
    for layer_ind, update in enumerate(updates):
        title = "_updates_for_layer_" + str(layer_ind/3) + title_iter
        plot_helper(layer_ind, title, update)


def run(dataset_path,
        batch_size=100,
        init_lr=0.01, max_epochs=100, 
        trainer_type="adadelta",
        layers_types=[ReLU, ReLU, ReLU, ReLU, ReLU],
        layers_sizes=[1400, 1400, 1400, 1400],
        dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
        prefix_fname='',
        debug_print=0,
        debug_time=False,
        debug_plot=0):
    """
    FIXME TODO
    """

    output_file_name = 'eeg_Leo'
    if prefix_fname != "":
        output_file_name = prefix_fname + "_"
    output_file_name += "_" + trainer_type
    output_file_name += "_emb_" + str(DIM_EMBEDDING)
    print "output file name:", output_file_name

    n_ins = None
    n_outs = None
    print "loading dataset from", dataset_path
    if dataset_path[-7:] != '.joblib':
        print >> sys.stderr, "prepare your dataset!!"
        sys.exit(-1)

    ### LOADING DATA
    data = joblib.load(dataset_path)
    shuffle(data)
    print data.shape

    has_dev_and_test_set = True
    dev_dataset_path = dataset_path[:-7].replace("train", "") + 'dev.joblib'
    test_dataset_path = dataset_path[:-7].replace("train", "") + 'test.joblib'
    dev_split_at = len(data)
    test_split_at = len(data)
    if not os.path.exists(dev_dataset_path) or not os.path.exists(test_dataset_path):
        has_dev_and_test_set = False
        dev_split_at = int(0.8 * dev_split_at)
        test_split_at = int(0.9 * test_split_at)
#        dev_split_at = int(0.96 * dev_split_at)
#        test_split_at = int(0.98 * test_split_at)

    n_ins = data[0].shape[0] - 2
    n_outs = DIM_EMBEDDING

    normalize = False
    min_max_scale = False

    ### TRAIN SET
    if has_dev_and_test_set:
        train_set_iterator = DatasetEEGCachedIterator(data,
                normalize=normalize, min_max_scale=min_max_scale,
                scale_f1=None, scale_f2=None, batch_size=batch_size)
    else:
        train_set_iterator = DatasetEEGCachedIterator(data[:dev_split_at],
                normalize=normalize, min_max_scale=min_max_scale,
                scale_f1=None, scale_f2=None, batch_size=batch_size)
    f1 = train_set_iterator._scale_f1
    f2 = train_set_iterator._scale_f2

    ### DEV SET
    if has_dev_and_test_set:
        data = joblib.load(dev_dataset_path)
        valid_set_iterator = DatasetEEGCachedIterator(data,
                normalize=normalize, min_max_scale=min_max_scale,
                scale_f1=f1, scale_f2=f2, batch_size=batch_size)
    else:
        valid_set_iterator = DatasetEEGCachedIterator(data[dev_split_at:test_split_at],
                normalize=normalize, min_max_scale=min_max_scale,
                scale_f1=f1, scale_f2=f2, batch_size=batch_size)

    ### TEST SET
    if has_dev_and_test_set:
        data = joblib.load(test_dataset_path)
        test_set_iterator = DatasetEEGCachedIterator(data,
                normalize=normalize, min_max_scale=min_max_scale,
                scale_f1=f1, scale_f2=f2, batch_size=batch_size)
    else:
        test_set_iterator = DatasetEEGCachedIterator(data[test_split_at:],
                normalize=normalize, min_max_scale=min_max_scale,
                scale_f1=f1, scale_f2=f2, batch_size=batch_size)

    assert n_ins != None
    assert n_outs != None

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'

    # TODO the proper network type other than just dropout or not
    nnet = None
    nnet = ABNeuralNet2Outputs(numpy_rng=numpy_rng, 
            n_ins=n_ins,
            layers_types=layers_types,
            layers_sizes=layers_sizes,
            n_outs=n_outs,
            loss='cos_cos2',
            rho=0.95,
            eps=1.E-6,
            max_norm=0.,
            debugprint=debug_print)
    print "Created a neural net as:",
    print str(nnet)

    # get the training, validation and testing function for the model
    print '... getting the training functions'
    print trainer_type
    train_fn = None
    if debug_plot or debug_print:
        if trainer_type == "adadelta":
            train_fn = nnet.get_adadelta_trainer(debug=True)
        elif trainer_type == "adagrad":
            train_fn = nnet.get_adagrad_trainer(debug=True)
        else:
            train_fn = nnet.get_SGD_trainer(debug=True)
    else:
        if trainer_type == "adadelta":
            train_fn = nnet.get_adadelta_trainer()
        elif trainer_type == "adagrad":
            train_fn = nnet.get_adagrad_trainer()
        else:
            train_fn = nnet.get_SGD_trainer()

    train_scoref_c = nnet.score_classif_same_diff_word_separated(train_set_iterator)
    valid_scoref_c = nnet.score_classif_same_diff_word_separated(valid_set_iterator)
    test_scoref_c = nnet.score_classif_same_diff_word_separated(test_set_iterator)
    train_scoref_s = nnet.score_classif_same_diff_spkr_separated(train_set_iterator)
    valid_scoref_s = nnet.score_classif_same_diff_spkr_separated(valid_set_iterator)
    test_scoref_s = nnet.score_classif_same_diff_spkr_separated(test_set_iterator)
    data_iterator = train_set_iterator

    print '... training the model'
    # early-stopping parameters

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    lr = init_lr
    timer = None
    if debug_plot:
        print_mean_weights_biases(nnet.params)
    #with open(output_file_name + 'epoch_0.pickle', 'wb') as f:
    #    cPickle.dump(nnet, f, protocol=-1)

    while (epoch < max_epochs):
        epoch = epoch + 1
        avg_costs = []
        avg_params_gradients_updates = []
        if debug_time:
            timer = time.time()
        for iteration, (x, y) in enumerate(data_iterator):
#            print "x[0]", x[0]
#            print "x[1]", x[1]
#            print "y[0]", y[0]
#            print "y[1]", y[1]
            avg_cost = 0.
            if "delta" in trainer_type:
                avg_cost = train_fn(x[0], x[1], y[0], y[1])
            else:
                avg_cost = train_fn(x[0], x[1], y[0], y[1], lr)
            if debug_print >= 3:
                print "cost:", avg_cost[0]
            if debug_plot >= 2:
                plot_costs(avg_cost[0])
                if not len(avg_params_gradients_updates):
                    avg_params_gradients_updates = map(numpy.asarray, avg_cost[1:])
                else:
                    avg_params_gradients_updates = rolling_avg_pgu(
                            iteration, avg_params_gradients_updates,
                            map(numpy.asarray, avg_cost[1:]))
            if debug_plot >= 3:
                plot_params_gradients_updates(iteration, avg_cost[1:])
            if type(avg_cost) == list:
                avg_costs.append(avg_cost[0])
            else:
                avg_costs.append(avg_cost)
            if iteration > 2:  # TODO remove
                break          # TODO remove
        if debug_print >= 2:
            print_mean_weights_biases(nnet.params)
        if debug_plot >= 2:
            plot_params_gradients_updates(epoch, avg_params_gradients_updates)
        if debug_time:
            print('  epoch %i took %f seconds' % (epoch, time.time() - timer))
        avg_cost = numpy.mean(avg_costs)
        if numpy.isnan(avg_cost):
            print("avg costs is NaN so we're stopping here!")
            break
        print('  epoch %i, avg costs %f' % \
              (epoch, avg_cost))

        tmp_train = zip(*train_scoref_c())
        print('  epoch %i, training sim same conds %f, diff conds %f' % \
              (epoch, numpy.mean(tmp_train[0]), numpy.mean(tmp_train[1])))
        tmp_train = zip(*train_scoref_s())
        print('  epoch %i, training sim same subjs %f, diff subjs %f' % \
              (epoch, numpy.mean(tmp_train[0]), numpy.mean(tmp_train[1])))

        # TODO update lr(t) = lr(0) / (1 + lr(0) * lambda * t)
        lr = numpy.float32(init_lr / (numpy.sqrt(iteration) + 1.)) ### TODO
        #lr = numpy.float32(init_lr / (iteration + 1.)) ### TODO
        # or another scheme for learning rate decay
        #with open(output_file_name + 'epoch_' +str(epoch) + '.pickle', 'wb') as f:
        #    cPickle.dump(nnet, f, protocol=-1)

        # we check the validation loss on every epoch
        validation_losses_c = zip(*valid_scoref_c())
        validation_losses_s = zip(*valid_scoref_s())
        this_validation_loss = 0.25*(1.-numpy.mean(validation_losses_c[0])) +\
                0.25*numpy.mean(validation_losses_c[1]) +\
                0.25*(1.-numpy.mean(validation_losses_s[0])) +\
                0.25*numpy.mean(validation_losses_s[1])

        print('  epoch %i, valid sim same conds %f, diff conds %f' % \
              (epoch, numpy.mean(validation_losses_c[0]), numpy.mean(validation_losses_c[1])))
        print('  epoch %i, valid sim same subjs %f, diff subjs %f' % \
              (epoch, numpy.mean(validation_losses_s[0]), numpy.mean(validation_losses_s[1])))
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            with open(output_file_name + '.pickle', 'wb') as f:
                cPickle.dump(nnet, f, protocol=-1)
            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            # test it on the test set
            test_losses_c = zip(*test_scoref_c())
            test_losses_s = zip(*test_scoref_s())
            print('  epoch %i, test sim same conds %f, diff conds %f' % \
                  (epoch, numpy.mean(test_losses_c[0]), numpy.mean(test_losses_c[1])))
            print('  epoch %i, test sim same subjs %f, diff subjs %f' % \
                  (epoch, numpy.mean(test_losses_s[0]), numpy.mean(test_losses_s[1])))

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f, '
           'with test performance %f') %
                 (best_validation_loss, test_score))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    with open(output_file_name + '_final.pickle', 'wb') as f:
        cPickle.dump(nnet, f, protocol=-1)

if __name__=='__main__':
    arguments = docopt.docopt(__doc__, version='run_exp version 0.1')
    dataset_path = ''
    if arguments['--dataset-path'] != None:
        dataset_path = arguments['--dataset-path']
    batch_size = 100
    if arguments['--batch-size'] != None:
        batch_size = int(arguments['--batch-size'])
    init_lr = 0.01
    if arguments['--init-lr'] != None:
        init_lr = float(arguments['--init-lr'])
    max_epochs = 100
    if arguments['--epochs'] != None:
        max_epochs = int(arguments['--epochs'])
    trainer_type = 'adadelta'
    if arguments['--trainer-type'] != None:
        trainer_type = arguments['--trainer-type']
    prefix_fname = ''
    if arguments['--prefix-output-fname'] != None:
        prefix_fname = arguments['--prefix-output-fname']
    debug_print = 0
    if arguments['--debug-print']:
        debug_print = int(arguments['--debug-print'])
    debug_time = False
    if arguments['--debug-time']:
        debug_time = True
    debug_plot = 0
    if arguments['--debug-plot']:
        debug_plot = int(arguments['--debug-plot'])

    run(dataset_path=dataset_path,
        batch_size=batch_size,
        init_lr=init_lr, max_epochs=max_epochs, 
        trainer_type=trainer_type,
        layers_types=[ReLU, SigmoidLayer],
        layers_sizes=[200],
        prefix_fname=prefix_fname,
        debug_print=debug_print,
        debug_time=debug_time,
        debug_plot=debug_plot)

    #THEANO_FLAGS='device=gpu0' python run_exp_AB_eeg.py --dataset-path=eeg.joblib --debug-print=1 --debug-time

