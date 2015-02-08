from layers import Linear, ReLU, dropout, fast_dropout
from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from nnet_archs import build_shared_zeros
import sys

class ABCNeuralNet(object):
    """
    A neural net that takes A B C triplets, along with Y_{AB}_i and Y_{AC}_i
    same/different information for i in [1..n_embs]
    """
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[ReLU, ReLU, ReLU, ReLU, ReLU],
            layers_sizes=[1024, 1024, 1024, 1024],
            n_outs=100,
            n_embs=2,
            loss='cos2',
            rho=0.9, eps=1.E-6,
            max_norm=0.,
            debugprint=False):
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x1 = T.fmatrix('x1')
        self.x2 = T.fmatrix('x2')
        self.x3 = T.fmatrix('x3')
        self.y12s = [T.ivector('y12') for _ in n_embs]
        self.y13s = [T.ivector('y13') for _ in n_embs]
        self.n_embs = n_embs
        
        self.layers_ins = [n_ins] + layers_sizes[:-1]
        self.layers_outs = layers_sizes
        layer_input1 = self.x1
        layer_input2 = self.x2
        layer_input3 = self.x3
        layer_embs1 = [None for _ in n_embs]
        layer_embs2 = [None for _ in n_embs]
        layer_embs3 = [None for _ in n_embs]
        
        for layer_ind, (layer_type, n_in, n_out) in enumerate(
                zip(layers_types[:-1], self.layers_ins, self.layers_outs)):
            this_layer1 = layer_type(rng=numpy_rng,
                    input=layer_input1, n_in=n_in, n_out=n_out)#, cap=6.)
            assert hasattr(this_layer1, 'output')
            layer_input1 = this_layer1.output
            self.params.extend(this_layer1.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer1.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer1.params])
            self.layers.append(this_layer1)
            this_layer2 = layer_type(rng=numpy_rng,
                    input=layer_input2, n_in=n_in, n_out=n_out,
                    W=this_layer1.W, b=this_layer1.b)#, cap=6.)
            assert hasattr(this_layer2, 'output')
            this_layer3 = layer_type(rng=numpy_rng,
                    input=layer_input2, n_in=n_in, n_out=n_out,
                    W=this_layer1.W, b=this_layer1.b)#, cap=6.)
            assert hasattr(this_layer3, 'output')
            layer_input2 = this_layer2.output
            self.layers.append(this_layer2)
            layer_input3 = this_layer3.output
            self.layers.append(this_layer3)

        for i_emb in xrange(n_embs):
            emb_layer_type = layers_types[-1]
            n_in = layers_sizes[-1]
            n_out = n_outs
            this_emb_layer1 = emb_layer_type(rng=numpy_rng,
                    input=layer_input1, n_in=n_in, n_out=n_out)#, cap=6.)
            assert hasattr(this_emb_layer1, 'output')
            layer_embs1[i_emb] = this_emb_layer1.output
            self.params.extend(this_emb_layer1.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_emb_layer1.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_emb_layer1.params])
            this_emb_layer2 = emb_layer_type(rng=numpy_rng,
                    input=layer_input2, n_in=n_in, n_out=n_out,
                    W=this_emb_layer1.W, b=this_emb_layer1.b)#, cap=6.)
            assert hasattr(this_emb_layer2, 'output')
            layer_embs2[i_emb] = this_emb_layer2.output
            this_emb_layer3 = emb_layer_type(rng=numpy_rng,
                    input=layer_input3, n_in=n_in, n_out=n_out,
                    W=this_emb_layer1.W, b=this_emb_layer1.b)#, cap=6.)
            assert hasattr(this_emb_layer3, 'output')
            layer_embs3[i_emb] = this_emb_layer3.output
            self.layers.append(this_emb_layer1)
            self.layers.append(this_emb_layer2)
            self.layers.append(this_emb_layer3)

        L2 = 0.
        for param in self.params:
            L2 += T.sum(param ** 2)
        L1 = 0.
        for param in self.params:
            L1 += T.sum(abs(param))

        self.cos2_costs = []
        self.acos_costs = []
        
        for i_emb in xrange(n_embs):
            y12 = self.y12s[i_emb]
            y13 = self.y13s[i_emb]
            self.cos_sim_12 = (T.sum(layer_input1 * layer_input2, axis=-1) /
                (layer_input1.norm(2, axis=-1) * layer_input2.norm(2, axis=-1)))
            self.cos_sim_13 = (T.sum(layer_input1 * layer_input3, axis=-1) /
                (layer_input1.norm(2, axis=-1) * layer_input3.norm(2, axis=-1)))
            self.acos_sim_12 = T.arccos(self.cos_sim_12)
            self.acos_sim_13 = T.arccos(self.cos_sim_13)

            self.cos2_sim_12 = self.cos_sim_12**2
            self.cos2_sim_13 = self.cos_sim_13**2

            cos2_sim_cost = (y12+y13)*1. +\
                    T.switch(y12, self.cos2_sim_12, -self.cos2_sim_12) +\
                    T.switch(y13, self.cos2_sim_13, -self.cos2_sim_13) 
            acos_sim_cost = (y12+y13)*1. +\
                    T.switch(y12, self.acos_sim_12, -self.acos_sim_12) +\
                    T.switch(y13, self.acos_sim_13, -self.acos_sim_13) 
            self.cos2_costs.append(cos2_sim_cost)
            self.acos_costs.append(acos_sim_cost)

 # TODO HERE
        self.cos2_sim_cost = T.sum(self.cos2_costs)
        self.mean_cos2_sim_cost = T.mean(self.cos2_sim_cost)
        self.sum_cos2_sim_cost = T.sum(self.cos2_sim_cost)

        self.acos_sim_cost = T.sum(self.acos_costs)
        self.mean_acos_sim_cost = T.mean(self.acos_sim_cost)
        self.sum_acos_sim_cost = T.sum(self.acos_sim_cost)

        if loss == 'cos2':
            self.cost = self.sum_cos_cos2_sim_cost
            self.mean_cost = self.mean_cos_cos2_sim_cost
        elif loss == 'acos':
            self.cost = self.sum_cos_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        else:
            print >> sys.stderr, "NO COST FUNCTION"
            sys.exit(-1)

        if debugprint:
            theano.printing.debugprint(self.cost)

        if hasattr(self, 'cost'):
            self.cost_training = self.cost
        if hasattr(self, 'mean_cost'):
            self.mean_cost_training = self.mean_cost

    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
            zip(self.layers_types, dimensions_layers_str)))

    def get_SGD_trainer(self, debug=False):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_x3 = T.fmatrix('batch_x3')
        batch_y12s = [T.ivector('batch_y12') for _ in self.n_embs]
        batch_y13s = [T.ivector('batch_y13') for _ in self.n_embs]
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent on the batch size
        cost = self.mean_cost_training
        gparams = T.grad(cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate 

        outputs = cost
        if debug:
            outputs = [cost] + self.params + gparams +\
                    [updates[param] for param in self.params]

        train_fn = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_x3),
            theano.Param(batch_y12s), theano.Param(batch_y13s), 
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3,
                self.y12s: batch_y12s, self.y13s: batch_y13s})

        return train_fn

    def get_adagrad_trainer(self, debug=False):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_x3 = T.fmatrix('batch_x3')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        cost = self.mean_cost_training
        gparams = T.grad(cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            if self.max_norm:
                W = param + dx
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param + dx
            updates[accugrad] = agrad

        outputs = cost
        if debug:
            outputs = [cost] + self.params + gparams +\
                    [updates[param] for param in self.params]

        train_fn = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_x3),
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3})

        return train_fn

    def get_adadelta_trainer(self, debug=False):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_x3 = T.fmatrix('batch_x3')
        # compute the gradients with respect to the model parameters
        cost = self.mean_cost_training
        gparams = T.grad(cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            if self.max_norm:
                W = param + dx
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param + dx
            updates[accugrad] = agrad

        outputs = cost
        if debug:
            outputs = [cost] + self.params + gparams +\
                    [updates[param] for param in self.params]

        train_fn = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_x3)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3})

        return train_fn

    def score_classif(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_x3 = T.fmatrix('batch_x3')
        score = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_x3)],
                outputs=self.cost,
                givens={self.x1: batch_x1, self.x2: batch_x2,
                    self.x3: batch_x3})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(x[0], x[1], x[2]) for x in given_set]

        return scoref

    def score_classif_same_diff_word_separated(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_x3 = T.fmatrix('batch_x3')
        # TODO HERE
        cost_same = T.mean(self.cos_sim1[T.eq(self.y1, 1).nonzero()], axis=-1)
        cost_diff = T.mean(self.cos_sim1[T.eq(self.y1, 0).nonzero()], axis=-1)
        score1 = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y1)],
                outputs=[cost_same, cost_diff],
                givens={self.x1: batch_x1, self.x2: batch_x2,
                    self.y1: batch_y1})

        # Create a function that scans the entire set given as input
        def scoref1():
            return [score1(x[0], x[1], y[0]) for (x, y) in given_set]

        return scoref1

    def score_classif_same_diff_spkr_separated(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y2 = T.ivector('batch_y2')
        cost_same = T.mean(self.cos_sim2[T.eq(self.y2, 1).nonzero()], axis=-1)
        cost_diff = T.mean(self.cos_sim2[T.eq(self.y2, 0).nonzero()], axis=-1)
        score2 = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y2)],
                outputs=[cost_same, cost_diff],
                givens={self.x1: batch_x1, self.x2: batch_x2,
                    self.y2: batch_y2})

        # Create a function that scans the entire set given as input
        def scoref2():
            return [score2(x[0], x[1], y[1]) for (x, y) in given_set]

        return scoref2

    def transform_x1(self):
        batch_x1 = T.fmatrix('batch_x1')
        transform = theano.function(inputs=[theano.Param(batch_x1)],
                outputs=[self.layers[-4].output, self.layers[-2].output],
                givens={self.x1: batch_x1})
        return transform

