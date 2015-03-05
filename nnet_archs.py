from layers import Linear, ReLU, dropout, fast_dropout
from classifiers import LogisticRegression
from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared
import sys


def build_shared_zeros(shape, name):
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


class NeuralNet(object):  # TODO refactor with a base class for this and AB
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
            layers_sizes=[1024, 1024, 1024, 1024],
            n_outs=62 * 3,
            rho=0.9, eps=1.E-6,  # TODO refine
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

        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
            this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])
            self.layers.append(this_layer)
            layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
        self.cost = self.layers[-1].training_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)

        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
            zip(self.layers_types, dimensions_layers_str)))


    def get_SGD_trainer(self, debug=False):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent on the batch size
        gparams = T.grad(self.mean_cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            if self.max_norm:
                W = param - gparam * learning_rate
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param - gparam * learning_rate 

        outputs = self.cost
        if debug:
            outputs = [self.cost] + self.params + gparams +\
                    [updates[param] for param in self.params]# +\

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self, debug=False):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and 
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)

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

        outputs = self.cost
        if debug:
            outputs = [self.cost] + self.params + gparams +\
                    [updates[param] for param in self.params]# +\

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y)],
            outputs=outputs,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adagrad_trainer(self, debug=False):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)

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

        outputs = self.cost
        if debug:
            outputs = [self.cost] + self.params + gparams +\
                    [updates[param] for param in self.params]# +\

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        """ Returns functions to get current classification scores. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x), theano.Param(batch_y)],
                outputs=self.errors,
                givens={self.x: batch_x, self.y: batch_y})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref

    def fit(self, X, y, max_epochs=30, batch_size=1000):
        # TODO early stopping
        train_fn = self.get_adadelta_trainer()
        #best_validation_loss = numpy.inf
        epoch = 0
        X_iter = [X[i*batch_size:(i+1)*batch_size] 
                for i in xrange((X.shape[0]+batch_size-1)/batch_size)]
        y_iter = [y[i*batch_size:(i+1)*batch_size] 
                for i in xrange((y.shape[0]+batch_size-1)/batch_size)]
        done_looping = False
        from itertools import izip
        while (epoch < max_epochs) and (not done_looping):
            epoch = epoch + 1
            for iteration, (xx, yy) in enumerate(izip(X_iter, y_iter)):
                avg_cost = train_fn(xx, yy)

    def predict(self, X):
        batch_x = T.fmatrix('batch_x')
        fun = theano.function(inputs=[theano.Param(batch_x)],
                outputs=self.layers[-1].output,
                givens={self.x: batch_x})
        return fun(X)


class DropoutNet(NeuralNet):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
            layers_sizes=[1024, 1024, 1024, 1024],
            dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
            n_outs=62 * 3,
            rho=0.95, eps=1.E-6,
            max_norm=0.,
            fast_drop=False,
            debugprint=False):
        super(DropoutNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, max_norm,
                debugprint)

        self.dropout_rates = dropout_rates
        if fast_drop:
            if dropout_rates[0]:
                dropout_layer_input = fast_dropout(numpy_rng, self.x,
                        dropout_rates[0])
            else:
                dropout_layer_input = self.x
        else:
            dropout_layer_input = dropout(numpy_rng, self.x,
                    p=dropout_rates[0])
        self.dropout_layers = []

        for layer, layer_type, n_in, n_out, dr in zip(self.layers,
                layers_types, self.layers_ins, self.layers_outs,
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything 
                                            # from the last layer !!!
            if dr:
                if fast_drop:
                    this_layer = layer_type(rng=numpy_rng,
                            input=dropout_layer_input, n_in=n_in, n_out=n_out,
                            W=layer.W, b=layer.b, fdrop=dr)
                else:
                    this_layer = layer_type(rng=numpy_rng,
                            input=dropout_layer_input, n_in=n_in, n_out=n_out,
                            W=layer.W * 1. / (1. - dr), # experimental
                            b=layer.b * 1. / (1. - dr)) # TODO check
                    # N.B. dropout with dr=1 does not dropanything!!
                    this_layer.output = dropout(numpy_rng,
                            this_layer.output, dr)
            else:
                this_layer = layer_type(rng=numpy_rng,
                        input=dropout_layer_input, n_in=n_in, n_out=n_out,
                        W=layer.W, b=layer.b)
            assert hasattr(this_layer, 'output')
            self.dropout_layers.append(this_layer)
            dropout_layer_input = this_layer.output
        print self

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        # these are the dropout costs
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        self.cost = self.dropout_layers[-1].training_cost(self.y)

        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)


class ABNeuralNet(object):  #NeuralNet):
    # TODO refactor
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[ReLU, ReLU, ReLU, ReLU, ReLU],
            layers_sizes=[1024, 1024, 1024, 1024],
            n_outs=100,
            loss='cos_cos2',
            rho=0.9, eps=1.E-6,
            max_norm=0.,
            debugprint=False):
        #super(AB_NeuralNet, self).__init__(numpy_rng, theano_rng,
        #        n_ins, layers_types, layers_sizes, n_outs, rho, eps,
        #        debugprint)
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
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        layer_input1 = self.x1
        layer_input2 = self.x2
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
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
            layer_input2 = this_layer2.output
            self.layers.append(this_layer2)

        L2 = 0.
        for param in self.params:
            L2 += T.sum(param ** 2)
        L1 = 0.
        for param in self.params:
            L1 += T.sum(abs(param))

        self.squared_error = (layer_input1 - layer_input2).norm(2, axis=-1) **2
        self.mse = T.mean(self.squared_error, axis=-1)
        self.rmse = T.sqrt(self.mse)
        self.sse = T.sum(self.squared_error, axis=-1)
        self.rsse = T.sqrt(self.sse)

        self.rsse_cost = T.switch(self.y, self.rsse, -self.rsse)
        self.rmse_cost = T.switch(self.y, self.rmse, -self.rmse)
        self.sum_rmse_costs = T.sum(self.rmse_cost)
        self.sum_rsse_costs = T.sum(self.rsse_cost)
        self.mean_rmse_costs = T.mean(self.rmse_cost)
        self.mean_rsse_costs = T.mean(self.rsse_cost)

        #self.cross_entropy = - (0.5 * T.sum(layer_input1 * T.log(layer_input2)
        #    + (1 - layer_input1) * T.log(1 - layer_input2), axis=-1)) + (0.5 *
        #        T.sum(layer_input2 * T.log(layer_input1) + (1 - layer_input2) *
        #            T.log(1 - layer_input1), axis=-1))
        #self.cross_entropy_cost = T.switch(self.y, self.cross_entropy,
        #        -self.cross_entropy)

        #self.cos_sim = (T.batched_dot(layer_input1, layer_input2) /
        self.cos_sim = (T.sum(layer_input1 * layer_input2, axis=-1) /
            (layer_input1.norm(2, axis=-1) * layer_input2.norm(2, axis=-1)))
        self.cos_sim_cost = T.switch(self.y, 1.-self.cos_sim, self.cos_sim)
        self.mean_cos_sim_cost = T.mean(self.cos_sim_cost)
        self.sum_cos_sim_cost = T.sum(self.cos_sim_cost)

        #self.cos_sim_cost = T.switch(self.y, 1.-abs(self.cos_sim), abs(self.cos_sim))
        self.cos2_sim_cost = T.switch(self.y, 1.-(self.cos_sim ** 2), self.cos_sim ** 2)
        self.mean_cos2_sim_cost = T.mean(self.cos2_sim_cost)
        self.sum_cos2_sim_cost = T.sum(self.cos2_sim_cost)

        self.cos_cos2_sim_cost = T.switch(self.y, (1.-self.cos_sim)/2, self.cos_sim ** 2)
        self.mean_cos_cos2_sim_cost = T.mean(self.cos_cos2_sim_cost)
        self.sum_cos_cos2_sim_cost = T.sum(self.cos_cos2_sim_cost)

        from layers import relu_f
        #self.dot_prod = T.batched_dot(layer_input1, layer_input2)
        self.dot_prod = T.sum(layer_input1 * layer_input2, axis=-1)
        self.cos_hinge_cost = T.switch(self.y, (1.-self.cos_sim)/2, relu_f(self.dot_prod)) # TODO
        self.mean_cos_hinge_cost = T.mean(self.cos_hinge_cost) # TODO
        self.sum_cos_hinge_cost = T.sum(self.cos_hinge_cost) # TODO

        self.dot_prod_cost = relu_f(T.switch(self.y, 1.-self.dot_prod, self.dot_prod))
        #self.dot_prod_cost = T.switch(self.y, 1.-self.dot_prod, self.dot_prod)
        self.mean_dot_prod_cost = T.mean(self.dot_prod_cost)
        self.sum_dot_prod_cost = T.sum(self.dot_prod_cost)

        self.euclidean = (layer_input1 - layer_input2).norm(2, axis=-1)
        self.euclidean_cost = T.switch(self.y, self.euclidean, -self.euclidean)
        self.mean_euclidean_cost = T.mean(self.euclidean_cost)
        self.sum_euclidean_cost = T.sum(self.euclidean_cost)

        self.normalized_euclidean = ((layer_input1 - layer_input2).norm(2, axis=-1) / (layer_input1.norm(2, axis=-1) * layer_input2.norm(2, axis=-1)))
        self.normalized_euclidean_cost = T.switch(self.y, self.normalized_euclidean, -self.normalized_euclidean)
        self.mean_normalized_euclidean_cost = T.mean(self.normalized_euclidean_cost)
        self.sum_normalized_euclidean_cost = T.sum(self.normalized_euclidean_cost)

        self.hellinger = 0.5 * T.sqrt(T.sum((T.sqrt(layer_input1) - T.sqrt(layer_input2))**2, axis=1))
        self.hellinger_cost = T.switch(self.y, self.hellinger, 1.-self.hellinger)
        self.mean_hellinger_cost = T.mean(self.hellinger_cost)
        self.sum_hellinger_cost = T.sum(self.hellinger_cost)

        if loss == 'cos_cos2':
            self.cost = self.sum_cos_cos2_sim_cost
            self.mean_cost = self.mean_cos_cos2_sim_cost
        elif loss == 'cos':
            self.cost = self.sum_cos_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        elif loss == 'cos2':
            self.cost = self.sum_cos2_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        elif loss == 'cos_hinge':
            #self.cost = self.sum_cos_hinge_cost
            #self.mean_cost = self.mean_cos_hinge_cost
            print >> sys.stderr, "COST TODO"
            sys.exit(-1)
        elif loss == 'dot_prod':
            self.cost = self.sum_dot_prod_cost
            self.mean_cost = self.mean_dot_prod_cost
        elif loss == 'euclidean':
            self.cost = self.sum_euclidean_cost
            self.mean_cost = self.mean_euclidean_cost
        elif loss == 'norm_euclidean':
            self.cost = self.sum_normalized_euclidean_cost
            self.mean_cost = self.mean_normalized_euclidean_cost
        elif loss == 'hellinger':
            self.cost = self.sum_hellinger_cost
            self.mean_cost = self.mean_hellinger_cost
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
        batch_y = T.ivector('batch_y')
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
            theano.Param(batch_x2), theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self, debug=False):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        cost = self.cost_training
        gparams = T.grad(cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            #gp = gparam.clip(-1./self._eps, 1./self._eps)
            #agrad = self._rho * accugrad + (1 - self._rho) * gp * gp
            #dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gp
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
                    [updates[param] for param in self.params]# +\
                    #[self.x1] +\
                    #[self.x2] +\
                    #[self.y] +\
                    #[self.cost]

        train_fn = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y)],
                outputs=self.cost,
                givens={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(x[0], x[1], y) for (x, y) in given_set]

        return scoref

    def score_classif_same_diff_separated(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y = T.ivector('batch_y')
        #cost_same = T.mean(self.normalized_euclidean[T.eq(self.y, 1).nonzero()], axis=-1)
        #cost_diff = T.mean(1. - self.normalized_euclidean[T.eq(self.y, 0).nonzero()], axis=-1)
        cost_same = T.mean(self.cos_sim[T.eq(self.y, 1).nonzero()], axis=-1)
        #cost_diff = T.mean(1. - self.cos_sim[T.eq(self.y, 0).nonzero()], axis=-1)
        cost_diff = T.mean(self.cos_sim[T.eq(self.y, 0).nonzero()], axis=-1)
        score = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y)],
                outputs=[cost_same, cost_diff],
                #outputs=self.cost,
                givens={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(x[0], x[1], y) for (x, y) in given_set]

        return scoref

    def transform_x1_x2(self):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        transform = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2)],
                outputs=[self.layers[-2].output, self.layers[-1].output],
                givens={self.x1: batch_x1, self.x2: batch_x2})
        return transform

    def transform_x1(self):
        batch_x1 = T.fmatrix('batch_x1')
        transform = theano.function(inputs=[theano.Param(batch_x1)],
                outputs=self.layers[-2].output,
                givens={self.x1: batch_x1})
        return transform


class ABClustNeuralNet(ABNeuralNet):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[ReLU, ReLU],
            layers_sizes=[200],
            n_outs=100,
            loss='cos_cos2',
            rho=0.9, eps=1.E-6,
            max_norm=0.,
            debugprint=False):
        super(ABClustNeuralNet, self).__init__(numpy_rng, theano_rng,
                n_ins, layers_types, layers_sizes, n_outs, loss, rho, eps,
                max_norm, debugprint)
        # TODO


class DropoutABNeuralNet(ABNeuralNet):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
            layers_sizes=[1024, 1024, 1024, 1024],
            dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
            n_outs=62 * 3,
            loss='cos_cos2',
            rho=0.95, eps=1.E-6,
            max_norm=0.,
            fast_drop=False,
            debugprint=False):
        super(DropoutABNeuralNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, loss,
                rho, eps, max_norm, debugprint)

        self.dropout_rates = dropout_rates
        if fast_drop:
            if dropout_rates[0]:
                dropout_layer_input1 = fast_dropout(numpy_rng, self.x1,
                        dropout_rates[0])
                dropout_layer_input2 = fast_dropout(numpy_rng, self.x2,
                        dropout_rates[0])
            else:
                dropout_layer_input1 = self.x1
                dropout_layer_input2 = self.x2
        else:
            dropout_layer_input1 = dropout(numpy_rng, self.x1,
                    p=dropout_rates[0])
            dropout_layer_input2 = dropout(numpy_rng, self.x2,
                    p=dropout_rates[0])
        self.dropout_layers1 = []
        self.dropout_layers2 = []

        for layer, layer_type, n_in, n_out, dr in zip(self.layers[::2],
                layers_types, self.layers_ins, self.layers_outs,
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything 
                                            # from the last layer !!!
            if dr:
                if fast_drop:
                    this_layer1 = layer_type(rng=numpy_rng,
                            input=dropout_layer_input1, n_in=n_in, n_out=n_out,
                            W=layer.W, b=layer.b, fdrop=dr)
                    assert hasattr(this_layer1, 'output')
                    self.dropout_layers1.append(this_layer1)
                    dropout_layer_input1 = this_layer1.output
                    this_layer2 = layer_type(rng=numpy_rng,
                            input=dropout_layer_input2, n_in=n_in, n_out=n_out,
                            W=layer.W, b=layer.b, fdrop=dr)
                    assert hasattr(this_layer2, 'output')
                    self.dropout_layers2.append(this_layer2)
                    dropout_layer_input2 = this_layer2.output
                else:
                    this_layer1 = layer_type(rng=numpy_rng,
                            input=dropout_layer_input1, n_in=n_in, n_out=n_out,
                            W=layer.W * 1. / (1. - dr), # experimental
                            b=layer.b * 1. / (1. - dr)) # TODO check
                    assert hasattr(this_layer1, 'output')
                    # N.B. dropout with dr=1 does not dropanything!!
                    this_layer1.output = dropout(numpy_rng, this_layer1.output, dr)
                    self.dropout_layers1.append(this_layer1)
                    dropout_layer_input1 = this_layer1.output
                    this_layer2 = layer_type(rng=numpy_rng,
                            input=dropout_layer_input2, n_in=n_in, n_out=n_out,
                            W=layer.W * 1. / (1. - dr), # experimental
                            b=layer.b * 1. / (1. - dr)) # TODO check
                    assert hasattr(this_layer2, 'output')
                    this_layer2.output = dropout(numpy_rng, this_layer2.output, dr)
                    self.dropout_layers2.append(this_layer2)
                    dropout_layer_input2 = this_layer2.output
            else:
                this_layer1 = layer_type(rng=numpy_rng,
                        input=dropout_layer_input1, n_in=n_in, n_out=n_out,
                        W=layer.W, b=layer.b)
                assert hasattr(this_layer1, 'output')
                self.dropout_layers1.append(this_layer1)
                dropout_layer_input1 = this_layer1.output
                this_layer2 = layer_type(rng=numpy_rng,
                        input=dropout_layer_input2, n_in=n_in, n_out=n_out,
                        W=layer.W, b=layer.b)
                assert hasattr(this_layer2, 'output')
                self.dropout_layers2.append(this_layer2)
                dropout_layer_input2 = this_layer2.output

        L2 = 0.
        for param in self.params:
            L2 += T.sum(param ** 2)
        L1 = 0.
        for param in self.params:
            L1 += T.sum(abs(param))

        self.squared_error_training = (dropout_layer_input1 - dropout_layer_input2).norm(2, axis=-1) **2
        self.mse_training = T.mean(self.squared_error_training, axis=-1)
        self.rmse_training = T.sqrt(self.mse_training)
        self.sse_training = T.sum(self.squared_error_training, axis=-1)
        self.rsse_training = T.sqrt(self.sse_training)

        self.rsse_cost_training = T.switch(self.y, self.rsse_training, -self.rsse_training)
        self.rmse_cost_training = T.switch(self.y, self.rmse_training, -self.rmse_training)
        self.sum_rmse_costs_training = T.sum(self.rmse_cost_training)
        self.sum_rsse_costs_training = T.sum(self.rsse_cost_training)
        self.mean_rmse_costs_training = T.mean(self.rmse_cost_training)
        self.mean_rsse_costs_training = T.mean(self.rsse_cost_training)

        self.cos_sim_training = (T.sum(dropout_layer_input1 * dropout_layer_input2, axis=-1) /
            (dropout_layer_input1.norm(2, axis=-1) * dropout_layer_input2.norm(2, axis=-1)))
        self.cos_sim_cost_training = T.switch(self.y, 1.-self.cos_sim_training, self.cos_sim_training)
        self.mean_cos_sim_cost_training = T.mean(self.cos_sim_cost_training)
        self.sum_cos_sim_cost_training = T.sum(self.cos_sim_cost_training)

        self.cos2_sim_cost_training = T.switch(self.y, 1.-(self.cos_sim_training ** 2), self.cos_sim_training ** 2)
        self.mean_cos2_sim_cost_training = T.mean(self.cos2_sim_cost_training)
        self.sum_cos2_sim_cost_training = T.sum(self.cos2_sim_cost_training)

        self.cos_cos2_sim_cost_training = T.switch(self.y, (1.-self.cos_sim_training)/2, self.cos_sim_training ** 2)
        self.mean_cos_cos2_sim_cost_training = T.mean(self.cos_cos2_sim_cost_training)
        self.sum_cos_cos2_sim_cost_training = T.sum(self.cos_cos2_sim_cost_training)

        self.normalized_euclidean_training = ((dropout_layer_input1 - dropout_layer_input2).norm(2, axis=-1) / (dropout_layer_input1.norm(2, axis=-1) * dropout_layer_input2.norm(2, axis=-1)))
        self.normalized_euclidean_cost_training = T.switch(self.y, self.normalized_euclidean_training, -self.normalized_euclidean_training)
        self.mean_normalized_euclidean_cost_training = T.mean(self.normalized_euclidean_cost_training)
        self.sum_normalized_euclidean_cost_training = T.sum(self.normalized_euclidean_cost_training)

        self.hellinger_training = 0.5 * T.sqrt(T.sum((T.sqrt(dropout_layer_input1) - T.sqrt(dropout_layer_input2))**2, axis=1))
        self.hellinger_cost_training = T.switch(self.y, self.hellinger_training, 1.-self.hellinger_training)
        self.mean_hellinger_cost_training = T.mean(self.hellinger_cost_training)
        self.sum_hellinger_cost_training = T.sum(self.hellinger_cost_training)

        if loss == 'cos_cos2':
            self.cost_training = self.sum_cos_cos2_sim_cost_training
            self.mean_cost_training = self.mean_cos_cos2_sim_cost_training
        elif loss == 'cos':
            self.cost_training = self.sum_cos_sim_cost_training
            self.mean_cost_training = self.mean_cos_sim_cost_training
        elif loss == 'cos2':
            self.cost_training = self.sum_cos2_sim_cost_training
            self.mean_cost_training = self.mean_cos_sim_cost_training
        elif loss == 'norm_euclidean':
            self.cost_training = self.sum_normalized_euclidean_cost_training
            self.mean_cost_training = self.mean_normalized_euclidean_cost_training
        elif loss == 'hellinger':
            self.cost_training = self.sum_hellinger_cost_training
            self.mean_cost_training = self.mean_hellinger_cost_training

        if debugprint:
            theano.printing.debugprint(self.cost_training)



class ABNeuralNet2Outputs(object):  #NeuralNet):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[ReLU, ReLU, ReLU, ReLU, ReLU],
            layers_sizes=[1024, 1024, 1024, 1024],
            n_outs=100,
            loss='cos_cos2',
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
        self.y1 = T.ivector('y1')
        self.y2 = T.ivector('y2')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        layer_input1 = self.x1
        layer_input2 = self.x2
        layer_input3 = None
        layer_input4 = None
        
        for layer_ind, (layer_type, n_in, n_out) in enumerate(
                zip(layers_types, self.layers_ins, self.layers_outs)):
            this_layer1 = layer_type(rng=numpy_rng,
                    input=layer_input1, n_in=n_in, n_out=n_out)#, cap=6.)
            assert hasattr(this_layer1, 'output')
            if layer_ind == len(layers_types)-1:
                this_layer3 = layer_type(rng=numpy_rng,
                        input=layer_input1, n_in=n_in, n_out=n_out)#, cap=6.)
                layer_input3 = this_layer3.output
            layer_input1 = this_layer1.output
            self.params.extend(this_layer1.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer1.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer1.params])
            self.layers.append(this_layer1)
            if layer_ind == len(layers_types)-1:
                self.params.extend(this_layer3.params)
                self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                    'accugrad') for t in this_layer3.params])
                self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                    'accudelta') for t in this_layer3.params])
            this_layer2 = layer_type(rng=numpy_rng,
                    input=layer_input2, n_in=n_in, n_out=n_out,
                    W=this_layer1.W, b=this_layer1.b)#, cap=6.)
            assert hasattr(this_layer2, 'output')
            if layer_ind == len(layers_types)-1:
                this_layer4 = layer_type(rng=numpy_rng,
                        input=layer_input2, n_in=n_in, n_out=n_out,
                        W=this_layer3.W, b=this_layer3.b)#, cap=6.)
                layer_input4 = this_layer4.output
                assert hasattr(this_layer4, 'output')
            layer_input2 = this_layer2.output
            self.layers.append(this_layer2)
            if layer_ind == len(layers_types)-1:
                self.layers.append(this_layer3)
                self.layers.append(this_layer4)

        L2 = 0.
        for param in self.params:
            L2 += T.sum(param ** 2)
        L1 = 0.
        for param in self.params:
            L1 += T.sum(abs(param))

        self.squared_error1 = (layer_input1 - layer_input2).norm(2, axis=-1) **2
        self.squared_error2 = (layer_input3 - layer_input4).norm(2, axis=-1) **2

        self.mse1 = T.mean(self.squared_error1, axis=-1)
        self.rmse1 = T.sqrt(self.mse1)
        self.sse1 = T.sum(self.squared_error1, axis=-1)
        self.rsse1 = T.sqrt(self.sse1)
        self.mse2 = T.mean(self.squared_error2, axis=-1)
        self.rmse2 = T.sqrt(self.mse2)
        self.sse2 = T.sum(self.squared_error2, axis=-1)
        self.rsse2 = T.sqrt(self.sse2)

        self.rsse_cost = T.switch(self.y1, self.rsse1, -self.rsse1) + T.switch(self.y2, self.rsse2, -self.rsse2)
        self.rmse_cost = T.switch(self.y1, self.rmse1, -self.rmse1) + T.switch(self.y2, self.rmse2, -self.rmse2)
        self.sum_rmse_costs = T.sum(self.rmse_cost)
        self.sum_rsse_costs = T.sum(self.rsse_cost)
        self.mean_rmse_costs = T.mean(self.rmse_cost)
        self.mean_rsse_costs = T.mean(self.rsse_cost)

        self.cos_sim1 = (T.sum(layer_input1 * layer_input2, axis=-1) /
            (layer_input1.norm(2, axis=-1) * layer_input2.norm(2, axis=-1)))
        self.cos_sim2 = (T.sum(layer_input3 * layer_input4, axis=-1) /
            (layer_input3.norm(2, axis=-1) * layer_input4.norm(2, axis=-1)))
        self.cos_sim_cost = T.switch(self.y1, 1.-self.cos_sim1, self.cos_sim1) + T.switch(self.y2, 1.-self.cos_sim2, self.cos_sim2)

        self.mean_cos_sim_cost = T.mean(self.cos_sim_cost)
        self.sum_cos_sim_cost = T.sum(self.cos_sim_cost)

        self.cos2_sim_cost = T.switch(self.y1, 1.-(self.cos_sim1 ** 2), self.cos_sim1 ** 2) +  T.switch(self.y2, 1.-(self.cos_sim2 ** 2), self.cos_sim2 ** 2)

        self.mean_cos2_sim_cost = T.mean(self.cos2_sim_cost)
        self.sum_cos2_sim_cost = T.sum(self.cos2_sim_cost)

        self.cos_cos2_sim_cost = T.switch(self.y1, (1.-self.cos_sim1)/2, self.cos_sim1 ** 2) + T.switch(self.y2, (1.-self.cos_sim2)/2, self.cos_sim2 ** 2)
        self.cos_cos2_sim_cost_s = 0*T.switch(self.y1, (1.-self.cos_sim1)/2, self.cos_sim1 ** 2) + T.switch(self.y2, (1.-self.cos_sim2)/2, self.cos_sim2 ** 2)  # just spkrs
        self.cos_cos2_sim_cost_w = T.switch(self.y1, (1.-self.cos_sim1)/2, self.cos_sim1 ** 2) + 0*T.switch(self.y2, (1.-self.cos_sim2)/2, self.cos_sim2 ** 2)  # just words

        self.mean_cos_cos2_sim_cost = T.mean(self.cos_cos2_sim_cost)
        self.sum_cos_cos2_sim_cost = T.sum(self.cos_cos2_sim_cost)

        from layers import relu_f
        self.dot_prod1 = T.batched_dot(layer_input1, layer_input2) # TODO
        self.dot_prod2 = T.batched_dot(layer_input3, layer_input4) # TODO
        self.dot_prod_cost = relu_f(T.switch(self.y1, 1.-self.dot_prod1, self.dot_prod1)) + relu_f(T.switch(self.y2, 1.-self.dot_prod2, self.dot_prod2)) # TODO
        self.mean_dot_prod_cost = T.mean(self.dot_prod_cost) # TODO
        self.sum_dot_prod_cost = T.sum(self.dot_prod_cost) # TODO

        # TODO T.arccos based loss (problem of gradients at extrumums?)

        self.euclidean1 = (layer_input1 - layer_input2).norm(2, axis=0)
        self.euclidean2 = (layer_input3 - layer_input4).norm(2, axis=-1)
        self.euclidean_cost = T.switch(self.y1, self.euclidean1, -self.euclidean1) + T.switch(self.y2, self.euclidean2, -self.euclidean2)

        self.mean_euclidean_cost = T.mean(self.euclidean_cost)
        self.sum_euclidean_cost = T.sum(self.euclidean_cost)

        self.hellinger1 = 0.5 * T.sqrt(T.sum((T.sqrt(layer_input1) - T.sqrt(layer_input2))**2, axis=1))
        self.hellinger2 = 0.5 * T.sqrt(T.sum((T.sqrt(layer_input3) - T.sqrt(layer_input4))**2, axis=1))
        self.hellinger_cost = T.switch(self.y1, self.hellinger1, 1.-self.hellinger1) + T.switch(self.y2, self.hellinger2, 1.-self.hellinger2)

        self.mean_hellinger_cost = T.mean(self.hellinger_cost)
        self.sum_hellinger_cost = T.sum(self.hellinger_cost)

        if loss == 'cos_cos2':
            self.cost = self.sum_cos_cos2_sim_cost
            self.mean_cost = self.mean_cos_cos2_sim_cost
        elif loss == 'cos_cos2_w':
            self.cost = T.sum(self.cos_cos2_sim_cost_w)
            self.mean_cost = T.mean(self.cos_cos2_sim_cost_w)
        elif loss == 'cos_cos2_s':
            self.cost = T.sum(self.cos_cos2_sim_cost_s)
            self.mean_cost = T.mean(self.cos_cos2_sim_cost_s)
        elif loss == 'cos':
            self.cost = self.sum_cos_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        elif loss == 'cos2':
            self.cost = self.sum_cos2_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        elif loss == 'dot_prod':
            self.cost = self.sum_dot_prod_cost
            self.mean_cost = self.mean_dot_prod_cost
        elif loss == 'asym_eucl':
            self.cost = self.sum_asym_eucl_cost
            self.mean_cost = self.mean_asym_eucl_cost
        elif loss == 'euclidean':
            self.cost = self.sum_euclidean_cost
            self.mean_cost = self.mean_euclidean_cost
        elif loss == 'hellinger':
            self.cost = self.sum_hellinger_cost
            self.mean_cost = self.mean_hellinger_cost
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
        batch_y1 = T.ivector('batch_y1')
        batch_y2 = T.ivector('batch_y2')
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
            theano.Param(batch_x2), theano.Param(batch_y1),
            theano.Param(batch_y2),
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2,
                self.y1: batch_y1, self.y2: batch_y2})

        return train_fn

    def get_adagrad_trainer(self, debug=False):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y1 = T.ivector('batch_y1')
        batch_y2 = T.ivector('batch_y2')
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
                    [updates[param] for param in self.params]# +\
                    #[self.x1] +\
                    #[self.x2] +\
                    #[self.y] +\
                    #[self.cost]

        train_fn = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y1),
            theano.Param(batch_y2), theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2,
                self.y1: batch_y1, self.y2: batch_y2})

        return train_fn

    def get_adadelta_trainer(self, debug=False):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y1 = T.ivector('batch_y1')
        batch_y2 = T.ivector('batch_y2')
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
                    [updates[param] for param in self.params]# +\
                    #[self.x1] +\
                    #[self.x2] +\
                    #[self.y] +\
                    #[self.cost]

        train_fn = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y1),
            theano.Param(batch_y2)],
            outputs=outputs,
            updates=updates,
            givens={self.x1: batch_x1, self.x2: batch_x2,
                self.y1: batch_y1, self.y2: batch_y2})

        return train_fn

    def score_classif(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y1 = T.ivector('batch_y1')
        batch_y2 = T.ivector('batch_y2')
        score = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2), theano.Param(batch_y1),
            theano.Param(batch_y2)],
                outputs=self.cost,
                givens={self.x1: batch_x1, self.x2: batch_x2,
                    self.y1: batch_y1, self.y2: batch_y2})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(x[0], x[1], y[0], y[1]) for (x, y) in given_set]

        return scoref

    def score_classif_same_diff_word_separated(self, given_set):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        batch_y1 = T.ivector('batch_y1')
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

    def transform_x1_x2(self):
        batch_x1 = T.fmatrix('batch_x1')
        batch_x2 = T.fmatrix('batch_x2')
        transform = theano.function(inputs=[theano.Param(batch_x1), 
            theano.Param(batch_x2)],
                outputs=[self.layers[-4].output, self.layers[-3].output,
                    self.layers[-2].output, self.layers[-1].output],
                givens={self.x1: batch_x1, self.x2: batch_x2})
        return transform

    def transform_x1(self):
        batch_x1 = T.fmatrix('batch_x1')
        transform = theano.function(inputs=[theano.Param(batch_x1)],
                outputs=[self.layers[-4].output, self.layers[-2].output],
                givens={self.x1: batch_x1})
        return transform


