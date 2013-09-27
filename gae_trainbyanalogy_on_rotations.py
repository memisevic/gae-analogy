import os
HOME = os.environ['HOME']

import pylab
import numpy
import numpy.random
import gatedAutoencoder_learnbyanalogy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


numpy_rng  = numpy.random.RandomState(1)


def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    from pylab import cm, ceil
    numimages = M.shape[1]
    if layout is None:
        n0 = int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * numpy.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = numpy.vstack((
                            numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*numpy.ones((height,border),dtype=float))),
                            bordercolor*numpy.ones((border,width+border),dtype=float)
                            ))
    pylab.imshow(im, cmap=cm.gray, interpolation='nearest', **kwargs)
    pylab.axis('off')
    pylab.show()


class GraddescentMinibatch(object):

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self, numbatches):
        def inplaceclip(x):
            x[:,:] *= x>0.0
            return x

        def inplacemask(x, mask):
            x[:,:] *= mask
            return x

        cost = 0.0
        stepcount = 0.0
        for i, batch_index in enumerate(self.rng.permutation(self.numbatches-1)):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            self.model.normalizefilters()
            if i > numbatches: 
                break

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)


print '... loading data'

patchsize = 13

train_features_numpy = numpy.load('./rotation_analogies13x13_numcases100000.npy').astype(theano.config.floatX)
train_features_numpy -= train_features_numpy.mean(2)[:, :, None]
train_features_numpy /= train_features_numpy.std(2)[:, :, None] + 0.1 * train_features_numpy.std()
train_features_numpy = train_features_numpy.reshape(-1, 4*patchsize**2)
train_features = theano.shared(train_features_numpy)

print '... done'
numvisX = patchsize*patchsize
numvisY = patchsize*patchsize


# INSTANTIATE MODEL
print '... instantiating model'
theano_rng = RandomStreams(1)
model = gatedAutoencoder_learnbyanalogy.FactoredGatedAutoencoder(numvisX=numvisX, numvisY=numvisY, 
                                                  numfac=500, nummap=50, output_type='real', 
                                                  corruption_type='zeromask', corruption_level=0.0,
                                                  numpy_rng=numpy_rng, theano_rng=theano_rng)

print '... done'


def show_examples(analogies, model):
    while True:
        pylab.subplot(1,2,1)
        dispims(analogies[:100, :patchsize**2].T, 13, 13, 1)
        pylab.draw()
        pylab.subplot(1,2,2)
        dispims(analogies[:100, :patchsize**2].T, 13, 13, 1)
        ginput()
        pylab.subplot(1,2,1)
        dispims(analogies[:100, patchsize**2:2*patchsize**2].T, 13, 13, 1)
        pylab.subplot(1,2,2)
        dispims(model.reconsY(analogies[:100]).T, 13, 13, 1)
        pylab.draw()
        ginput()


# TRAIN MODEL
numepochs = 100
trainer = GraddescentMinibatch(model, train_features, batchsize=100, learningrate=0.01)
pylab.ion()
batches_per_epoch = 400
for epoch in xrange(numepochs):
    dispims(model.wyf.get_value()[:169, :], 13, 13, 1, bordercolor=model.wyf.get_value().mean())
    pylab.draw()
    trainer.step(numbatches=batches_per_epoch)
    pylab.clf()



