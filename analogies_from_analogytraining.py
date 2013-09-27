import pylab
import numpy
import numpy.random
import gatedAutoencoder_learnbyanalogy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)



def make_analogies(source_inputs_outputs, target_inputs, model):
    mappings = model.mappingsNonoise(source_inputs_outputs)
    factors_target = numpy.dot(target_inputs, model.wxf.get_value())
    factors_mappings = numpy.dot(mappings, model.whf.get_value())
    return numpy.dot((factors_target * factors_mappings), model.wyf.get_value().T) + model.bvisY.get_value()[None,:]


def show_analogies(model, analogy_data):
    sourceimages = analogy_data
    targetimages = numpy.vstack((numpy.zeros((4, patchsize), dtype=theano.config.floatX), 
                                 numpy.hstack((numpy.zeros((5, 4), dtype=theano.config.floatX), numpy.ones((5, 5), dtype=theano.config.floatX), numpy.zeros((5, 4), dtype=theano.config.floatX) )), numpy.zeros((4, patchsize), dtype=theano.config.floatX) )).reshape(1, patchsize**2).repeat(sourceimages.shape[0],axis=0)
    targetimages -= targetimages.mean(1)[:,None]
    targetimages /= targetimages.std(1)[:,None]

    #PLOT IN TWO FIGURES
    #pylab.figure(1)
    #pylab.subplot(1, 2, 1)
    #dispims(sourceimages[:,:patchsize**2].T, patchsize, patchsize, layout=(3,10), border=1)
    #pylab.subplot(1, 2, 2)
    #dispims(targetimages[:,:patchsize**2].T, patchsize, patchsize, layout=(3,10), border=1)
    #pylab.figure(2)
    #pylab.subplot(1, 2, 1)
    #dispims(sourceimages[:,patchsize**2:2*patchsize**2].T, patchsize, patchsize, layout=(3,10), border=1)
    #pylab.subplot(1, 2, 2)
    #dispims(make_analogies(sourceimages[:], targetimages[:, :patchsize**2], model).T, patchsize, patchsize, layout=(3,10), border=1) 

    #PLOT IN ONE FIGURE:
    clf()
    pylab.subplot(2, 2, 1)
    dispims(sourceimages[:,:patchsize**2].T, patchsize, patchsize, layout=(3,10), border=1)
    pylab.subplot(2, 2, 2)
    dispims(sourceimages[:,patchsize**2:2*patchsize**2].T, patchsize, patchsize, layout=(3,10), border=1)
    pylab.subplot(2, 2, 3)
    dispims(targetimages[:,:patchsize**2].T, patchsize, patchsize, layout=(3,10), border=1)
    pylab.subplot(2, 2, 4)
    dispims(make_analogies(sourceimages[:], targetimages[:, :patchsize**2], model).T, patchsize, patchsize, layout=(3,10), border=1) 





def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
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
    pylab.show()
    pylab.axis('off')


class GraddescentMinibatch(object):
    """ Gradient descent trainer class. """

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

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            self.model.normalizefilters()

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)


#MAKE DATA:
print 'making analogy data'
from scipy import ndimage 
numcases = 50000
horizon = 2
patchsize = 13
border = 5

baseims1 = numpy.random.rand(numcases, patchsize+2*border, patchsize+2*border)
baseims2 = numpy.random.rand(numcases, patchsize+2*border, patchsize+2*border)
analogy_data = numpy.zeros((numcases, 4, patchsize+2*border, patchsize+2*border), dtype='float32')
shift_deltas = numpy.random.rand(numcases, 2) * 6 - 3

for i in range(numcases):
    print "\b\b\b\b\b\b{0:6d}".format(i),
    shift = 0.0
    shift_delta = shift_deltas[i]
    for j in range(2):
        shift += shift_delta
        analogy_data[i, j, :, :] = ndimage.shift(baseims1[i], shift, mode='wrap')
        analogy_data[i, j+2, :, :] = ndimage.shift(baseims2[i], shift, mode='wrap')

analogy_data = analogy_data[:, :, border:-border, border:-border].reshape(numcases, 4, patchsize**2)

analogy_data -= analogy_data.mean(2)[:,:,None]
analogy_data /= analogy_data.std(2)[:,:,None] + analogy_data.std() * 0.1
analogy_data = analogy_data.reshape(-1, 4*patchsize**2)
train_features = theano.shared(analogy_data)
print '... done'


# INSTANTIATE MODEL
print '... instantiating model'
theano_rng = RandomStreams(1)
model = gatedAutoencoder_learnbyanalogy.FactoredGatedAutoencoder(numvisX=patchsize**2,
                                                                 numvisY=patchsize**2,
                                                                 numfac=200, nummap=50,
                                                                 output_type='real', 
                                                                 corruption_type='zeromask', 
                                                                 corruption_level=0.0, 
                                                                 numpy_rng=numpy_rng, 
                                                                 theano_rng=theano_rng)

print '... done'


# TRAIN MODEL
numepochs = 500
trainer = GraddescentMinibatch(model, train_features, batchsize=100, learningrate=0.01)
for epoch in xrange(numepochs):
    #pylab.subplot(1, 2, 1)
    #dispims(model.wxf.get_value(), patchsize, patchsize, 2)
    #pylab.subplot(1, 2, 2)
    #dispims(model.wyf.get_value(), patchsize, patchsize, 2)
    #pylab.show()
    show_analogies(model, analogy_data[:30])
    trainer.step()
    if epoch == 200: 
        trainer.set_learningrate(0.001)




