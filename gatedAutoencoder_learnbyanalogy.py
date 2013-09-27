import numpy, pylab
import cPickle

import theano
import theano.tensor as T
import theano.tensor.signal.conv 
from theano.tensor.shared_randomstreams import RandomStreams

SMALL = 0.000001


class FactoredGatedAutoencoder(object):
    def __init__(self, numvisX, numvisY, numfac, nummap, output_type, 
                 corruption_type='zeromask', corruption_level=0.0, 
                 weight_decay_vis=0.0, weight_decay_map=0.0, 
                 numpy_rng=None, theano_rng=None):
        self.numvisX = numvisX
        self.numvisY = numvisY
        self.numfac  = numfac
        self.nummap  = nummap
        self.output_type  = output_type
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.weight_decay_vis = weight_decay_vis
        self.weight_decay_map = weight_decay_map
        self.inputs = T.matrix(name='inputs') 

        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)

        if not theano_rng:  
            theano_rng = RandomStreams(1)

        wxf_init = numpy_rng.normal(size=(numvisX, numfac)).astype(theano.config.floatX)*0.01
        wyf_init = numpy_rng.normal(size=(numvisY, numfac)).astype(theano.config.floatX)*0.01

        self.whf_init = numpy.exp(numpy_rng.uniform(low=-3.0, high=-2.0, size=(nummap, numfac)).astype(theano.config.floatX))
        self.whf_in_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(nummap, numfac)).astype(theano.config.floatX)
        self.whf = theano.shared(value = self.whf_init, name='whf')
        self.whf_in = theano.shared(value = self.whf_in_init, name='whf_in')
        self.wxf = theano.shared(value = wxf_init, name = 'wxf')
        self.bvisX = theano.shared(value = numpy.zeros(numvisX, dtype=theano.config.floatX), name='bvisX')
        self.wyf = theano.shared(value = wyf_init, name = 'wyf')
        self.bvisY = theano.shared(value = numpy.zeros(numvisY, dtype=theano.config.floatX), name='bvisY')
        self.bmap = theano.shared(value = 0.0*numpy.ones(nummap, dtype=theano.config.floatX), name='bmap')
        self.params = [self.wxf, self.wyf, self.whf_in, self.whf, self.bmap, self.bvisX, self.bvisY]

        # DEFINE THE LAYER FUNCTION 
        self.inputsX = self.inputs[:, :numvisX]
        self.inputsY = self.inputs[:, numvisX:2*numvisX]
        self.targetsX = self.inputs[:, 2*numvisX:3*numvisX]
        self.targetsY = self.inputs[:, 3*numvisX:]
        if self.corruption_level > 0.0: 
            if self.corruption_type=='zeromask':
                self._corruptedinputsX = theano_rng.binomial(size=self.inputsX.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsX
                self._corruptedinputsY = theano_rng.binomial(size=self.inputsY.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsY
                self._corruptedtargetsX = theano_rng.binomial(size=self.targetsX.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsX
                self._corruptedtargetsY = theano_rng.binomial(size=self.targetsY.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsY
            elif self.corruption_type=='gaussian':
                self._corruptedinputsX = theano_rng.normal(size=self.inputsX.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsX
                self._corruptedinputsY = theano_rng.normal(size=self.inputsY.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsY
                self._corruptedtargetsX = theano_rng.normal(size=self.targetsX.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.targetsX
                self._corruptedtargetsY = theano_rng.normal(size=self.targetsY.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.targetsY
            elif self.corruption_type=='none':
                self._corruptedinputsX = self.inputsX
                self._corruptedinputsY = self.inputsY
                self._corruptedtargetsX = self.targetsX
                self._corruptedtargetsY = self.targetsY
            else:
                assert False, "unknown corruption type"
        else:
            self._corruptedinputsX = self.inputsX
            self._corruptedinputsY = self.inputsY
            self._corruptedtargetsX = self.targetsX
            self._corruptedtargetsY = self.targetsY

        self._inputfactorsX = T.dot(self._corruptedinputsX, self.wxf)
        self._inputfactorsY = T.dot(self._corruptedinputsY, self.wyf)
        self._targetfactorsX = T.dot(self._corruptedtargetsX, self.wxf)
        self._targetfactorsY = T.dot(self._corruptedtargetsY, self.wyf)
        self._inputfactorsXNonoise = T.dot(self.inputsX, self.wxf)
        self._inputfactorsYNonoise = T.dot(self.inputsY, self.wyf)
        self._mappings = T.nnet.sigmoid(T.dot(self._inputfactorsX*self._inputfactorsY, self.whf_in.T)+self.bmap)
        self._mappingsNonoise = T.nnet.sigmoid(T.dot(self._inputfactorsXNonoise*self._inputfactorsYNonoise, self.whf_in.T)+self.bmap)
        self._factorsH = T.dot(self._mappings, self.whf)
        self._outputX_acts = T.dot(self._targetfactorsY*self._factorsH, self.wxf.T)+self.bvisX
        self._outputY_acts = T.dot(self._targetfactorsX*self._factorsH, self.wyf.T)+self.bvisY
        if self.output_type == 'binary':
            self._reconsX = T.nnet.sigmoid(self._outputX_acts)
            self._reconsY = T.nnet.sigmoid(self._outputY_acts)
        elif self.output_type == 'real':
            self._reconsX = self._outputX_acts
            self._reconsY = self._outputY_acts
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        self._weightcost = self.weight_decay_vis*( (self.wxf**2).sum() + (self.wyf**2).sum()) \
                         + self.weight_decay_map*(self.whf**2).sum() \
                         + self.weight_decay_map*(self.whf_in**2).sum()

        self.corruptedinputsX = theano.function([self.inputs], self._corruptedinputsX)
        self.corruptedinputsY = theano.function([self.inputs], self._corruptedinputsY)
        self.mappings = theano.function([self.inputs], self._mappings)
        self.mappingsNonoise = theano.function([self.inputs], self._mappingsNonoise)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)
        if self.output_type == 'binary':
            self._costpercase = - T.sum(
                  0.5* (self.targetsY*T.log(self._reconsY) + (1.0-self.targetsY)*T.log(1.0-self._reconsY))
                  +0.5* (self.targetsX*T.log(self._reconsX) + (1.0-self.targetsX)*T.log(1.0-self._reconsX)),
                                   axis=1)
        elif self.output_type == 'real':
            self._costpercase = T.sum(0.5*((self.targetsX-self._reconsX)**2)
                                     +0.5*((self.targetsY-self._reconsY)**2), axis=1)
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        self._weightcost = self._weightcost
        self._cost = T.mean(self._costpercase) + self._weightcost
        self._cost_pure = T.mean(self._costpercase) 
        self._grads = T.grad(self._cost, self.params)

        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER 
        self.mappings = theano.function([self.inputs], self._mappingsNonoise)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)
        self.cost = theano.function([self.inputs], self._cost)
        self.cost_pure = theano.function([self.inputs], self._cost_pure)
        self.grads = theano.function([self.inputs], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def normalizefilters(self, center=True):
        def inplacemult(x, v):
            x[:, :] *= v
            return x
        def inplacesubtract(x, v):
            x[:, :] -= v
            return x
        nwxf = (self.wxf.get_value().std(0)+SMALL)[numpy.newaxis, :]
        nwyf = (self.wyf.get_value().std(0)+SMALL)[numpy.newaxis, :]
        meannxf = nwxf.mean()
        meannyf = nwyf.mean()
        wxf = self.wxf.get_value(borrow=True)
        wyf = self.wyf.get_value(borrow=True)
        # CENTER FILTERS 
        if center:
            self.wxf.set_value(inplacesubtract(wxf, wxf.mean(0)[numpy.newaxis,:]), borrow=True)
            self.wyf.set_value(inplacesubtract(wyf, wyf.mean(0)[numpy.newaxis,:]), borrow=True)
        # FIX STANDARD DEVIATION 
        self.wxf.set_value(inplacemult(wxf, meannxf/nwxf),borrow=True)
        self.wyf.set_value(inplacemult(wyf, meannyf/nwyf),borrow=True)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))


