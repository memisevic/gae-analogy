import numpy
from scipy import ndimage

pi2 = numpy.pi * 2

numpy.random.seed(1)


numcases = 100000
patchsize = 13

baseims1 = numpy.random.randn(numcases, patchsize, patchsize)
baseims2 = numpy.random.randn(numcases, patchsize, patchsize)
analogies = numpy.zeros((numcases, 4, patchsize**2), dtype='float')

for i in range(numcases):
    print i
    _angle = 0.0
    angle_delta = (numpy.random.vonmises(0.0, 1.0)/numpy.pi) * 180
    for j in range(2):
        _angle += angle_delta
        analogies[i, j, :] = ndimage.rotate(baseims1[i], _angle, reshape=False, mode='wrap').flatten()
        analogies[i, j+2, :] = ndimage.rotate(baseims2[i], _angle, reshape=False, mode='wrap').flatten()

numpy.save('rotation_analogies'+str(patchsize)+'x'+str(patchsize)+'_numcases'+str(numcases)+'.npy', analogies)


