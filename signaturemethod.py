# path is a smooth mapping from a time interval into a multidimensional space
# => continuous stream of data
# signature is a 'canonical' transform of such a data stream into a high dim 
# feature space

# use function stream2sig to convert numpy arrays of shape l,n (l = number of 
# sample points, n = dim of target space)
import esig as es
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1)
two_dim_stream = np.random.random(size = (10,2))
two_dim_stream[0,0]

# calculating signature of degree 2
two_dim_sig = es.tosig.stream2sig(two_dim_stream, 2)
two_dim_sig

# dta = np.array([(1,3,5,8), (1,4,2,6)])
x = np.array([1,3,5,8])
y = np.array([1,4,2,6])

fig = plt.figure()
plt.plot(x,y,'o-')
fig.suptitle('test title', fontsize=20)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=16)
# fig.savefig('test.jpg')

# The dots are discrete data points and the solid line is a path continuously 
# connecting the data points. In fact, we took two 1-dimensional sequences and
# embedded into a single (1-dim) path in 2-dimensions. Generalising the idea, 
# any collection of d 1-dim paths can be embedded into a single path in d-dimensions.
# https://github.com/kormilitzin/the-signature-method-in-machine-learning

