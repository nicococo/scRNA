import numpy as np
import matplotlib.pyplot as plt
# Code and examples of Kernel Target Alignments (Christianini et al, NIPS 2001 and JMLR 2002).
# Author: Nico Goernitz, TU Berlin, 2016

from utils import *

if __name__ == "__main__":
    N = 100 # Number of datapoints
    D = 4  # Number of features

    # generate data
    X = np.random.randn(D, N)
    # ...and corresponding labels
    beta = 100.*np.random.randn(D,1)  # some random linear model that generates labels from data
    y = beta.T.dot(X) + 10.  # ..add an intercept to later show that centering is important
    y_bin = np.sign(y)  # and the binary labels

    # setup kernels (all linear, but other kernels are applicable too)
    K_lin = X.T.dot(X)
    YY = y.reshape((N,1)).dot(y.reshape((1,N)))
    YY_bin =  y_bin.reshape((N, 1)).dot(y_bin.reshape((1, N)))

    # plot the kernels
    # Show that the higher the dimensionality D, the less obvious the label-data-correspondance
    # and therefore, kta is decreasing with increasing number of features D
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.pcolor(K_lin)
    plt.subplot(1, 2, 2)
    plt.pcolor(YY)
    plt.show()

    print 'Some tests to highlight common mistakes/behavior (#pos/#neg = {0}/{1}).'.format(np.sum(y_bin==1), np.sum(y_bin==-1))
    print '--------------------------------------------------------------------------'

    print ''
    print '(A) Do both KTA methods deliver the same results when given same inputs? (Answer: Yes!)'
    print '  -K_lin. and y_bin: ', kta_align_general(K_lin, YY_bin), kta_align_binary(K_lin, y_bin)

    print ''
    print '(B) How do the cont. labels compare...'
    print '  -K_lin. and y: ', kta_align_general(K_lin, YY)

    print ''
    print '(C) Now, how does scaling affect the outcome? (Answer: None! The kernels are normalized in KTA.)'
    print '  -Scaled K_lin. and y_bin: ', kta_align_general(2.*K_lin, YY_bin), kta_align_binary(2.*K_lin, y_bin)

    print ''
    print '(D) We need to center the kernels (in feature space)!'
    K_lin = center_kernel(K_lin)
    YY = center_kernel(YY)
    YY_bin = center_kernel(YY_bin)
    print '  -K_lin. and y_bin: ', kta_align_general(K_lin, YY_bin), kta_align_binary(K_lin, y_bin)
    print '  -K_lin. and y: ', kta_align_general(K_lin, YY)

    # FAZIT:
    print ''
    print 'FAZIT: '
    print 'Use kta_align_general and center both kernels before.'
    K_lin = center_kernel(K_lin)
    YY = center_kernel(YY)
    print '  -K_lin. and y_bin: ', kta_align_general(K_lin, YY_bin)
    print '  -K_lin. and y: ', kta_align_general(K_lin, YY)
