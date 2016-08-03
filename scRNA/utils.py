import os
import numpy as np


def load_dataset_tsv(fname, fgenes=None, flabels=None):
    # check data filename
    if not os.path.exists(fname):
        raise StandardError('File \'{0}\' not found.'.format(fname))

    print('Loading TSV data file from {0}.'.format(fname))
    data = np.loadtxt(fname, delimiter='\t')
    print data.shape

    gene_ids = np.arange(0, data.shape[0]).astype(np.str)
    # Some scripts expect the gene ids (esp. for multitask learning of two or
    # more datasets). If not specified, inform the user.
    if fgenes is None:
        print('Warning! Gene identifier file is are specified. Gene ids are now generated.')
    else:
        gene_ids = np.loadtxt(fgenes, delimiter='\t', dtype=np.str)
        print('Gene ids loaded. There are ids for {0} genes.'.format(gene_ids.shape[0]))
        if not np.unique(gene_ids).shape[0] == gene_ids.shape[0]:
            print('Warning! Gene ids are supposed to be unique. '
                  'Only {0} of {1}  entries are unique.'.format(np.unique(gene_ids).shape[0], gene_ids.shape[0]))

    labels = None
    if flabels is not None:
        # TODO: Load TSV label file
        raise Exception('Not implemented yet.')

    return data, gene_ids, labels


def load_dataset(fname):
    if not os.path.exists(fname):
        raise StandardError('File \'{0}\' not found.'.format(fname))
    foo = np.load(fname)
    data  = foo['data']
    gene_ids = foo['transcripts']
    # look for labels
    labels = None
    if 'labels' in foo:
        labels = foo['labels']
    return data, gene_ids, labels


def normalize_kernel(K):
    # A kernel K is normalized, iff K_ii = 1 \forall i
    N = K.shape[0]
    a = np.sqrt(np.diag(K)).reshape((N, 1))
    if any(np.isnan(a)) or any(np.isinf(a)) or any(np.abs(a)<=1e-16):
        print 'Numerical instabilities.'
        C = np.eye(N)
    else:
        b = 1. / a
        C =  b.dot(b.T)
    return K * C


def center_kernel(K):
    # Mean free in feature space
    N = K.shape[0]
    a = np.ones((N, N)) / np.float(N)
    return K - a.dot(K) - K.dot(a) + a.dot(K.dot(a))


def kta_align_general(K1, K2):
    # Computes the (empirical) alignment of two kernels K1 and K2

    # Definition 1: (Empirical) Alignment
    #   a = <K1, K2>_Frob
    #   b = sqrt( <K1, K1> <K2, K2>)
    #   kta = a / b
    # with <A, B>_Frob = sum_ij A_ij B_ij = tr(AB')
    return K1.dot(K2.T).trace() / np.sqrt(K1.dot(K1.T).trace() * K2.dot(K2.T).trace())


def kta_align_binary(K, y):
    # Computes the (empirical) alignment of kernel K1 and
    # a corresponding binary label  vector y \in \{+1, -1\}^m

    m = np.float(y.size)
    YY = y.reshape((m, 1)).dot(y.reshape((1, m)))
    return K.dot(YY).trace() / (m * np.sqrt(K.dot(K.T).trace()))




