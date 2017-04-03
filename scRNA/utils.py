import os
import numpy as np
import sklearn.metrics as metrics
import sc3_clustering_impl as sc

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
        print('Warning! Gene identifier file is not specified. Gene ids are now generated.')
    else:
        gene_ids = np.loadtxt(fgenes, delimiter='\t', dtype=np.str)
        print('Gene ids loaded for {0} genes.'.format(gene_ids.shape[0]))
        if not np.unique(gene_ids).shape[0] == gene_ids.shape[0]:
            print('Warning! Gene ids are supposed to be unique. '
                  'Only {0} of {1}  entries are unique.'.format(np.unique(gene_ids).shape[0], gene_ids.shape[0]))

    labels = None
    labels_2_ids = None
    if flabels is not None:
        print('Loading labels from \'{0}\'.'.format(flabels))
        # labels are handled as string values even though they are numerical
        label_ids = np.loadtxt(flabels, delimiter='\t', dtype=np.str_)
        assert label_ids.size == data.shape[1]

        labels_2_ids = np.unique(label_ids)
        unique_ind = np.arange(start=0, stop=labels_2_ids.shape[0])
        labels = np.zeros((data.shape[1]))
        print('Found {0} unique labels:'.format(labels_2_ids.size))
        print labels_2_ids
        for i in range(unique_ind.size):
            inds = np.where(label_ids == labels_2_ids[i])[0]
            labels[inds] = unique_ind[i]
            print('Label {0} occured {1} times. Assigned class is {2}.'.format(labels_2_ids[i], inds.size, unique_ind[i]))

    return data, gene_ids, labels, labels_2_ids


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
        C = b.dot(b.T)
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

    m = np.int(y.size)
    YY = y.reshape((m, 1)).dot(y.reshape((1, m)))
    return K.dot(YY).trace() / (m * np.sqrt(K.dot(K.T).trace()))


def get_kernel(X, Y, type='linear', param=1.0):
    """Calculates a kernel given the data X and Y (dims x exms)"""
    (Xdims, Xn) = X.shape
    (Ydims, Yn) = Y.shape

    kernel = 1.0
    if type=='linear':
        print('Calculating linear kernel with size {0}x{1}.'.format(Xn, Yn))
        kernel = X.T.dot(Y)

    if type=='rbf':
        print('Calculating Gaussian kernel with size {0}x{1} and sigma2={2}.'.format(Xn, Yn, param))
        Dx = (np.ones((Yn, 1)) * np.diag(X.T.dot(X)).reshape(1, Xn)).T
        Dy = (np.ones((Xn, 1)) * np.diag(Y.T.dot(Y)).reshape(1, Yn))
        kernel = Dx - 2.* np.array(X.T.dot(Y)) + Dy
        kernel = np.exp(-kernel / param)
        print kernel.shape
    return kernel


def unsupervised_acc_silhouette(X, labels, metric='euclidean'):
    dists = sc.distances(X, gene_ids=np.arange(X.shape[1]), metric=metric)
    num_lbls = np.unique(labels).size
    if num_lbls > 1:
        return metrics.silhouette_score(dists, labels, metric='precomputed')
    return 0.0


def unsupervised_acc_kta(X, labels, kernel='linear', param=1.0):
    Ky = np.zeros((labels.size, np.max(labels) + 1))
    for i in range(len(labels)):
        Ky[i, labels[i]] = 1.

    if kernel == 'rbf':
        Kx = get_kernel(X, X, type='rbf', param=param)
        Ky = get_kernel(Ky.T, Ky.T, type='rbf', param=param)
    else:
        Kx = X.T.dot(X)
        Ky = Ky.dot(Ky.T)

    Kx = center_kernel(Kx)
    Ky = center_kernel(Ky)
    Kx = normalize_kernel(Kx)
    Ky = normalize_kernel(Ky)
    return kta_align_general(Kx, Ky)


