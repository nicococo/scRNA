import os
import numpy as np
import sklearn.metrics as metrics
import sklearn.decomposition as decomp
import pdb
import matplotlib.pyplot as plt

from . import sc3_clustering_impl as sc


def load_dataset_tsv(fname, fgenes=None, flabels=None):
    # check data filename
    if not os.path.exists(fname):
        raise Exception('File \'{0}\' not found.'.format(fname))

    #print('Loading TSV data file from {0}.'.format(fname))
    data = np.loadtxt(fname, delimiter='\t')
    #print data.shape

    gene_ids = np.arange(0, data.shape[0]).astype(np.str)
    # Some scripts expect the gene ids (esp. for multitask learning of two or
    # more datasets). If not specified, inform the user.
    if fgenes is None:
        print('Warning! Gene identifier file is not specified. Gene ids are now generated.')
    else:
        gene_ids = np.loadtxt(fgenes, delimiter='\t', dtype=np.str)
        #print('Gene ids loaded for {0} genes.'.format(gene_ids.shape[0]))
        if not np.unique(gene_ids).shape[0] == gene_ids.shape[0]:
            print(('Warning! Gene ids are supposed to be unique. '
                  'Only {0} of {1}  entries are unique.'.format(np.unique(gene_ids).shape[0], gene_ids.shape[0])))

    labels = None
    labels_2_ids = None
    if flabels is not None:
        #print('Loading labels from \'{0}\'.'.format(flabels))
        # labels are handled as string values even though they are numerical
        label_ids = np.loadtxt(flabels, delimiter='\t', dtype=np.str_)
        assert label_ids.size == data.shape[1]

        labels_2_ids = np.unique(label_ids)
        unique_ind = np.arange(start=0, stop=labels_2_ids.shape[0])
        labels = np.zeros((data.shape[1]), dtype=np.int)
        #print('Found {0} unique labels:'.format(labels_2_ids.size))
        #print labels_2_ids
        for i in range(unique_ind.size):
            inds = np.where(label_ids == labels_2_ids[i])[0]
            labels[inds] = unique_ind[i]
            #print('Label {0} occured {1} times. Assigned class is {2}.'.format(labels_2_ids[i], inds.size, unique_ind[i]))

    return data, gene_ids, labels, labels_2_ids


def load_dataset(fname):
    if not os.path.exists(fname):
        raise Exception('File \'{0}\' not found.'.format(fname))
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
        print('Numerical instabilities.')
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
    """
    Computes the (empirical) alignment of two kernels K1 and K2
    Definition 1: (Empirical) Alignment
      a = <K1, K2>_Frob
      b = sqrt( <K1, K1> <K2, K2>)
      kta = a / b
    with <A, B>_Frob = sum_ij A_ij B_ij = tr(AB')
    """
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
        #print('Calculating linear kernel with size {0}x{1}.'.format(Xn, Yn))
        kernel = X.T.dot(Y)

    if type=='rbf':
        #print('Calculating Gaussian kernel with size {0}x{1} and sigma2={2}.'.format(Xn, Yn, param))
        Dx = (np.ones((Yn, 1)) * np.diag(X.T.dot(X)).reshape(1, Xn)).T
        Dy = (np.ones((Xn, 1)) * np.diag(Y.T.dot(Y)).reshape(1, Yn))
        kernel = Dx - 2.* np.array(X.T.dot(Y)) + Dy
        kernel = np.exp(-kernel / param)
        #print kernel.shape
    return kernel


def unsupervised_acc_silhouette(X, labels, metric='euclidean'):
    dists = sc.distances(X, gene_ids=np.arange(X.shape[1]), metric=metric)
    num_lbls = np.unique(labels).size
    if num_lbls > 1 and not np.any(np.isnan(dists)) and not np.any(np.isinf(dists)):
        return metrics.silhouette_score(dists, labels, metric='precomputed')
    return 0.0


def unsupervised_acc_kta(X, labels, kernel='linear', param=1.0, center=True, normalize=True):
    Ky = np.zeros((labels.size, np.max(labels) + 1))
    for i in range(len(labels)):
        Ky[i, labels[i]] = 1.

    if kernel == 'rbf':
        Kx = get_kernel(X, X, type='rbf', param=param)
        Ky = get_kernel(Ky.T, Ky.T, type='linear', param=param)
    else:
        Kx = X.T.dot(X)
        Ky = Ky.dot(Ky.T)

    if center:
        Kx = center_kernel(Kx)
        Ky = center_kernel(Ky)
    if normalize:
        Kx = normalize_kernel(Kx)
        Ky = normalize_kernel(Ky)
    return kta_align_general(Kx, Ky)


def get_transferability_score(W, H, trg_data, reps=100, alpha=0.0, l1=0.75, max_iter=100, rel_err=1e-3):
    # estimate maximum error without any transfer
    errs = np.zeros((reps,))
    for i in range(errs.size):
        rand_gene_inds = np.random.permutation(W.shape[0])
        _, _, _, errs[i] = get_transferred_data_matrix(W[rand_gene_inds, :], trg_data, max_iter=max_iter, rel_err=rel_err)

    #print 'Calculating non-permuted error score'
    _, _, _, err_nonpermuted = get_transferred_data_matrix(W, trg_data, max_iter=max_iter, rel_err=rel_err)  # minimum transfer error

    nmf = decomp.NMF(alpha=alpha, init='nndsvdar', l1_ratio=l1, max_iter=max_iter,
                     n_components=W.shape[1], random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
    W_best = nmf.fit_transform(trg_data)
    H_best = nmf.components_

    err_best = np.sum(np.abs(trg_data - W_best.dot(H_best))) / np.float(trg_data.size)  # absolute
    err_curr = np.sum(np.abs(trg_data - W.dot(H))) / np.float(trg_data.size)  # absolute
    err_worst = np.max(errs)

    errs[errs < err_best] = err_best
    percs = 1.0 - (errs - err_best) / (err_worst - err_best)
    score = 1.0 - np.max([err_curr - err_best, 0]) / (err_worst - err_best)

    p_value = sum(errs < err_nonpermuted)/reps
    # plt.hist(errs)
    # plt.title("Histogram of random error scores")
    # plt.axvline(err_best, color='k', linestyle='dashed', linewidth=1)
    # plt.show()

    return score, percs, p_value


def get_transferred_data_matrix(W, trg_data, normalize_H2=False, max_iter=100, rel_err=1e-3):
    # initialize H: data matrix
    H = np.random.randn(W.shape[1], trg_data.shape[1])
    a1, a2 = np.where(H < 0.)
    H[a1, a2] *= -1.
    a1, a2 = np.where(H < 1e-10)
    H[a1, a2] = 1e-10

    n_iter = 0
    err = 1e10
    while n_iter < max_iter:
        n_iter += 1
        if np.any(W.T.dot(W.dot(H))==0.):
            raise Exception('DA target: division by zero.')
        H *= W.T.dot(trg_data) / W.T.dot(W.dot(H))
        new_err = np.sum(np.abs(trg_data - W.dot(H))) / np.float(trg_data.size)  # absolute
        # new_err = np.sqrt(np.sum((Xtrg - W.dot(H))*(Xtrg - W.dot(H)))) / np.float(Xtrg.size)  # frobenius
        if np.abs((err - new_err) / err) <= rel_err and err >= new_err:
            break
        err = new_err
    # print '  Number of iterations for reconstruction + reconstruction error    : ', n_iter, new_err
    H2 = np.zeros((W.shape[1], trg_data.shape[1]))

    H2[(np.argmax(H, axis=0), np.arange(trg_data.shape[1]))] = 1
    # H2[(np.argmax(H, axis=0), np.arange(trg_data.shape[1]))] = np.sum(H, axis=0)  # DOES NOT WORK WELL!

    # normalization
    if normalize_H2:
        #print 'Normalize H2.'
        n_iter = 0
        err = 1e10
        sparse_rec_err = np.sum(np.abs(trg_data - W.dot(H2))) / np.float(trg_data.size)  # absolute
        #print n_iter, ': sparse rec error: ', sparse_rec_err
        while n_iter < max_iter:
            n_iter += 1
            H2 *= W.T.dot(trg_data) / W.T.dot(W.dot(H2))
            # foo = 0.05 * W.T.dot(trg_data - W.dot(H2))
            # H2[np.argmax(H, axis=0), :] -= foo[np.argmax(H, axis=0), :]
            sparse_rec_err = np.sum(np.abs(trg_data - W.dot(H2))) / np.float(trg_data.size)  # absolute
            #print n_iter, ': sparse rec error: ', sparse_rec_err
            if np.abs((err - sparse_rec_err) / err) <= rel_err and err >= sparse_rec_err:
                break
            err = sparse_rec_err
    return W, H, H2, new_err


def get_matching_gene_inds(src_gene_ids, trg_gene_ids):
    if not np.unique(src_gene_ids).size == src_gene_ids.size:
        # raise Exception('(MTL) Gene ids are supposed to be unique.')
        print(('\nWarning! (MTL gene ids) Gene ids are supposed to be unique. '
              'Only {0} of {1}  entries are unique.'.format(np.unique(src_gene_ids).shape[0], src_gene_ids.shape[0])))
        print('Only first occurance will be used.\n')
    if not np.unique(trg_gene_ids).size == trg_gene_ids.size:
        # raise Exception('(Target) Gene ids are supposed to be unique.')
        print(('\nWarning! (Target gene ids) Gene ids are supposed to be unique. '
              'Only {0} of {1}  entries are unique.'.format(np.unique(trg_gene_ids).shape[0], trg_gene_ids.shape[0])))
        print('Only first occurance will be used.\n')

    # common_ids = np.intersect1d(trg_gene_ids, src_gene_ids)
    # sort the common ids according to target gene ids
    common_ids = []
    for i in range(trg_gene_ids.size):
        if np.any(trg_gene_ids[i] == src_gene_ids):
            common_ids.append(trg_gene_ids[i])
    # common_ids = np.array(common_ids, dtype=np.str)
    common_ids = np.array(common_ids)

    #print('Both datasets have (after processing) {0} (src={1}%,trg={2}%) gene ids in common.'.format(
    #    common_ids.shape[0],
    #    np.int(np.float(common_ids.size) / np.float(src_gene_ids.size)*100.0),
    #    np.int(np.float(common_ids.size) / np.float(trg_gene_ids.size)*100.0)))

    #print('Number of common genes must not be 0!')
    assert(common_ids.shape[0] > 0)

    # find indices of common_ids in pgene_ids and gene_ids
    inds1 = np.zeros(common_ids.shape[0], dtype=np.int)
    inds2 = np.zeros(common_ids.shape[0], dtype=np.int)
    for i in range(common_ids.shape[0]):
        # 1: inds1[i] = np.where(common_ids[i] == trg_gene_ids)[0][0]
        inds = np.where(common_ids[i] == trg_gene_ids)[0]
        if inds.size > 1:
            inds1[i] = inds[0]
        else:
            inds1[i] = inds
        # 2: inds2[i] = np.where(common_ids[i] == src_gene_ids)[0][0]
        inds = np.where(common_ids[i] == src_gene_ids)[0]
        if inds.size > 1:
            inds2[i] = inds[0]
        else:
            inds2[i] = inds
    return inds1, inds2
