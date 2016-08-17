import numpy as np
from sklearn import decomposition as decomp
import sklearn.metrics as metrics

from sc3_pipeline_impl import cell_filter, gene_filter, data_transformation_log2, data_transformation_null, distances
from utils import load_dataset_tsv


def gene_names_conversion(gene_ids1, gene_ids2):
    import utils
    mypath = utils.__file__
    mypath = mypath.rsplit('/', 1)[0]
    # print mypath
    gene_names = np.loadtxt('{0}/gene_names.txt'.format(mypath), skiprows=1, dtype='object')
    gene_id_map = dict()
    for i in range(gene_names.shape[0]):
        gene_id_map[gene_names[i, 0]] = gene_names[i, 1]
    return gene_id_map


def filter_and_sort_genes(gene_ids1, gene_ids2):
    gene_id_map = gene_names_conversion(gene_ids1, gene_ids2)

    inds1 = []
    inds2 = []
    # print gene_ids1.size, gene_ids2.size
    for i in range(gene_ids1.size):
        id = gene_ids1[i]
        if gene_id_map.has_key(id):
            ens_id = gene_id_map[id]
            ind = np.where(gene_ids2 == ens_id)[0]
            if ind.size == 1:
                # exactly 1 id found
                inds1.append(i)
                inds2.append(ind[0])

    # print len(inds1), len(inds2)
    return np.array(inds1, dtype=np.int), np.array(inds2, dtype=np.int)


def mtl_distance(data, gene_ids, fmtl=None, fmtl_geneids=None, metric='euclidean',
                 mixture=0.75, nmf_k=10, nmf_alpha=1.0, nmf_l1=0.75, data_transformation_fun=None):
    pdata, pgene_ids, labels = load_dataset_tsv(fmtl, fgenes=fmtl_geneids)
    num_transcripts, num_cells = data.shape

    # filter cells
    remain_inds = np.arange(0, num_cells)
    res = cell_filter(pdata, num_expr_genes=2000, non_zero_threshold=2)
    remain_inds = np.intersect1d(remain_inds, res)
    A = pdata[:, remain_inds]

    # filter genes
    remain_inds = np.arange(0, num_transcripts)
    res = gene_filter(A, perc_consensus_genes=0.94, non_zero_threshold=2)
    remain_inds = np.intersect1d(remain_inds, res)

    # transform data
    X = data_transformation_fun(A[remain_inds, :])
    pgene_ids = pgene_ids[remain_inds]

    # find (and translate) a common set of genes
    # inds1, inds2 = filter_and_sort_genes(gene_ids, pgene_ids)

    # expect identifiers to be unique
    print np.unique(pgene_ids).shape, pgene_ids.shape
    print np.unique(gene_ids).shape, gene_ids.shape

    print np.unique(gene_ids)
    print np.intersect1d(pgene_ids, np.unique(pgene_ids))

    if not np.unique(pgene_ids).shape[0] == pgene_ids.shape[0]:
        # raise Exception('(MTL) Gene ids are supposed to be unique.')
        print('\nError! (MTL gene ids) Gene ids are supposed to be unique. '
              'Only {0} of {1}  entries are unique.'.format(np.unique(pgene_ids).shape[0], pgene_ids.shape[0]))
        print('Only first occurance will be used.\n')
    if not np.unique(gene_ids).shape[0] == gene_ids.shape[0]:
        # raise Exception('(Target) Gene ids are supposed to be unique.')
        print('\nError! (Target gene ids) Gene ids are supposed to be unique. '
              'Only {0} of {1}  entries are unique.'.format(np.unique(gene_ids).shape[0], gene_ids.shape[0]))
        print('Only first occurance will be used.\n')

    common_ids = np.intersect1d(gene_ids, pgene_ids)
    print('Both datasets have (after processing) {0} gene ids in common.'.format(common_ids.shape[0]))

    # find indices of common_ids in pgene_ids and gene_ids
    inds1 = np.zeros(common_ids.shape[0], dtype=np.int)
    inds2 = np.zeros(common_ids.shape[0], dtype=np.int)
    for i in range(common_ids.shape[0]):
        inds1[i] = np.argwhere(common_ids[i] == gene_ids)[0]
        inds2[i] = np.argwhere(common_ids[i] == pgene_ids)[0]

    print 'MTL source {0} genes -> {1} genes.'.format(pgene_ids.size, inds2.size)
    print 'MTL target {0} genes -> {1} genes.'.format(gene_ids.size, inds1.size)

    X = X[inds2, :]
    nmf = decomp.NMF(alpha=nmf_alpha, init='nndsvdar', l1_ratio=nmf_l1, max_iter=1000,
        n_components=nmf_k, random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    # print nmf.reconstruction_err_

    # reconstruct given dataset using the Pfizer dictionary
    H = np.random.randn(nmf_k, data.shape[1])
    a1, a2 = np.where(H < 0.)
    H[a1, a2] *= -1.
    Y = data[inds1, :].copy()
    # TODO: some NMF MU steps
    for i in range(800):
        # print 'Iteration: ', i
        # print '  Elementwise absolute reconstruction error: ', np.sum(np.abs(Y - W.dot(H)))/np.float(Y.size)
        # print '  Fro-norm reconstruction error: ', np.sqrt(np.sum((Y - W.dot(H))*(Y - W.dot(H))))
        H = H * W.T.dot(Y) / W.T.dot(W.dot(H))

    H2 = np.zeros((nmf_k, data.shape[1]))
    H2[ (np.argmax(H, axis=0), np.arange(data.shape[1])) ] = 1
    print H2

    # convex combination of vanilla distance and nmf distance
    dist1 = distances(data, [], metric=metric)
    dist2 = distances(W.dot(H2), [], metric=metric)
    # normalize distance
    print 'Max dists: ', np.max(dist1), np.max(dist2)
    normalizer = np.max(dist1) / np.max(dist2)
    dist2 *= normalizer

    return mixture*dist2 + (1.-mixture)*dist1


def mtl_toy_distance(data, gene_ids, src_data, src_labels=None, trg_labels=None, metric='euclidean', mixture=0.75, nmf_k=4):
    if mixture == 0.0:
        print('No MTL used (mixture={0})'.format(mixture))
        return distances(data, [], metric=metric)

    # transform data
    X = data_transformation(src_data[gene_ids, :])
    nmf = decomp.NMF(alpha=1., init='nndsvdar', l1_ratio=0.5, max_iter=1000,
        n_components=nmf_k, random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    # print nmf.reconstruction_err_
    # print H
    if src_labels is not None:
        print 'Labels in src: ', np.unique(src_labels)
        print 'ARI: ', metrics.adjusted_rand_score(src_labels, np.argmax(H, axis=0))

    # reconstruct given dataset using the Pfizer dictionary
    H = np.random.randn(nmf_k, data.shape[1])
    Y = data.copy()
    # TODO: some NMF MU steps
    for i in range(800):
        # print 'Iteration: ', i
        # print '  Absolute elementwise reconstruction error: ', np.sum(np.abs(Y - W.dot(H)))/np.float(Y.size)
        # print '  Fro-norm reconstruction error: ', np.sqrt(np.sum((Y - W.dot(H))*(Y - W.dot(H))))
        H = H * W.T.dot(Y) / W.T.dot(W.dot(H))

    if trg_labels is not None:
        print 'Labels in trg: ', np.unique(trg_labels)
        print 'ARI: ', metrics.adjusted_rand_score(trg_labels, np.argmax(H, axis=0))

    H2 = np.zeros((nmf_k, data.shape[1]))
    H2[ (np.argmax(H, axis=0), np.arange(data.shape[1])) ] = 1.
    # H2[ (np.argmax(H, axis=0), np.arange(data.shape[1])) ] = np.max(H, axis=0)
    print H2

    # convex combination of vanilla distance and nmf distance
    dist1 = distances(data, [], metric=metric)
    # dist2 = distances(W.dot(H), [], metric=metric)
    dist2 = distances(W.dot(H2), [], metric=metric)

    # normalize distance
    print 'Max dists: ', np.max(dist1), np.max(dist2)
    normalizer = np.max(dist1) / np.max(dist2)
    dist2 *= normalizer

    # import scipy.stats as stats
    # import matplotlib.pyplot as plt
    #
    # plt.figure(1)
    # print 'Number of class 2 datapoints is {0} of {1}'.format(np.sum(trg_labels==1), trg_labels.size)
    #
    # plt.subplot(1, 3, 1)
    # kurts = stats.kurtosis(H, fisher=False, axis=0)
    # sinds = np.argsort(kurts)
    # inds = np.where(trg_labels[sinds] == 1)[0]
    #
    # plt.plot(np.arange(sinds.size), kurts[sinds], '.r', markersize=4)
    # plt.plot(inds, kurts[sinds[inds]], '.b', markersize=4)
    #
    #
    # plt.subplot(1, 3, 2)
    # dists = np.sum( (np.abs(Y - W.dot(H))**2. ), axis=0)
    # sinds = np.argsort(dists)
    # inds = np.where(trg_labels[sinds] == 1)[0]
    #
    # plt.plot(np.arange(sinds.size), dists[sinds], '.r', markersize=4)
    # plt.plot(inds, dists[sinds[inds]], '.b', markersize=4)
    #
    #
    # plt.subplot(1, 3, 3)
    # dists = np.sum( (np.abs(Y - W.dot(H2))**2. ), axis=0)
    # sinds = np.argsort(dists)
    # inds = np.where(trg_labels[sinds] == 1)[0]
    #
    # plt.plot(np.arange(sinds.size), dists[sinds], '.r', markersize=4)
    # plt.plot(inds, dists[sinds[inds]], '.b', markersize=4)
    #
    # plt.show()

    print np.max(dist1), np.max(dist2)
    return mixture*dist2 + (1.-mixture)*dist1