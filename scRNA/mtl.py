import numpy as np
from sklearn import decomposition as decomp
import sklearn.metrics as metrics

from sc3_pipeline_impl import cell_filter, gene_filter, data_transformation_log2, distances
from utils import load_dataset_tsv


def gene_names_conversion():
    """
    :return: a dictionary for gene ids
    """
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
    """
    Remove genes from either list that do not appear in the other. Sort the indices.
    :param gene_ids1: list of gene names
    :param gene_ids2: list of gene names
    :return: (same-size) lists of indices
    """
    gene_id_map = gene_names_conversion(gene_ids1, gene_ids2)
    inds1 = []
    inds2 = []
    for i in range(gene_ids1.size):
        id = gene_ids1[i]
        if gene_id_map.has_key(id):
            ens_id = gene_id_map[id]
            ind = np.where(gene_ids2 == ens_id)[0]
            if ind.size == 1:
                # exactly 1 id found
                inds1.append(i)
                inds2.append(ind[0])
    return np.array(inds1, dtype=np.int), np.array(inds2, dtype=np.int)


def mtl_distance(data, gene_ids, fmtl=None, fmtl_geneids=None, metric='euclidean',
                 mixture=0.75, nmf_k=10, nmf_alpha=1.0, nmf_l1=0.75, data_transformation_fun=None,
                 num_expr_genes=2000, non_zero_threshold=2, perc_consensus_genes=0.94):
    """
    Multitask SC3 distance function.
    :param data: Target dataset (trg-genes x trg-cells)
    :param gene_ids: Target gene ids
    :param fmtl: Filename of the scRNA source dataset (src-genes x src-cells)
    :param fmtl_geneids: Filename for corresponding source gene ids
    :param metric: Which metric should be applied.
    :param mixture: [0,1] Convex combination of target only distance and mtl distance (0: no mtl influence)
    :param nmf_k: Number of latent components (cluster)
    :param nmf_alpha: Regularization influence
    :param nmf_l1: [0,1] strength of l1-regularizer within regularization
    :param data_transformation_fun: Target data transformation function (e.g. log2+1 transfor, or None)
    :param num_expr_genes: cell filter parameter
    :param non_zero_threshold: cell- and gene-filter parameter
    :param perc_consensus_genes: gene filter parameter
    :return: Distance matrix trg-cells x trg-cells
    """
    pdata, pgene_ids, labels = load_dataset_tsv(fmtl, fgenes=fmtl_geneids)
    num_transcripts, num_cells = data.shape

    # filter cells
    remain_inds = np.arange(0, num_cells)
    res = cell_filter(pdata, num_expr_genes=num_expr_genes, non_zero_threshold=non_zero_threshold)
    remain_inds = np.intersect1d(remain_inds, res)
    A = pdata[:, remain_inds]

    # filter genes
    remain_inds = np.arange(0, num_transcripts)
    res = gene_filter(A, perc_consensus_genes=perc_consensus_genes, non_zero_threshold=non_zero_threshold)
    remain_inds = np.intersect1d(remain_inds, res)

    # transform data
    X = A[remain_inds, :]
    if data_transformation_fun is not None:
        X = data_transformation_fun(X)
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

    W, H, H2, Hsrc = mtl_nmf(X[inds2, :], data[inds1, :], nmf_k=nmf_k, nmf_alpha=nmf_alpha, nmf_l1=nmf_l1)

    # convex combination of vanilla distance and nmf distance
    dist1 = distances(data, [], metric=metric)
    dist2 = distances(W.dot(H2), [], metric=metric)
    # normalize distance
    print 'Max dists: ', np.max(dist1), np.max(dist2)
    normalizer = np.max(dist1) / np.max(dist2)
    dist2 *= normalizer
    return mixture*dist2 + (1.-mixture)*dist1


def mtl_nmf(Xsrc, Xtrg, nmf_k=10, nmf_alpha=1.0, nmf_l1=0.75, max_iter=5000, rel_err=1e-6, verbosity=1):
    """
    Multitask clustering. The source dataset 'Xsrc' is clustered using NMF. Resulting
    dictionary 'W' is then used to reconstruct 'Xtrg'
    :param Xsrc: genes x src_cells matrix
    :param Xtrg: genes x trg_cells matrix
    :param nmf_k: number of latent components (cluster)
    :param nmf_alpha: regularization influence
    :param nmf_l1: [0,1] strength of influence of l1-regularizer within regularization
    :return: dictionary W (genes x nmf_k), trg-data matrix H (nmf_k x trg-cells), trg-data matrix H2,
    and src-data matrix (nmf_k x src-cells)
    """
    nmf = decomp.NMF(alpha=nmf_alpha, init='nndsvdar', l1_ratio=nmf_l1, max_iter=1000,
        n_components=nmf_k, random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
    W = nmf.fit_transform(Xsrc)
    Hsrc = nmf.components_

    H = np.random.randn(nmf_k, Xtrg.shape[1])
    a1, a2 = np.where(H < 0.)
    H[a1, a2] *= -1.
    n_iter = 0
    err = 1e10
    while n_iter < max_iter:
        n_iter += 1
        H *= W.T.dot(Xtrg) / W.T.dot(W.dot(H))
        new_err = np.sum(np.abs(Xtrg - W.dot(H)))/np.float(Xtrg.size)  # absolute
        # new_err = np.sqrt(np.sum((Xtrg - W.dot(H))*(Xtrg - W.dot(H)))) / np.float(Xtrg.size)  # frobenius
        if np.abs((err - new_err) / err) <= rel_err:
            break
        err = new_err
    print '  Number of iterations for reconstruction     : ', n_iter
    print '  Elementwise absolute reconstruction error   : ', np.sum(np.abs(Xtrg - W.dot(H))) / np.float(Xtrg.size)
    print '  Fro-norm reconstruction error               : ', np.sqrt(np.sum((Xtrg - W.dot(H))*(Xtrg - W.dot(H)))) / np.float(Xtrg.size)

    H2 = np.zeros((nmf_k, Xtrg.shape[1]))
    H2[(np.argmax(H, axis=0), np.arange(Xtrg.shape[1]))] = 1
    # H2[ (np.argmax(H, axis=0), np.arange(Xtrg.shape[1])) ] = np.sum(H, axis=0)

    print '  H2 Elementwise absolute reconstruction error: ', np.sum(np.abs(Xtrg - W.dot(H2))) / np.float(Xtrg.size)
    print '  H2 Fro-norm reconstruction error            : ', np.sqrt(np.sum((Xtrg - W.dot(H2))*(Xtrg - W.dot(H2)))) / np.float(Xtrg.size)

    return W, H, H2, Hsrc


def mtl_toy_distance(data, gene_ids, src_data, src_labels=None, trg_labels=None, metric='euclidean', mixture=0.75, nmf_k=4):
    """
    Multitask SC3 distance function for toy data (i.e. no transformation, no gene id matching).
    :param data:
    :param gene_ids: (not used)
    :param src_data:
    :param src_labels: (optional)
    :param trg_labels: (optional)
    :param metric: Which metric should be applied.
    :param mixture: [0,1] Convex combination of target only distance and mtl distance (0: no mtl influence)
    :param nmf_k: Number of latent components (cluster)
    :return:
    """
    if mixture == 0.0:
        print('No MTL used (mixture={0})'.format(mixture))
        return distances(data, [], metric=metric)

    W, H, H2, Hsrc = mtl_nmf(src_data, data, nmf_k=nmf_k, nmf_alpha=1.)

    if src_labels is not None:
        print 'Labels in src: ', np.unique(src_labels)
        print 'ARI: ', metrics.adjusted_rand_score(src_labels, np.argmax(Hsrc, axis=0))
    if trg_labels is not None:
        print 'Labels in trg: ', np.unique(trg_labels)
        print 'ARI: ', metrics.adjusted_rand_score(trg_labels, np.argmax(H, axis=0))

    # convex combination of vanilla distance and nmf distance
    dist1 = distances(data, [], metric=metric)
    dist2 = distances(W.dot(H2), [], metric=metric)
    # normalize distance
    print 'Max dists before normalization: ', np.max(dist1), np.max(dist2)
    dist2 *= np.max(dist1) / np.max(dist2)

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
    # plt.subplot(1, 3, 2)
    # dists = np.sum( (np.abs(Y - W.dot(H))**2. ), axis=0)
    # sinds = np.argsort(dists)
    # inds = np.where(trg_labels[sinds] == 1)[0]
    #
    # plt.plot(np.arange(sinds.size), dists[sinds], '.r', markersize=4)
    # plt.plot(inds, dists[sinds[inds]], '.b', markersize=4)
    #
    # plt.subplot(1, 3, 3)
    # dists = np.sum( (np.abs(Y - W.dot(H2))**2. ), axis=0)
    # sinds = np.argsort(dists)
    # inds = np.where(trg_labels[sinds] == 1)[0]
    #
    # plt.plot(np.arange(sinds.size), dists[sinds], '.r', markersize=4)
    # plt.plot(inds, dists[sinds[inds]], '.b', markersize=4)
    # plt.show()
    print 'Max dists after normalization: ', np.max(dist1), np.max(dist2)
    fdist = mixture*dist2 + (1.-mixture)*dist1
    print mixture
    if np.any(fdist < 0.0):
        raise Exception('Final distance matrix contains negative values.')
    if np.any(np.isnan(fdist)):
        raise Exception('Final distance matrix contains NaNs.')
    if np.any(np.isinf(fdist)):
        raise Exception('Final distance matrix contains Infs.')
    return fdist