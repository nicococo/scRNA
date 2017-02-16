import scipy.cluster.hierarchy as spc
import scipy.spatial.distance as dist
import scipy.stats as stats
import scipy.linalg as sl
import sklearn.cluster as cluster

from utils import *

# These are the SC3 labels for Ting with 7 clusters, PCA, Euclidean distances
SC3_Ting7_results = [
[1,2,3,4,5,6,7,12,35,38,39,40,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,91,122,123,124,125,126,127,128,129,132,133,134,137,138,139,141,160,172,176,177,178,179,180,181,182,183,184,185,186,187],
[31,32,33,34,36,37,41,42,70,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105],
[19,20,21,22,23,24,25,26,27,28,29,30],
[46,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121],
[130,131,135,140,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,161,162,163,164,165,166,167,168,169,171,173,174,175],
[8,9,10,11,13,14,15,16,17,18],
[136,170]]


def get_sc3_Ting_labels():
    """
    :return: array {0,..,6}^187 (results from SC3 Ting dataset)
    """
    sc3_labels = np.zeros(187)
    for lbl in range(len(SC3_Ting7_results)):
        inds = np.array(SC3_Ting7_results[lbl], dtype=np.int)-1
        # print lbl, inds.shape
        sc3_labels[inds] = lbl
    return sc3_labels


def cell_filter(data, num_expr_genes=2000, non_zero_threshold=2):
    """
    :param data: transcripts x cells data matrix
    :return: indices of valid cells
    """
    print('SC3 cell filter with num_expr_genes={0} and non_zero_threshold={1}'.format(num_expr_genes, non_zero_threshold))
    ai, bi = np.where(np.isnan(data))
    data[ai, bi] = 0
    num_transcripts, num_cells = data.shape
    res = np.sum(data >= non_zero_threshold , axis=0)
    return np.where(np.isfinite(res) & (res >= num_expr_genes))[0]


def gene_filter(data, perc_consensus_genes=0.94, non_zero_threshold=2):
    """
    :param data: transcripts x cells data matrix
    :return: indices of valid transcripts
    """
    print('SC3 gene filter with perc_consensus_genes={0} and non_zero_threshold={1}'.format(perc_consensus_genes, non_zero_threshold))
    ai, bi = np.where(np.isnan(data))
    data[ai, bi] = 0
    num_transcripts, num_cells = data.shape
    res_l = np.sum(data >= non_zero_threshold , axis=1)
    res_h = np.sum(data > 0 , axis=1)
    lower_bound = np.float(num_cells)*(1.-perc_consensus_genes)
    upper_bound = np.float(num_cells)*perc_consensus_genes
    return np.where((res_l >= lower_bound) & (res_h <= upper_bound))[0]


def data_transformation_log2(data):
    """
    :param data: transcripts x cells data matrix
    :return: log2 transformed data
    """
    print('SC3 log2 data transformation.')
    return np.log2(data + 1.)


def da_nmf_distances(data, gene_ids, da_model, reject_ratio=0., metric='euclidean', mixture=0.5):
    if mixture == 0.0:
        return distances(data, [], metric=metric)

    W, H, H2 = da_model

    # convex combination of vanilla distance and nmf distance
    dist1 = distances(data, [], metric=metric)
    dist2 = distances(W.dot(H2), [], metric=metric)

    # normalize distance
    if np.max(dist2) < 1e-10:
        if mixture == 1.0:
            raise Exception('Distances are all zero and mixture=1.0. Seems that source and target'
                            ' data do not go well together.')
        else:
            print 'Warning! Max distance is 0.0.'
    else:
        print 'Max dists before normalization: ', np.max(dist1), np.max(dist2)
        dist2 *= np.max(dist1) / np.max(dist2)
    return mixture*dist2 + (1.-mixture)*dist1


def distances(data, gene_ids, metric='euclidean'):
    """
    :param data: transcripts x cells data matrix
    :param gene_ids: #transcripts vector with corresponding gene(transcript) ids
    :param metric: string with distance metric name (ie. 'euclidean','pearson','spearman')
    :return: cells x cells distance matrix
    """
    print('SC3 pairwise distance computations (metric={0}).'.format(metric))

    # Euclidean: Use the standard Euclidean (as-the-crow-flies) distance.
    # Euclidean Squared: Use the Euclidean squared distance in cases where you would use regular Euclidean distance in Jarvis-Patrick or K-Means clustering.
    # Manhattan: Use the Manhattan (city-block) distance.
    # Pearson Correlation: Use the Pearson Correlation coefficient to cluster together genes or samples with similar behavior; genes or samples with opposite behavior are assigned to different clusters.
    # Pearson Squared: Use the squared Pearson Correlation coefficient to cluster together genes with similar or opposite behaviors (i.e. genes that are highly correlated and those that are highly anti-correlated are clustered together).
    # Chebychev: Use Chebychev distance to cluster together genes that do not show dramatic expression differences in any samples; genes with a large expression difference in at least one sample are assigned to different clusters.
    # Spearman: Use Spearman Correlation to cluster together genes whose expression profiles have similar shapes or show similar general trends (e.g. increasing expression with time), but whose expression levels may be very different.

    if metric == 'pearson':
        X = 1. - np.corrcoef(data.T)
    elif metric == 'spearman':
        X, _ = stats.spearmanr(data, axis=0)
        X = 1.-X
    else:
        X = dist.pdist(data.T, metric=metric)
        X = dist.squareform(X)
        # print X.shape
    return X


def transformations(dm, components=5, method='pca'):
    """
    :param dm: cells x cells distance matrix
    :param components: number of eigenvector/eigenvalues to use
    :param method: either 'pca' or 'spectral'
    :return: cells x cells (centered!) distance matrix, cells x components Eigenvectors
    """
    print('SC3 {1} transformation (components={0}).'.format(components, method.upper()))
    if method == 'spectral':
        A = np.exp(-dm/np.max(dm))
        D = np.sum(dm, axis=1)
        L = D - A
        D1 = np.diag(D.__pow__(-0.5))
        D1[np.isinf(D)] = 0.0
        dm = D1.dot(L.dot(D1))
        # Laplacian:
        #  L := D - A
        # symmetric normalized laplacian:
        #  L_sym := D^-0.5 L D^-0.5
        inds = range(components)
    else:
        # column-wise scaling and normalizing
        num_cells = dm.shape[0]
        # J = np.eye(num_cells) - 1./np.float(num_cells)*np.ones((num_cells, num_cells))
        # dm = 0.5*J.dot(dm.dot(J))
        dm = dm - np.repeat(np.mean(dm, axis=0).reshape((1, num_cells)), num_cells, axis=0)
        dm = dm / np.repeat(np.std(dm, axis=0).reshape((1, num_cells)), num_cells, axis=0)

    # vals: the eigenvalues in ascending order, each repeated according to its multiplicity.
    # vecs: the column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]
    # vals, vecs = np.linalg.eigh(dm)
    _, vals, ev = sl.svd(dm)
    vecs = ev.T
    vals /= np.sqrt(np.max((1, num_cells - 1)))

    if method == 'pca':
        # This part is done to imitate sc3 behavior which only sorts absolute Eigenvalues
        # making the highest Eigenvalue first followed by the smallest (ascending) Eigenvalues
        # x = np.sqrt(vals*vals)
        # inds = np.argsort(-x)  # argsort is ascending order
        # inds = np.argsort(vals)  # argsort is ascending order
        # inds = inds[:components]
        inds = np.arange(components)
        # print inds

    # inds = range(vals.size-components, vals.size)
    # inds = range(components)
    # print inds
    # print vals.size, vals
    D = np.diag(vals[inds])
    return vecs[:, inds].dot(D.dot(vecs[:, inds].T)), vecs[:, inds]


def intermediate_kmeans_clustering(X, k=5, n_init=10, max_iter=10000, init='k-means++'):
    """
    :param X: cells x d vector
    :param k: number of clusters
    :param n_init: number of re-starts for k-means
    :param max_iter: maximum number of iterations per run
    :param init: initialization strategy for k-means (either 'k-means++' or 'random')
    :return: cells x 1 labels
    """
    kmeans = cluster.KMeans(n_clusters=k, precompute_distances=True, n_init=n_init, max_iter=max_iter,
                            init=init, n_jobs=1)
    labels = kmeans.fit_predict(X)
    assert labels.size == X.shape[0]
    return labels


def build_consensus_matrix(X):
    """
    :param X: n x cells label matrix
    :return: cells x cells consensus matrix
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    n, cells = X.shape
    consensus = np.zeros((cells, cells), dtype=np.float)
    for i in range(n):
        t = dist.squareform(dist.pdist(X[i, :].reshape(cells, 1)))
        t = np.array(t, dtype=np.int)
        # print np.unique(t)
        t[t != 0] = -1
        t[t == 0] = +1
        t[t == -1] = 0
        # print np.unique(t)
        consensus += np.array(t, dtype=np.float)
    consensus /= np.float(n)
    return consensus


def consensus_clustering(consensus, n_components=5):
    """
    :param consensus: cells x cells consensus matrix
    :param n_components: number of clusters
    :return: cells x 1 labels
    """
    print 'SC3 Agglomorative hierarchical clustering.'
    # condensed distance matrix
    cdm = dist.pdist(consensus)
    # hierarchical clustering (SC3: complete agglomeration + cutree)
    hclust = spc.complete(cdm)
    cutree = spc.cut_tree(hclust, n_clusters=n_components)
    labels = cutree.reshape(consensus.shape[0])
    # Below is the hclust code for the older version, fyi
    # hclust = spc.linkage(cdm)
    # labels = spc.fcluster(hclust, n_components, criterion='maxclust')
    return labels, dist.squareform(cdm)