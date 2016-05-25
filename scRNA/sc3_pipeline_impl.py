import numpy as np
import sklearn as skl
import sklearn.cluster as cluster
import scipy.cluster.hierarchy as spc
import scipy.spatial.distance as dist
import scipy.stats as stats


def cell_filter(data, num_expr_genes=2000, non_zero_threshold=2):
    """
    :param data: transcripts x cells data matrix
    :return: indices of valid cells
    """
    print('SC3 cell filter with num_expr_genes={0} and non_zero_threshold={1}'.format(num_expr_genes, non_zero_threshold))
    num_transcripts, num_cells = data.shape
    res = np.sum(data >= non_zero_threshold , axis=0)
    return np.where(np.isfinite(res) & (res >= num_expr_genes))[0]


def gene_filter(data, perc_consensus_genes=0.94, non_zero_threshold=2):
    """
    :param data: transcripts x cells data matrix
    :return: indices of valid transcripts
    """
    print('SC3 gene filter with perc_consensus_genes={0} and non_zero_threshold={1}'.format(perc_consensus_genes, non_zero_threshold))
    num_transcripts, num_cells = data.shape
    res = np.sum(data >= non_zero_threshold , axis=1)
    lower_bound = np.float(num_cells)*(1.-perc_consensus_genes)
    upper_bound = np.float(num_cells)*perc_consensus_genes
    return np.where((res >= lower_bound) & (res <= upper_bound))[0]


def data_transformation(data):
    """
    :param data: transcripts x cells data matrix
    :return: log2 transformed data
    """
    print('SC3 log2 data transformation.')
    return np.log2(data+1.)


def distances(data, metric='euclidean'):
    """
    :param data: transcripts x cells data matrix
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
    print X.shape
    return X


def transformations(dm, components=5, method='pca'):
    """
    :param dm: cells x cells distance matrix
    :param components: number of eigenvector/eigenvalues to use
    :param method: either 'pca' or 'spectral'
    :return: cells x cells distance matrix, cells x components Eigenvectors
    """
    print('SC3 {1} transformation (components={0}).'.format(components, method.upper()))
    if method=='spectral':
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
    else:
        dm /= np.max(dm)
        dm -= np.mean(dm)

    # vals: the eigenvalues in ascending order, each repeated according to its multiplicity.
    # vecs: the column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]
    vals, vecs = np.linalg.eigh(dm)
    # inds = range(vals.size-components, vals.size)
    inds = range(components)
    #print inds
    #print vals.size, vals
    D = np.diag(vals[inds])
    return vecs[:, inds].dot(D.dot(vecs[:, inds].T)), vecs[:, inds]


def intermediate_kmeans_clustering(X, k=5):
    """
    :param X: cells x d vector
    :param k: number of clusters
    :return: cells x 1 labels
    """
    kmeans = cluster.KMeans(n_clusters=k, n_init=1000, max_iter=10^9)
    labels = kmeans.fit_predict(X)
    assert labels.size == X.shape[0]
    return labels


def consensus_clustering(X, n_components=5):
    """
    :param X: n x cells matrix
    :param n_components: number of clusters
    :return: cells x 1 labels
    """
    cells = X.shape[1]
    n = X.shape[0]
    print 'SC3 Build consensus matrix with inputs', X.shape
    consensus = np.zeros((cells, cells))
    for i in range(n):
        A = np.zeros((cells, cells))
        for j in range(cells):
            A[:, j] = (X[i, :] == X[i, j])
            A[j, :] = (X[i, :] == X[i, j])
        consensus += np.array(A, dtype=np.float)
    consensus /= np.float(n)
    print consensus

    print 'SC3 Agglomorative hierarchical clustering.'
    # spc.linkage()

    # condensed distance matrix
    cdm = dist.pdist(consensus)
    #print dm.shape
    #cdm = dist.squareform(dm)
    #print cdm.shape, consensus.size
    hclust = spc.linkage(cdm)

    labels = spc.fcluster(hclust, n_components, criterion='maxclust')
    print np.sort(np.unique(labels)), labels

    import matplotlib.pyplot as plt
    #spc.dendrogram(hclust, truncate_mode='lastp', p=40, show_contracted=True)

    inds = np.argsort(labels)
    X = X[:, inds]
    consensus = np.zeros((cells, cells))
    for i in range(n):
        A = np.zeros((cells, cells))
        for j in range(cells):
            A[:, j] = (X[i, :] == X[i, j])
            A[j, :] = (X[i, :] == X[i, j])
        consensus += np.array(A, dtype=np.float)
    consensus /= np.float(n)

    plt.imshow(consensus, cmap='bwr')
    plt.show()

    for i in range(n_components):
        inds = np.where(labels==i+1)[0]
        foo = '[{0}'.format(inds[0])
        for j in range(1, inds.size):
            foo = '{0},{1}'.format(foo, inds[j])
        print foo, ']'
        print '-------------------------------------'

    # cut = spc.cut_tree(hclust, n_clusters=n_components)
    # inds = spc.leaves_list(cut)
    # print cut[inds]
    return -1