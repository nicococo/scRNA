import numpy as np
import os

def load_dataset_by_name(name):
    if name == 'Pfizer':
        foo = np.load('pfizer_data.npz')
        data  = foo['rpm_data']
        gene_ids = foo['transcripts']
    if name == 'Usoskin':
        foo = np.load('/Users/nicococo/Documents/scRNA-data/Usoskin.npz')
        data  = foo['data']
        gene_ids = foo['transcripts']
    if name == 'Ting':
        foo = np.load('/Users/nicococo/Documents/scRNA-data/Ting.npz')
        data  = foo['data']
        gene_ids = foo['transcripts']
    return data, gene_ids


def load_dataset(fname):
    if not os.path.exists(fname):
        raise StandardError('File \'{0}\' not found.'.format(fname))
    foo = np.load(fname)
    data  = foo['data']
    gene_ids = foo['transcripts']
    if 'pfizer' in fname:
        print('Load rpm data instead.')
        data = foo['rpm_data']
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


def generate_toy_data(num_genes=5000, num_cells= 400,num_clusters= 5, dirichlet_parameter_cluster_size=1,
                      shape_power_law=0.1, upper_bound_counts=300000, dirichlet_parameter_counts=1):
    # Toy experiment parameters
    # shape_power_law = 0.1  # shape parameter of the power law -  between 0 and 1, the smaller this value the more extreme the power law
    # num_clusters = 5
    # dirichlet_parameter_cluster_size = 1  # between 0 and inf, smaller values make cluster sizes more similar

    # num_genes = 5000  # 20000
    # num_cells = 400  # 400
    # upper_bound_counts = 300000  # 300000
    # dirichlet_parameter_counts = 1  # between 0 and inf, noise parameter: smaller values make counts within cluster more similar (splitting the total
    # abundances in more equal parts for each cell)

    # Generate Cluster sizes
    cluster_sizes = np.squeeze(np.round(np.random.dirichlet(np.ones(num_clusters) * dirichlet_parameter_cluster_size, size=1) * (num_cells - num_clusters))) + 1
    while min(cluster_sizes) == 1:
        cluster_sizes = np.squeeze(np.round(np.random.dirichlet(np.ones(num_clusters) * dirichlet_parameter_cluster_size, size=1) * (num_cells - num_clusters))) + 1

    if np.sum(cluster_sizes) != num_cells:
        cluster_sizes[0] = cluster_sizes[0] - (np.sum(cluster_sizes) - num_cells)

    # Generate data for each cluster
    data_complete = []
    labels_now = []
    for cluster_ind in range(num_clusters):
        # Draw samples from the power law distribution
        sample = np.round(np.random.power(shape_power_law, num_genes) * upper_bound_counts)

        # the abundance plot
        # plt.plot(np.sort(sample)[::-1], 'o')

        # Generate data
        data_now = []
        for i in range(num_genes):
            row_total = int(sample[i])
            one_gene = np.round(np.random.dirichlet(np.ones(cluster_sizes[cluster_ind]) * dirichlet_parameter_counts, size=1) * row_total)
            data_now.append(one_gene)

        # plt.show()
        data_complete.append(np.squeeze(np.asarray(data_now)))
        labels_now.append(np.tile(cluster_ind+1, cluster_sizes[cluster_ind]))

    assert min(cluster_sizes) > 1
    data = np.squeeze(np.hstack(data_complete))
    labels = np.hstack(labels_now)
    return [data, labels]