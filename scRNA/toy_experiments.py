import pdb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import sklearn.decomposition as decomp
from sklearn.metrics import adjusted_rand_score
import statsmodels.stats.api as sms
from sklearn.cross_validation import train_test_split


def generate_toy_data(num_genes=5000, num_cells= 400,num_clusters= 5,dirichlet_parameter_cluster_size=1, shape_power_law=0.1,upper_bound_counts=300000,
                      dirichlet_parameter_counts=1):
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
    pdb.set_trace()
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


def TSNE_plot(data, labels):
    model = TSNE(n_components=2, random_state=0)
    tsne_comps = model.fit_transform(np.transpose(data))

    df = pd.DataFrame(dict(x=tsne_comps[:, 0], y=tsne_comps[:, 1], label=labels))

    groups = df.groupby('label')

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    ax.legend(numpoints=1)
    plt.show()


def cell_filter(data, num_expr_genes=2000, non_zero_threshold=2):
    res = np.sum(data >= non_zero_threshold, axis=0)
    return np.where(np.isfinite(res) & (res >= num_expr_genes))[0]


def gene_filter(data, perc_consensus_genes=0.94, non_zero_threshold=2):
    num_transcripts, num_cells = data.shape
    res_l = np.sum(data >= non_zero_threshold, axis=1)
    res_h = np.sum(data > 0 , axis=1)
    lower_bound = np.float(num_cells)*(1.-perc_consensus_genes)
    upper_bound = np.float(num_cells)*perc_consensus_genes
    return np.where((res_l >= lower_bound) & (res_h <= upper_bound))[0]


def data_transformation(data):
    return np.log2(data + 1.)


def NMF_clustering(data, num_clusters):
    num_transcripts, num_cells = data.shape
    labels = np.full(num_cells, np.inf)

    # Cell filtering
    remain_inds_cells = np.arange(0, num_cells)
    res = cell_filter(data, num_expr_genes=num_transcripts/10, non_zero_threshold=1)
    remain_inds_cells = np.intersect1d(remain_inds_cells, res)
    A = data[:, remain_inds_cells]

    # Gene filtering
    remain_inds_genes = np.arange(0, num_transcripts)
    res = gene_filter(data, perc_consensus_genes=0.94, non_zero_threshold=2)
    remain_inds_genes = np.intersect1d(remain_inds_genes, res)

    # Log transformation
    X = data_transformation(A[remain_inds_genes, :])
    num_transcripts, num_cells = X.shape

    # Show filtering results
    # print X.shape

    # Perform NMF clustering
    if num_cells >= num_clusters:
        nmf = decomp.NMF(alpha=10.1, init='nndsvdar', l1_ratio=0.9, max_iter=1000, n_components=num_clusters, random_state=0, shuffle=True, solver='cd', tol=0.00001,
                         verbose=0)
        nmf.fit_transform(X)
        H = nmf.components_
        labels[remain_inds_cells] = np.argmax(H, axis=0)

    return labels


def split_source_target(toy_data, true_toy_labels, proportion_source, mode):
    # 1 = split randomly, 2 = split randomly, but stratified, 3 = Have some overlapping and some exclusive clusters, 4 = have only exclusive clusters

    if mode == 1:
        toy_data_source, toy_data_target, true_toy_labels_source, true_toy_labels_target = train_test_split(np.transpose(toy_data), true_toy_labels,
                                                                                                            train_size=proportion_source)
    elif mode == 2:
        toy_data_source, toy_data_target, true_toy_labels_source, true_toy_labels_target = train_test_split(np.transpose(toy_data), true_toy_labels,
                                                                                                            train_size=proportion_source, stratify=true_toy_labels)
    elif mode == 3:
        cluster_names = np.unique(true_toy_labels)

        # Assign exclusive clusters
        # To source data set
        source_cluster = cluster_names[0]
        source_indices = (true_toy_labels == source_cluster)
        toy_data_source_exclusive = toy_data[:, source_indices]
        true_toy_labels_source_exclusive = true_toy_labels[source_indices]

        # To target data set
        target_cluster = cluster_names[1]
        target_indices = (true_toy_labels == target_cluster)
        toy_data_target_exclusive = toy_data[:, target_indices]
        true_toy_labels_target_exclusive = true_toy_labels[target_indices]

        # Distribute the rest randomly
        random_clusters = cluster_names[2:]
        random_indices_mat = []
        for i in range(len(random_clusters)):
            random_indices_mat.append(true_toy_labels == random_clusters[i])
        random_indices = np.any(random_indices_mat, 0)
        toy_data_random = toy_data[:, random_indices]
        true_toy_labels_random = true_toy_labels[random_indices]

        toy_data_source_random, toy_data_target_random, true_toy_labels_source_random, true_toy_labels_target_random = train_test_split(np.transpose(toy_data_random),
                                                                                                                 true_toy_labels_random, train_size=proportion_source)

        # Combine exclusive and random data
        toy_data_source = np.concatenate((toy_data_source_exclusive, np.transpose(toy_data_source_random)), axis=1)
        toy_data_target = np.concatenate((toy_data_target_exclusive, np.transpose(toy_data_target_random)), axis=1)
        true_toy_labels_source = np.concatenate((true_toy_labels_source_exclusive, np.transpose(true_toy_labels_source_random)))
        true_toy_labels_target = np.concatenate((true_toy_labels_target_exclusive, np.transpose(true_toy_labels_target_random)))

    elif mode == 4:

        cluster_names = np.unique(true_toy_labels)
        source_clusters = cluster_names[0: int(len(cluster_names) * proportion_source)]
        target_clusters = cluster_names[int(len(cluster_names) * proportion_source):len(cluster_names)]

        source_indices_mat = []
        for i in range(len(source_clusters)):
            source_indices_mat.append(true_toy_labels == source_clusters[i])
        source_indices = np.any(source_indices_mat, 0)
        toy_data_source = toy_data[:, source_indices]
        true_toy_labels_source = true_toy_labels[source_indices]

        target_indices_mat = []
        for i in range(len(target_clusters)):
            target_indices_mat.append(true_toy_labels == target_clusters[i])
        target_indices = np.any(target_indices_mat,0)
        toy_data_target = toy_data[:,target_indices]
        true_toy_labels_target = true_toy_labels[target_indices]

    else:
        print "Unknown mode!"
        toy_data_source=[]
        toy_data_target=[]
        true_toy_labels_source=[]
        true_toy_labels_target = []

    return toy_data_source, toy_data_target, true_toy_labels_source, true_toy_labels_target


if __name__ == "__main__":

    # Toy experiment parameters
    reps = 1  # 100

    # Data generation parameters
    num_genes = 5000  # 20000
    num_cells = 400  # 400
    true_num_clusters = 5  # 5
    dirichlet_parameter_cluster_size = 1  # 1, between 0 and inf, smaller values make cluster sizes more similar
    shape_power_law = 0.1  # 0.1, shape parameter of the power law -  between 0 and 1, the smaller this value the more extreme the power law
    upper_bound_counts = 1000000  # 1000000
    dirichlet_parameter_counts = 1  # 1, between 0 and inf, inverse noise parameter: smaller values make counts within cluster more similar (splitting the total
    # abundances in more equal parts for each cell)

    # Parameters for splitting data in source and target set
    proportion_source = 0.2  # How much of data will be source data? Not exact for mode 3 and 4, where the proportion is applied to clusters not cells.
    mode = 3  # 1 = split randomly, 2 = split randomly, but stratified, 3 = Have some overlapping and some exclusive clusters, 4 = have only non-overlapping clusters

    #  NMF parameters
    NMF_num_clusters = 10

    # Run toy experiments
    ARIs_SC3 = np.zeros(reps)
    ARIs_NMF = np.zeros(reps)
    ARIs_MTL_NMF = np.zeros(reps)

    for repetition in range(reps):
        print str(((np.float(repetition)+1)/reps)*100), "% calculated."
        # Generate toy data
        [toy_data, true_toy_labels] = generate_toy_data(num_genes=num_genes, num_cells=num_cells,num_clusters=true_num_clusters,
                                                        dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size, shape_power_law=shape_power_law,
                                                        upper_bound_counts=upper_bound_counts,dirichlet_parameter_counts=dirichlet_parameter_counts)
        # TSNE_plot(toy_data, true_toy_labels)

        # Split in source and target data
        toy_data_source, toy_data_target, true_toy_labels_source, true_toy_labels_target = split_source_target(toy_data, true_toy_labels, proportion_source, mode)

        # Run SC3 on target data
        # TODO
        # SC3_labels = SC3_clustering(toy_data_target, num_clusters=SC3_num_clusters)
        # ARIs_SC3[repetition] = adjusted_rand_score(true_toy_labels_target, SC3_labels)

        # Run NMF on target data
        NMF_labels = NMF_clustering(toy_data_target, num_clusters=NMF_num_clusters)
        ARIs_NMF[repetition] = adjusted_rand_score(true_toy_labels_target, NMF_labels)

        # Run MTL NMF on target data
        # TODO
        # MTL_NMF_labels = MTL_NMF_clustering(toy_data_target, toy_data_source, num_clusters=MTL_NMF_num_clusters)
        # ARIs_MTL_NMF[repetition] = adjusted_rand_score(true_toy_labels_target, MTL_NMF_labels)

    # print ARIs
    print "Mean ARI: ", str(np.round(np.mean(ARIs_SC3), decimals=3)), ", 95% Confidence Interval: ", str(np.round(sms.DescrStatsW(ARIs_SC3).tconfint_mean(), decimals=3))
    print "Mean ARI: ", str(np.round(np.mean(ARIs_NMF), decimals=3)), ", 95% Confidence Interval: ", str(np.round(sms.DescrStatsW(ARIs_NMF).tconfint_mean(), decimals=3))
    print "Mean ARI: ", str(np.round(np.mean(ARIs_MTL_NMF), decimals=3)), ", 95% Confidence Interval: ", str(np.round(sms.DescrStatsW(ARIs_MTL_NMF).tconfint_mean(), decimals=3))





