import numpy as np
import pdb
from sklearn.cross_validation import train_test_split


def generate_toy_data(num_genes=10000, num_cells=1000, num_clusters=4, dirichlet_parameter_cluster_size=10, mode=3, shape_power_law=0.1, upper_bound_counts=1000000,
                      dirichlet_parameter_counts=0.05, binomial_parameter=1e-05, dirichlet_parameter_num_de_genes=10, gamma_shape=2, gamma_rate=2):
    # Toy experiment parameters
    # Data generation parameters
    # num_genes = 10000  # 10000, number of genes
    # num_cells = 1000  # 1000, number of cells
    # num_clusters = 4  # 4, number of clusters
    # dirichlet_parameter_cluster_size = 10  # 10, Dirichlet parameter for cluster sizes, between 0 and inf, bigger values make cluster sizes more similar
    # dirichlet_parameter_num_de_genes = 10 # 10, Dirichlet parameter for the number of of DE genes per cluster (big)
    # total_counts_mode = 3 # 3, How to generate the total counts, 1 = Power law, 2 = Negative Binomial Distribution, 3 = simulation from Len et al.
    # shape_power_law = 0.1  # 0.1, shape parameter of the power law -  between 0 and 1, the smaller this value the more extreme the power law
    # upper_bound_counts = 1000000  # 1000000, upper bound for the total counts
    # dirichlet_parameter_counts = 0.05  # 0.05, Dirichlet parameter for the individual counts, between 0 and inf (not too high - otherwise error),
    # inverse noise parameter: bigger values make counts within cluster more similar (splitting the total abundances in more equal parts for each cell)
    # binomial_parameter = 1e-05 # 1e-05, parameter of the negative binomial distribution, between 0 and 1, the greater this value the more extreme the shape

    # Generate Cluster sizes
    cluster_sizes = np.squeeze(np.round(np.random.dirichlet(np.ones(num_clusters) * dirichlet_parameter_cluster_size, size=1) * (num_cells - num_clusters))) + 1
    while min(cluster_sizes) == 1:
        cluster_sizes = np.squeeze(np.round(np.random.dirichlet(np.ones(num_clusters) * dirichlet_parameter_cluster_size, size=1) * (num_cells - num_clusters))) + 1

    if np.sum(cluster_sizes) != num_cells:
        cluster_sizes[0] = cluster_sizes[0] - (np.sum(cluster_sizes) - num_cells)
    assert min(cluster_sizes) > 1

    # Generate data for each cluster
    data_complete = []
    labels_now = []

    # Simulation from Len et al.
    if mode == 3:
        
        nb_dispersion = 0.1
        
        #Define the extent (effect size of DE per cluster) units are log2 FC
        de_logfc = np.random.choice([1, 2], num_clusters, replace=True)
        
        #Define the number of genes DE in each cluster and then their indices
        nde = np.squeeze(
        np.round(np.random.dirichlet(np.ones(num_clusters) * dirichlet_parameter_num_de_genes, size=1) * (0.6 * num_genes))) + 1 #Why +1?
        nde_cumsum = np.cumsum(nde).astype(int)
        nde_cumsum_zero = np.insert(nde_cumsum, 0, 0).astype(int)
        
        #Define the 'true' population mean expression levels
        true_means = np.random.gamma(gamma_shape, scale=1 / float(gamma_rate), size=num_genes)

        for cluster_ind in range(num_clusters):

            # Draw samples from
            all_facs = np.power(2,np.random.normal(loc=0, scale=0.5, size=cluster_sizes[cluster_ind]))
            effective_means = np.outer(true_means, all_facs)

            chosen = range(nde_cumsum_zero[cluster_ind], nde_cumsum[cluster_ind])
            up_down = np.sign(np.random.normal(size=len(chosen)))
            effective_means[chosen, ] = np.transpose(np.multiply(np.transpose(effective_means[chosen, ]), np.power(2, (de_logfc[cluster_ind] * up_down))))

            # Generate data
            sample = np.random.negative_binomial(p=(1/nb_dispersion)/((1/nb_dispersion)+effective_means), n=1/nb_dispersion, size=[num_genes, cluster_sizes[cluster_ind]])

            # the abundance plot
            # plt.plot(np.sort(np.sum(sample, axis=1))[::-1], 'o')
            # plt.show()

            data_complete.append(np.asarray(sample))
            labels_now.append(np.tile(cluster_ind + 1, cluster_sizes[cluster_ind]))

        data = np.squeeze(np.hstack(data_complete))
        labels = np.hstack(labels_now)


    # Our simulation
    else:
        for cluster_ind in range(num_clusters):

            # Draw samples from the power law distribution or negative binomial
            # pdb.set_trace()
            if mode == 1:
                sample = np.round(np.random.power(shape_power_law, num_genes) * upper_bound_counts)
            elif mode == 2:
                sample = np.random.negative_binomial(1, binomial_parameter, num_genes)
            else:
                print "Wrong mode!"

            # the abundance plot
            # plt.plot(np.sort(sample)[::-1], 'o')
            # plt.plot(np.sort(sample_nb)[::-1], 'o')
            # plt.show()

            # Generate data
            data_now = []
            for i in range(num_genes):
                row_total = int(sample[i])
                one_gene = np.round(np.random.dirichlet(np.ones(cluster_sizes[cluster_ind]) * dirichlet_parameter_counts, size=1) * row_total)
                data_now.append(one_gene)

            data_complete.append(np.squeeze(np.asarray(data_now)))
            labels_now.append(np.tile(cluster_ind+1, cluster_sizes[cluster_ind]))

        data = np.squeeze(np.hstack(data_complete))
        labels = np.hstack(labels_now)
    return [data, labels]


def split_source_target(toy_data, true_toy_labels, proportion_target=0.4, mode=2):
    # Parameters for splitting data in source and target set
    # proportion_target = 0.4 # 0.4, How much of the data will be target data? Not exact for mode 3 and 4, where the proportion is applied to clusters not cells.
    # splitting_mode = 2  # 2, Splitting mode: 1 = split randomly, 2 = split randomly, but stratified, 3 = Have some overlapping and some exclusive clusters,
    # 4 = have only non-overlapping clusters
    if mode == 1:
        toy_data_source_now, toy_data_target_now, true_toy_labels_source, true_toy_labels_target = train_test_split(np.transpose(toy_data), true_toy_labels,
                                                                                                            test_size=proportion_target)
        toy_data_source = np.transpose(toy_data_source_now)
        toy_data_target = np.transpose(toy_data_target_now)

    elif mode == 2:
        toy_data_source_now, toy_data_target_now, true_toy_labels_source, true_toy_labels_target = train_test_split(np.transpose(toy_data), true_toy_labels,
                                                                                                            test_size=proportion_target, stratify=true_toy_labels)
        toy_data_source = np.transpose(toy_data_source_now)
        toy_data_target = np.transpose(toy_data_target_now)

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
                                                                                                                 true_toy_labels_random, test_size=proportion_target)

        # Combine exclusive and random data
        toy_data_source = np.concatenate((toy_data_source_exclusive, np.transpose(toy_data_source_random)), axis=1)
        toy_data_target = np.concatenate((toy_data_target_exclusive, np.transpose(toy_data_target_random)), axis=1)
        true_toy_labels_source = np.concatenate((true_toy_labels_source_exclusive, np.transpose(true_toy_labels_source_random)))
        true_toy_labels_target = np.concatenate((true_toy_labels_target_exclusive, np.transpose(true_toy_labels_target_random)))

        # toy_data_rebuilt = np.concatenate((toy_data_source, toy_data_target), axis=1)
        # toy_labels_rebuilt = np.concatenate((true_toy_labels_source, true_toy_labels_target), axis=0)
        # pdb.set_trace()
        # TSNE_plot(toy_data_rebuilt, toy_labels_rebuilt)
        # TSNE_plot(toy_data, true_toy_labels)
        # TSNE_plot(toy_data_source, true_toy_labels_source)
        # TSNE_plot(toy_data_target, true_toy_labels_target)

    elif mode == 4:
        proportion_source = 1-proportion_target
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