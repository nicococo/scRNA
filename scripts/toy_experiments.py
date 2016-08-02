import pdb
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import pandas as pd
import sklearn.decomposition as decomp
from sklearn.metrics import adjusted_rand_score
import statsmodels.stats.api as sms
from functools import partial

import sys
sys.path.append('../..')

from scRNA import sc3_pipeline_impl as sc
from scRNA.utils import *
import scRNA.mtl as mtl
from scRNA.sc3_pipeline import SC3Pipeline
from scRNA.simulation import generate_toy_data, split_source_target


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


def cell_filter(data, num_expr_genes=20, non_zero_threshold=2):
    res = np.sum(data >= non_zero_threshold, axis=0)
    return np.where(np.isfinite(res) & (res >= num_expr_genes))[0]


def gene_filter(data, perc_consensus_genes=0.98, non_zero_threshold=0):
    num_transcripts, num_cells = data.shape
    res_l = np.sum(data >= non_zero_threshold, axis=1)
    res_h = np.sum(data > 0, axis=1)
    lower_bound = np.float(num_cells)*(1.-perc_consensus_genes)
    upper_bound = np.float(num_cells)*perc_consensus_genes
    return np.where((res_l >= lower_bound) & (res_h <= upper_bound))[0]


def data_transformation(data):
    return np.log2(data + 1.)


def SC3_clustering(target_data, source_data, num_clusters=4):

    cp = SC3Pipeline(target_data)
    max_pca_comp = np.ceil(cp.num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells * 0.04).astype(np.int)

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=20))
    # cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.98, non_zero_threshold=0))

    cp.set_data_transformation(sc.data_transformation)
    cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=num_clusters))
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=num_clusters))
    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    SC3_labels = cp.cluster_labels

    return SC3_labels


def SC3_MTL_clustering(target_data, source_data, num_clusters=4, mixture=0.6):
    # SC3_MTL_num_clusters = 4 # 4, number of clusters for SC3_MTL
    # SC3_MTL_mixture_parameter = 0.6 # 0.6, Mixture of distance calculation, between 0 and 1, 0 = use only target data, 1 = use only source data
    cp = SC3Pipeline(target_data)

    max_pca_comp = np.ceil(cp.num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells * 0.04).astype(np.int)

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=20))
    # cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.98, non_zero_threshold=0))

    cp.set_data_transformation(sc.data_transformation)
    cp.add_distance_calculation(partial(mtl.mtl_toy_distance, src_data=source_data, metric='euclidean', mixture=mixture))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=num_clusters))
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=num_clusters))
    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    SC3_MTL_labels = cp.cluster_labels

    return SC3_MTL_labels


def NMF_clustering(data, num_clusters=4):
    #  Clustering parameters
    # NMF_num_clusters = 4 # 4, number of clusters for NMF

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


if __name__ == "__main__":

    # Toy experiment parameters
    reps = 20  # 10, number of repetitions

    # Data generation parameters
    num_genes = 100000  # 10000, number of genes
    # num_cells = 1000  # 1000, number of cells
    num_cells_source = 1000 # 1000
    target_sizes = [50, 100, 500, 1000, 5000,]  # [50, 100, 500, 1000, 5000, 10000]
    true_num_clusters = 4  # 4, number of clusters
    dirichlet_parameter_cluster_size = 10  # 10, Dirichlet parameter for cluster sizes, between 0 and inf, bigger values make cluster sizes more similar
    total_counts_mode = 3  # 3, How to generate the total counts, 1 = Power law, 2 = Negative Binomial Distribution, 3=simulation from Len et al

    # Parameters for splitting data in source and target set
    # proportion_target = 0.5 # 0.4, How much of the data will be target data? Not exact for mode 3 and 4, where the proportion is applied to clusters not cells.
    splitting_mode = 2  # 2, Splitting mode: 1 = split randomly, 2 = split randomly, but stratified, 3 = Have some overlapping and some exclusive clusters,
    # 4 = have only non-overlapping clusters

    # new Parameters from R code
    # gene_length = 1000  # 1000, assumed length of genes
    dirichlet_parameter_num_de_genes = 10  # 10, Dirichlet parameter for number of DE genes, between 0 and inf, bigger values numbers more similar
    gamma_shape = 2
    gamma_rate = 2
    # nb_dispersion = 0.1

    # Parameters of other simulations (not needed for total_counts_mode = 3)
    shape_power_law = 0.1  # 0.1, shape parameter of the power law -  between 0 and 1, the smaller this value the more extreme the power law
    upper_bound_counts = 1000000  # 1000000, upper bound for the total counts
    dirichlet_parameter_counts = 0.05  # 0.05, Dirichlet parameter for the individual counts, between 0 and inf (not too high - otherwise error),
    # inverse noise parameter: bigger values make counts within cluster more similar (splitting the total abundances in more equal parts for each cell)
    binomial_parameter = 1e-05  # 1e-05, parameter of the negative binomial distribution, between 0 and 1, the greater this value the more extreme the shape

    #  Clustering parameters
    NMF_num_clusters = 4  # 4, number of clusters for NMF
    SC3_num_clusters = 4  # 4, number of clusters for SC3
    SC3_MTL_num_clusters = 4  # 4, number of clusters for SC3_MTL
    # SC3_MTL_mixture_parameter = 0.1 # 0.6, Mixture of distance calculation, between 0 and 1, 0 = use only target data, 1 = use only source data
    SC3_MTL_mixtures = [0.1, 0.2, 0.5, 0.8]  # [0.1, 0.2, 0.5, 0.8]

    # Save directories
    fig_filename = 'simulation_results.png'
    npz_filename = 'simulation_results.npz'

    # Initialization
    SC3_ARI_means = np.zeros(len(target_sizes))
    SC3_ARI_errorbars = np.zeros(len(target_sizes))
    NMF_ARI_means = np.zeros(len(target_sizes))
    NMF_ARI_errorbars = np.zeros(len(target_sizes))
    SC3_MTL_ARI_means = np.zeros((len(target_sizes), len(SC3_MTL_mixtures)))
    SC3_MTL_ARI_errorbars = np.zeros((len(target_sizes), len(SC3_MTL_mixtures)))

    for target_size_index in range(len(target_sizes)):
        num_cells_target = target_sizes[target_size_index]
        # Run toy experiments
        ARIs_SC3 = np.zeros(reps)
        ARIs_NMF = np.zeros(reps)
        ARIs_SC3_MTL = np.zeros((reps, len(SC3_MTL_mixtures)))
        # ARIs_MTL_NMF = np.zeros(reps)

        num_cells = num_cells_source+num_cells_target
        proportion_target = float(num_cells_target)/num_cells

        for repetition in range(reps):
            print str(((np.float(repetition)+1)/reps)*100), "% of repetitions of target size ", str(num_cells_target), " done."
            # Generate toy data

            [toy_data, true_toy_labels] = generate_toy_data(num_genes=num_genes, num_cells=num_cells, num_clusters=true_num_clusters,
                                                            dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size, mode=total_counts_mode,
                                                            shape_power_law=shape_power_law, upper_bound_counts=upper_bound_counts,
                                                            dirichlet_parameter_counts=dirichlet_parameter_counts, binomial_parameter=binomial_parameter,
                                                            dirichlet_parameter_num_de_genes=dirichlet_parameter_num_de_genes, gamma_shape=gamma_shape,
                                                            gamma_rate=gamma_rate)

            # Split in source and target data
            toy_data_source, toy_data_target, true_toy_labels_source, true_toy_labels_target = split_source_target(toy_data, true_toy_labels, proportion_target, splitting_mode)
            # print "source data:", toy_data_source.shape
            # print "target data:", toy_data_target.shape

            # TSNE_plot(toy_data, true_toy_labels)
            # TSNE_plot(toy_data_source, true_toy_labels_source)
            # TSNE_plot(toy_data_target, true_toy_labels_target)

            # np.savetxt('SC3_labels.txt', SC3_labels, delimiter=' ')

            # Run SC3 on target data
            SC3_labels = SC3_clustering(toy_data_target, toy_data_source, SC3_num_clusters)
            ARIs_SC3[repetition] = adjusted_rand_score(true_toy_labels_target, SC3_labels)

            # Run SC3 with MTL distances
            for mixture_index in range(len(SC3_MTL_mixtures)):
                SC3_MTL_mixture_parameter = SC3_MTL_mixtures[mixture_index]
                SC3_MTL_labels = SC3_MTL_clustering(toy_data_target, toy_data_source, SC3_MTL_num_clusters, SC3_MTL_mixture_parameter)
                ARIs_SC3_MTL[repetition, mixture_index] = adjusted_rand_score(true_toy_labels_target, SC3_MTL_labels)

            # Run NMF on target data
            NMF_labels = NMF_clustering(toy_data_target, NMF_num_clusters)
            ARIs_NMF[repetition] = adjusted_rand_score(true_toy_labels_target, NMF_labels)

            # Run MTL NMF on target data
            # TODO
            # MTL_NMF_labels = MTL_NMF_clustering(toy_data_target, toy_data_source, num_clusters=MTL_NMF_num_clusters)
            # ARIs_MTL_NMF[repetition] = adjusted_rand_score(true_toy_labels_target, MTL_NMF_labels)

        SC3_ARI_means[target_size_index] = np.mean(ARIs_SC3)
        SC3_ARI_CIs = sms.DescrStatsW(ARIs_SC3).tconfint_mean()
        SC3_ARI_errorbars[target_size_index] = (SC3_ARI_CIs[1] - SC3_ARI_CIs[0])/2
        NMF_ARI_means[target_size_index] = np.mean(ARIs_NMF)
        NMF_ARI_CIs = sms.DescrStatsW(ARIs_NMF).tconfint_mean()
        NMF_ARI_errorbars[target_size_index] = (NMF_ARI_CIs[1] - NMF_ARI_CIs[0])/2
        SC3_MTL_ARI_means[target_size_index] = np.mean(ARIs_SC3_MTL, axis=0)
        SC3_MTL_ARI_CIs = sms.DescrStatsW(ARIs_SC3_MTL).tconfint_mean()
        SC3_MTL_ARI_errorbars[target_size_index] = (SC3_MTL_ARI_CIs[1] - SC3_MTL_ARI_CIs[0])/2

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Size of target dataset', fontsize=16)
    ax.set_ylabel('ARI with 95% errorbars', fontsize=16)
    ax.set_title("Clustering performances for varying target size", fontsize=20)
    ax.axis([0, max(target_sizes)+100, -0.1, 1.1])
    linestyle = {"linestyle":"--", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":10}
    linestyle2 = {"linestyle":":", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":10}
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(SC3_MTL_mixtures)+2)))
    c = next(color)
    ax.errorbar(target_sizes, SC3_ARI_means, yerr=SC3_ARI_errorbars, color=c, label='SC3', **linestyle)
    c = next(color)
    ax.errorbar(target_sizes, NMF_ARI_means, yerr=NMF_ARI_errorbars, color=c, label='NMF', **linestyle)

    for mixture_index in range(len(SC3_MTL_mixtures)):
        c = next(color)
        ax.errorbar(target_sizes, SC3_MTL_ARI_means[:, mixture_index], yerr=NMF_ARI_errorbars[mixture_index], color=c, label='SC3_MTL with mixture ' + str(SC3_MTL_mixtures[mixture_index]),
                    **linestyle2)
    ax.legend(loc='best', fontsize='12')
    fig.savefig(fig_filename)
    np.savez(npz_filename, SC3_ARI_means=SC3_ARI_means, SC3_ARI_errorbars=SC3_ARI_errorbars, NMF_ARI_means=NMF_ARI_means, NMF_ARI_errorbars=NMF_ARI_errorbars,
             SC3_MTL_ARI_means=SC3_MTL_ARI_means, SC3_MTL_ARI_errorbars=SC3_MTL_ARI_errorbars, SC3_MTL_mixtures=SC3_MTL_mixtures,
             SC3_MTL_num_clusters=SC3_MTL_num_clusters, SC3_num_clusters=SC3_num_clusters, NMF_num_clusters=NMF_num_clusters, gamma_rate=gamma_rate,
             gamma_shape=gamma_shape, dirichlet_parameter_num_de_genes=dirichlet_parameter_num_de_genes, splitting_mode=splitting_mode,
             total_counts_mode=total_counts_mode, dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size, true_num_clusters=true_num_clusters,
             target_sizes=target_sizes, num_cells_source=num_cells_source, num_genes=num_genes, reps=reps)
    plt.show()
    pdb.set_trace()

