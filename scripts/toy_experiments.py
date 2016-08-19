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


def SC3_clustering(target_data, num_clusters=4):

    cp = SC3Pipeline(target_data)
    max_pca_comp = np.ceil(cp.num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells * 0.04).astype(np.int)

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=20))
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=1))

    # cp.set_data_transformation(sc.data_transformation)
    cp.add_distance_calculation(partial(sc.distances, metric='pearson'))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=num_clusters))
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=num_clusters))
    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    SC3_labels = cp.cluster_labels
    return SC3_labels


def SC3_original_clustering(target_data, num_clusters=4):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # doParallel = importr('doParallel', lib_loc="C:/Users/Bettina/Documents/R/win-library/3.3")
    # base = importr('base')
    # print(base._libPaths())
    # base._libPaths("C:/Users/Bettina/Documents/R/win-library/3.3")
    # print(base._libPaths())
    SC3_original = importr('SC3')
    # doParallel = importr('doParallel')

    data_as_row = np.reshape(np.transpose(target_data), newshape=(1, -1))
    data_in_r_as_row = robjects.IntVector(data_as_row[0])
    data_in_r = robjects.r['matrix'](data_in_r_as_row, nrow=target_data.shape[0])
    data_in_r.colnames = robjects.StrVector(map(str, np.array(range(target_data.shape[1]))))
    data_in_r.rownames = robjects.StrVector(map(str, np.array(range(target_data.shape[0]))))

    SC3_original.sc3(data_in_r, ks=num_clusters, interactivity=False,
                     cell_filter=False, gene_filter=False, log_scale=False)

    robjects.r('d <- sc3.interactive.arg$cons.table')
    robjects.r('res <- d[d[,1] == "pearson" & d[,2] == "PCA" & d[,3] == "{0}"]'.format(num_clusters))
    robjects.r('clust_res <- res[[4]]')
    robjects.r('hc <- clust_res[[3]]')
    robjects.r('clusts <- cutree(hc, {0})'.format(num_clusters))

    clusts = robjects.r('clusts')
    SC3_original_labels = np.asarray(clusts)

    # R statements to convert:
    # d <- sc3.interactive.arg$cons.table
    # res <- d[1,]    % bzw. res <- d[d[,1] == "euclidean" & d[,2] == "PCA" & d[,3] == "4"]
    # clust_res <- res[[4]]
    # hc <- clust.res[[3]]
    # clusts <- cutree(hc, 4) % bzw. 4 = num_clusters

    return SC3_original_labels


def SC3_MTL_clustering(target_data, source_data, num_clusters=4, mixture=0.6, nmf_k=4):
    # SC3_MTL_num_clusters = 4 # 4, number of clusters for SC3_MTL
    # SC3_MTL_mixture_parameter = 0.6 # 0.6, Mixture of distance calculation, between 0 and 1, 0 = use only target data, 1 = use only source data
    cp = SC3Pipeline(target_data)

    max_pca_comp = np.ceil(cp.num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells * 0.04).astype(np.int)

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=20))
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=1))

    # cp.set_data_transformation(sc.data_transformation)
    cp.add_distance_calculation(partial(mtl.mtl_toy_distance, src_data=source_data, metric='pearson', mixture=mixture, nmf_k=nmf_k))

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
        nmf = decomp.NMF(alpha=1., init='nndsvdar', l1_ratio=0.5, max_iter=1000, n_components=num_clusters, random_state=0, shuffle=True, solver='cd', tol=0.00001,
                         verbose=0)
        nmf.fit_transform(X)
        H = nmf.components_
        labels[remain_inds_cells] = np.argmax(H, axis=0)

    return labels


if __name__ == "__main__":
    # Toy experiment parameters
    reps = 10  # 50, number of repetitions

    # Data generation parameters
    num_genes = 500  # 5000, number of genes
    # num_cells = 1000  # 1000, number of cells
    num_cells_source = 300  # 500
    target_sizes = [50, 100, 200, 300]  # [50, 100, 250, 500]
    # true_num_clusters = 4  # 4, number of clusters
    cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]
    dirichlet_parameter_cluster_size = 10  # 10, Dirichlet parameter for cluster sizes, between 0 and inf, bigger values make cluster sizes more similar
    # total_counts_mode = 3  # 3, How to generate the total counts, 1 = Power law, 2 = Negative Binomial Distribution, 3=simulation from Len et al

    # Parameters for splitting data in source and target set
    # proportion_target = 0.5 # 0.4, How much of the data will be target data? Not exact for mode 3 and 4, where the proportion is applied to clusters not cells.
    splitting_mode = 2
    # Splitting mode: 1 = split randomly,
    #                 2 = split randomly, but stratified,
    #                 3 = split randomly, but anti-stratified [not implemented]
    #                 4 = Have some overlapping and some exclusive clusters,
    #                 5 = have only non-overlapping clusters
    #                 6 = Define source matrix clusters
    source_clusters = None  # for splitting_mode=6

    # new Parameters from R code
    # gene_length = 1000  # 1000, assumed length of genes
    # dirichlet_parameter_num_de_genes = 10  # 10, Dirichlet parameter for number of DE genes, between 0 and inf, bigger values numbers more similar
    gamma_shape = 2
    gamma_rate = 2
    # nb_dispersion = 0.1

    # Parameters of other simulations (not needed for total_counts_mode = 3)
    # shape_power_law = 0.1  # 0.1, shape parameter of the power law -  between 0 and 1, the smaller this value the more extreme the power law
    # upper_bound_counts = 1000000  # 1000000, upper bound for the total counts
    # dirichlet_parameter_counts = 0.05  # 0.05, Dirichlet parameter for the individual counts, between 0 and inf (not too high - otherwise error),
    # inverse noise parameter: bigger values make counts within cluster more similar (splitting the total abundances in more equal parts for each cell)
    # binomial_parameter = 1e-05  # 1e-05, parameter of the negative binomial distribution, between 0 and 1, the greater this value the more extreme the shape

    #  Clustering parameters
    num_clusters = 8  # 4, number of clusters
    k_nmf = 8
    # SC3_MTL_mixture_parameter = 0.1 # 0.6, Mixture of distance calculation, between 0 and 1, 0 = use only target data, 1 = use only source data
    SC3_MTL_mixtures = [0.1, 0.5, 0.9]  # [0.1, 0.2, 0.5, 0.8]

    # Save directories
    fig_filename = 'simulation_results.png'
    npz_filename = 'simulation_results.npz'

    # Run toy experiments
    ARIs_SC3 = np.zeros((reps, len(target_sizes)))
    ARIs_SC3_original = np.zeros((reps, len(target_sizes)))
    ARIs_NMF = np.zeros((reps, len(target_sizes)))
    ARIs_SC3_MTL = np.zeros((reps, len(target_sizes), len(SC3_MTL_mixtures)))
    # ARIs_MTL_NMF = np.zeros((reps, len(target_sizes), len(SC3_MTL_mixtures)))
    for repetition in range(reps):
        print str(((np.float(repetition) + 1) / reps) * 100), "% of repetitions done."
        # Generate toy data
        [toy_data, true_toy_labels] = generate_toy_data(num_genes=num_genes, num_cells=max(target_sizes)+num_cells_source, cluster_spec=cluster_spec,
                                                        dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size,
                                                        gamma_shape=gamma_shape, gamma_rate=gamma_rate)
        # Split in source and target data
        toy_data_source, toy_data_target, true_toy_labels_source, true_toy_labels_target = split_source_target(toy_data=toy_data, true_toy_labels=true_toy_labels,
                                                                                                               target_ncells=max(target_sizes),
                                                                                                               source_ncells=num_cells_source,
                                                                                                               mode=splitting_mode, source_clusters=source_clusters,
                                                                                                               noise_target=False, noise_sd=0.5)
        inds = np.random.permutation(np.max(target_sizes))-1

        # Use perfect number of latent states for nmf and sc3
        if not splitting_mode == 2:
            num_cluster = np.unique(true_toy_labels_target).size
            k_nmf = np.unique(true_toy_labels_source).size

        for target_size_index in range(len(target_sizes)):
            num_cells_target = target_sizes[target_size_index]

            target_data = toy_data_target[:, inds[:num_cells_target]].copy()
            target_labels = np.array(true_toy_labels_target)[inds[:num_cells_target]].copy()

            # print "source data:", toy_data_source.shape
            # print "target data:", toy_data_target.shape

            # TSNE_plot(toy_data, true_toy_labels)
            # TSNE_plot(toy_data_source, true_toy_labels_source)
            # TSNE_plot(toy_data_target, true_toy_labels_target)

            # np.savetxt('SC3_labels.txt', SC3_labels, delimiter=' ')

            # Run SC3 on target data
            SC3_labels = SC3_clustering(target_data, num_clusters)
            SC3_original_labels = SC3_original_clustering(target_data, num_clusters)
            ARIs_SC3[repetition, target_size_index] = adjusted_rand_score(target_labels, SC3_labels)
            ARIs_SC3_original[repetition, target_size_index] = adjusted_rand_score(target_labels, SC3_original_labels)

            # Run SC3 with MTL distances
            for mixture_index in range(len(SC3_MTL_mixtures)):
                SC3_MTL_mixture_parameter = SC3_MTL_mixtures[mixture_index]
                SC3_MTL_labels = SC3_MTL_clustering(target_data, toy_data_source, num_clusters, SC3_MTL_mixture_parameter, k_nmf)
                ARIs_SC3_MTL[repetition, target_size_index, mixture_index] = adjusted_rand_score(target_labels, SC3_MTL_labels)

            # Run NMF on target data
            NMF_labels = NMF_clustering(target_data, num_clusters)
            ARIs_NMF[repetition, target_size_index] = adjusted_rand_score(target_labels, NMF_labels)

            # Run MTL NMF on target data
            # TODO
            # MTL_NMF_labels = MTL_NMF_clustering(toy_data_target, toy_data_source, num_clusters=MTL_NMF_num_clusters)
            # ARIs_MTL_NMF[repetition] = adjusted_rand_score(true_toy_labels_target, MTL_NMF_labels)

        #np.savez(npz_filename, ARIs_SC3=ARIs_SC3, ARIs_NMF=ARIs_NMF, ARIs_SC3_MTL=ARIs_SC3_MTL, SC3_MTL_mixtures=SC3_MTL_mixtures,
        #         num_clusters=num_clusters, k_nmf=k_nmf, gamma_rate=gamma_rate, gamma_shape=gamma_shape, splitting_mode=splitting_mode,
        #         dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size, cluster_spec=cluster_spec,
        #         target_sizes=target_sizes, num_cells_source=num_cells_source, num_genes=num_genes, reps=reps)

    SC3_ARI_means = np.mean(ARIs_SC3, axis=0)
    SC3_ARI_CIs = sms.DescrStatsW(ARIs_SC3).tconfint_mean()
    SC3_ARI_errorbars = (SC3_ARI_CIs[1] - SC3_ARI_CIs[0])/2
    SC3_original_ARI_means = np.mean(ARIs_SC3_original, axis=0)
    SC3_original_ARI_CIs = sms.DescrStatsW(ARIs_SC3_original).tconfint_mean()
    SC3_original_ARI_errorbars = (SC3_original_ARI_CIs[1] - SC3_original_ARI_CIs[0]) / 2
    NMF_ARI_means = np.mean(ARIs_NMF, axis=0)
    NMF_ARI_CIs = sms.DescrStatsW(ARIs_NMF).tconfint_mean()
    NMF_ARI_errorbars = (NMF_ARI_CIs[1] - NMF_ARI_CIs[0])/2
    SC3_MTL_ARI_means = np.mean(ARIs_SC3_MTL, axis=0)
    SC3_MTL_ARI_errorbars = np.zeros((len(target_sizes), len(SC3_MTL_mixtures)))
    for mixture_index in range(len(SC3_MTL_mixtures)):
        SC3_MTL_ARI_CIs = sms.DescrStatsW(ARIs_SC3_MTL[:, :, mixture_index]).tconfint_mean()
        SC3_MTL_ARI_errorbars[:, mixture_index] = (SC3_MTL_ARI_CIs[1] - SC3_MTL_ARI_CIs[0])/2

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Size of target dataset', fontsize=16)
    ax.set_ylabel('ARI with 95% errorbars', fontsize=16)
    ax.set_title("Clustering performances for varying target size", fontsize=20)
    ax.axis([0, max(target_sizes)+100, -0.1, 1.1])
    linestyle = {"linestyle":"--", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":10}
    linestyle2 = {"linestyle":":", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":10}
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(SC3_MTL_mixtures)+3)))
    c = next(color)
    ax.errorbar(target_sizes, SC3_ARI_means, yerr=SC3_ARI_errorbars, color=c, label='SC3', **linestyle)
    c = next(color)
    ax.errorbar(target_sizes, SC3_original_ARI_means, yerr=SC3_original_ARI_errorbars, color=c, label='SC3 original', **linestyle)
    c = next(color)
    ax.errorbar(target_sizes, NMF_ARI_means, yerr=NMF_ARI_errorbars, color=c, label='NMF', **linestyle)

    for mixture_index in range(len(SC3_MTL_mixtures)):
        c = next(color)
        ax.errorbar(target_sizes, SC3_MTL_ARI_means[:, mixture_index], yerr=SC3_MTL_ARI_errorbars[:,mixture_index], color=c, label='SC3_MTL with mixture ' +
                                                                                                                                   str(SC3_MTL_mixtures[mixture_index]),
                    **linestyle2)
    ax.legend(loc='best', fontsize='12')
    #fig.savefig(fig_filename)
    #np.savez(npz_filename, SC3_ARI_means=SC3_ARI_means, SC3_ARI_errorbars=SC3_ARI_errorbars, NMF_ARI_means=NMF_ARI_means, NMF_ARI_errorbars=NMF_ARI_errorbars,
    #         SC3_MTL_ARI_means=SC3_MTL_ARI_means, SC3_MTL_ARI_errorbars=SC3_MTL_ARI_errorbars, SC3_MTL_mixtures=SC3_MTL_mixtures,
    #         num_clusters=num_clusters, k_nmf=k_nmf, gamma_rate=gamma_rate,
    #         gamma_shape=gamma_shape, splitting_mode=splitting_mode, dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size, cluster_spec=cluster_spec,
    #         target_sizes=target_sizes, num_cells_source=num_cells_source, num_genes=num_genes, reps=reps)
    plt.show()

# To save the last rep as an example dataset as npz
# SC3_ARI = adjusted_rand_score(true_toy_labels_target, SC3_labels)
# NMF_ARI = adjusted_rand_score(true_toy_labels_target, NMF_labels)
# np.savez('example_toy_data.npz', toy_data_target=toy_data_target, true_toy_labels_target=true_toy_labels_target, SC3_labels=SC3_labels, NMF_labels=NMF_labels,
# NMF_num_clusters=NMF_num_clusters, SC3_num_clusters=SC3_num_clusters, SC3_ARI=SC3_ARI, NMF_ARI=NMF_ARI)

# Load npz data and save as tsv
# data=np.load('example_toy_data.npz')
# np.savetxt('example_toy_data.tsv', data["toy_data_target"], fmt='%u', delimiter='\t')
# np.savetxt('example_toy_data_labels.tsv', data["true_toy_labels_target"], fmt='%u', delimiter='\t')
# np.savetxt('example_toy_data_SC3labels.tsv', data["SC3_labels"], fmt='%u', delimiter='\t')



