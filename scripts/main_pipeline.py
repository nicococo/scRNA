import matplotlib.pyplot as plt
from functools import partial

import sc3_pipeline_impl as sc
from sc3_pipeline import SC3Pipeline
from toy_experiments import SC3_original_clustering
from simulation import *
from mtl import *


def single_run(trg, trg_labels, src, src_labels, mix=0.5, n_cluster=4, k_nmf=4, consensus_mode=0):
    cp = SC3Pipeline(trg)

    max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=1, num_expr_genes=2000))
    # cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=1))
    # cp.set_data_transformation(sc.data_transformation)
    # cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))
    cp.add_distance_calculation(partial(mtl_toy_distance, src_data=src, src_labels=src_labels,
                                        trg_labels=trg_labels, metric='euclidean', mixture=mix, nmf_k=k_nmf))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp+1, method='pca'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))

    cp.apply(pc_range=[min_pca_comp, max_pca_comp], consensus_mode=consensus_mode)

    ari = metrics.adjusted_rand_score(trg_labels, cp.cluster_labels)
    print 'ARI: ', ari
    return ari


def plot_results(fname):
    foo = np.load(fname)
    aris = foo['aris']
    mix = foo['mix']
    percs = foo['percs']
    n_trg = foo['n_trg']

    plt.figure(1)
    np.random.seed(8)
    cols = np.random.rand(3, len(mix))
    cols[:, 0] = cols[:, 0] / np.max(cols[:, 0]) * 0.3
    for i in range(len(mix)-1):
        cols[:, i+1] = cols[:, i+1] / np.max(cols[:, i+1]) * np.max( [(0.4 + np.float(i)*0.1), 1.0] )

    legend = []
    for m in range(len(mix)):
        res = np.mean(aris[:,:,m], axis=0)
        res_stds = np.std(aris[:,:,m], axis=0)

        if m > 0:
            plt.plot(percs, res, '-', color=cols[:, m], linewidth=4)
            plt.errorbar(percs, res, res_stds, fmt='-', color=cols[:, m], linewidth=4, elinewidth=1)
        else:
            plt.plot(percs, res, '--', color=cols[:, m], linewidth=4)
            plt.errorbar(percs, res, res_stds, fmt='--', color=cols[:, m], linewidth=4, elinewidth=1)

        print 'Mix=', mix[m], ' res=', res, '   stds=', res_stds
        legend.append('Influence of src={0}%'.format(np.int(mix[m]*100.)))

    plt.xlabel('Fraction of target samples ('
               '1.0={0} samples)'.format(n_trg), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim([4e-2, 1.3])
    plt.ylim([0., 1.])

    plt.legend(legend, loc=4)
    plt.semilogx()
    plt.show()


def main(mode=2, reps=10, cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]]):
    flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    n_cluster = len(flatten(cluster_spec))
    k_nmf = n_cluster
    print 'Number of cluster is ', n_cluster

    fname = 'res_mtl_m{0}_r{1}.npz'.format(mode, reps)

    n_trg = 1000
    n_src = 1000

    mix = [0.0, 0.05, 0.1, 0.2, 0.3, 1.0]
    percs = np.logspace(-1.3, -0, 12)[[0,1,2,3,4,5,6,9,11]]
    aris = np.zeros((reps, len(percs), len(mix)))

    mix = [-2.0, -1.0, 0.0, 0.25, 0.99, 1.0]
    # percs = np.array([0.5, 1.0])
    percs = np.logspace(-1.3, -0, 12)[[2,3,4,5,6,9,11]]
    aris = np.zeros((reps, len(percs), len(mix)))

    np.random.seed(1)
    for r in range(reps):
        # 1. Generate scRNA data
        data, labels = generate_toy_data(num_genes=1000,
                                         num_cells= n_trg+n_src,
                                         cluster_spec=cluster_spec)
        # 2. Split source and target according to specified mode/setting
        src, trg, src_labels, trg_labels = split_source_target(data, labels,
                                            target_ncells=n_trg, source_ncells=n_src,
                                            mode=mode, source_clusters = None,
                                            noise_target=False, noise_sd=0.0)
        trg_labels = np.array(trg_labels, dtype=np.int)
        src_labels = np.array(src_labels, dtype=np.int)
        # 3.a. Subsampling order for target
        inds = np.random.permutation(trg_labels.size)

        # 3.b. Use perfect number of latent states for nmf and sc3
        n_cluster = np.unique(trg_labels).size
        k_nmf = np.unique(src_labels).size

        # 3.c. Target data subsampling loop
        for i in range(len(percs)):
            n_trg_perc = np.int(n_trg*percs[i])

            p_trg = trg[:, inds[:n_trg_perc]].copy()
            p_trg_labels = trg_labels[inds[:n_trg_perc]].copy()
            # 4. MTL/DA mixing parameter loop
            for m in range(len(mix)):
                if np.int(mix[m]) >= 0:
                    aris[r, i, m] = single_run(p_trg.copy(), p_trg_labels.copy(), src.copy(), src_labels.copy(),
                                               mix=mix[m], n_cluster=n_cluster, k_nmf=k_nmf)
                elif np.int(mix[m]) == -1:
                    # lbls = SC3_original_clustering(p_trg.copy(), num_clusters=n_cluster)
                    # aris[r, i, m] = metrics.adjusted_rand_score(p_trg_labels, lbls)
                    aris[r, i, m] = single_run(p_trg.copy(), p_trg_labels.copy(), src.copy(), src_labels.copy(),
                                               mix=0.0, n_cluster=n_cluster, k_nmf=k_nmf, consensus_mode=1)
                elif np.int(mix[m]) == -2:
                    lbls = SC3_original_clustering(p_trg.copy(), num_clusters=n_cluster)
                    aris[r, i, m] = metrics.adjusted_rand_score(p_trg_labels, lbls)

    # save the result and then plot
    np.savez(fname, aris=aris, percs=percs, reps=reps, mix=mix, n_src=n_src, n_trg=n_trg)
    plot_results(fname)
    print('Done.')


if __name__ == "__main__":
    main(mode=1, reps=15, cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]])

    # fname = 'res_mtl_m2_r10.npz'
    # plot_results(fname)

    # np.random.seed(2)
    # data, labels = generate_toy_data(num_genes=1000,
    #                                  num_cells= 1000,
    #                                  cluster_spec=[1,2,3,4])
    #
    # src, trg, src_labels, trg_labels = split_source_target(data, labels,
    #                                         target_ncells=80, source_ncells=500,
    #                                         mode=4, source_clusters = None,
    #                                         noise_target=False, noise_sd=0.1)
    # trg_labels = np.array(trg_labels, dtype=np.int)
    # src_labels = np.array(src_labels, dtype=np.int)
    #
    # print 'Src: ', np.unique(src_labels), src_labels.size
    # print 'Trg: ', np.unique(trg_labels), trg_labels.size
    #
    # ari = single_run(trg, trg_labels, src, src_labels, mix=0.5, n_cluster=4)
    #
    # print 'Result: ', ari
