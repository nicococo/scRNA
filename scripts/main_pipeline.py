import matplotlib.pyplot as plt
from functools import partial

import sc3_pipeline_impl as sc
from sc3_pipeline import SC3Pipeline
from simulation import *
from mtl import *


def single_run(trg, trg_labels, src, src_labels, mix=0.5, n_cluster=4):
    cp = SC3Pipeline(trg)

    max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=1, num_expr_genes=2000))
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=1))

    cp.set_data_transformation(sc.data_transformation)
    # cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))
    cp.add_distance_calculation(partial(mtl_toy_distance, src_data=src, src_labels=src_labels,
                                        trg_labels=trg_labels, metric='euclidean', mixture=mix, nmf_k=n_cluster))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))

    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    ari = metrics.adjusted_rand_score(trg_labels, cp.cluster_labels)
    print 'ARI: ', ari
    return ari


def plot_results(fname):
    foo = np.load(fname)
    aris = foo['aris']
    mix = foo['mix']
    percs = foo['percs']

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
        # plt.plot(percs, res, '-', color=cols[:, m], linewidth=4)
        plt.errorbar(percs, res, res_stds, fmt='-', color=cols[:, m], linewidth=4, elinewidth=1)
        print 'Mix=', mix[m], ' res=', res, '   stds=', res_stds
        legend.append('Influence of src={0}%'.format(np.int(mix[m]*100.)))

    plt.xlabel('Fraction of target samples ('
               '1.0=600 samples)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim([4e-2, 1.3])
    plt.ylim([0., 1.])

    plt.legend(legend, loc=4)
    plt.semilogx()
    plt.show()


if __name__ == "__main__":
    flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    cluster_spec = [1, 2, [3, 4]]
    n_cluster = len(flatten(cluster_spec))
    print n_cluster

    fname = 'res_mtl_1v2.npz'
    plot_results(fname)

    n_trg = 600
    n_src = 600

    reps = 30
    mix = [0.0, 0.05, 0.1, 0.2]
    percs = np.logspace(-1.3, -0, 12)[[0,1,2,3,4,5,6,9,11]]
    #percs = [0.05, 0.075, 0.1, 0.2, 0.5, 1.0]

    aris = np.zeros((reps, len(percs), len(mix)))

    # np.random.seed(1)
    for r in range(reps):
        data, labels = generate_toy_data(num_genes=1000,
                                         num_cells= n_trg+n_src,
                                         cluster_spec=cluster_spec)
        print data.shape
        print len(labels)
        # convert labels to np.array
        labels = np.array(labels, dtype=np.int)
        inds = np.random.permutation(n_trg+n_src)

        for i in range(len(percs)):
            n_trg_perc = np.int(n_trg*percs[i])
            trg = data[:, inds[:n_trg_perc]].copy()
            trg_labels = labels[inds[:n_trg_perc]].copy()
            src = data[:, inds[n_trg:]].copy()
            src_labels = labels[inds[n_trg:]].copy()

            for m in range(len(mix)):
                aris[r, i, m] = single_run(trg, trg_labels, src, src_labels, mix=mix[m], n_cluster=n_cluster)

    # save the result and then plot
    np.savez(fname, aris=aris, percs=percs, reps=reps, mix=mix)
    plot_results(fname)

    print('Done.')