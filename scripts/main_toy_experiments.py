import matplotlib.pyplot as plt
from functools import partial

import sc3_pipeline_impl as sc
from sc3_pipeline import SC3Pipeline
from toy_experiments import SC3_original_clustering
from da_nmf_pipeline import DaNmfPipeline
from simulation import *
from mtl import *


def method_sc3(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster,
               mix=0.5, consensus_mode=0, metric='euclidean'):
    num_cells = trg.shape[1]
    max_pca_comp = np.ceil(num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(num_cells * 0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    cp = SC3Pipeline(trg, pc_range=[min_pca_comp, max_pca_comp],
                     consensus_mode=consensus_mode, sub_sample=True)
    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=1, num_expr_genes=2000))
    # cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=1))
    # cp.set_data_transformation(sc.data_transformation)
    # cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))
    cp.add_distance_calculation(partial(mtl_toy_distance, src_data=src, src_labels=src_labels,
                                        trg_labels=trg_labels, metric=metric, mixture=mix, nmf_k=n_src_cluster))
    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_trg_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_trg_cluster))
    cp.apply()

    desc = 'SC3'
    if mix > 0.0:
        desc = 'DA-SC3 ({0}%, {1})'.format(np.int(mix*100.), metric[:3])
    return desc, cp.cluster_labels, None


def method_da_nmf(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster, use_strat=False):
    ids = np.arange(src.shape[0])
    cp = DaNmfPipeline(src, ids, trg, ids.copy())
    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=1, num_expr_genes=2000))
    # cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=1))
    # cp.set_data_transformation(sc.data_transformation)
    cp.apply(trg_k=n_trg_cluster, src_k=n_src_cluster)
    lbls = cp.cluster_labels
    desc = 'DA-NMF'
    if use_strat:
        trg_lbls = cp.trg_standalone_cluster_labels
        bin_lbls = cp.trg_reject_option[1][1]
        inds = np.where(bin_lbls == -1)[0]
        lbls[inds] = trg_lbls[inds]
        desc = 'DA-NMF*'
    return desc, lbls, cp.trg_reject_option


def plot_percs_accs(fname):
    foo = np.load(fname)
    aris = foo['aris']
    methods = foo['methods']
    desc = foo['desc']
    percs = foo['percs']
    n_trg = foo['n_trg']
    n_src = foo['n_src']

    plot_single(1, 'Acc', aris, methods, percs, n_src, n_trg, desc)

    if 'aris_strat' in foo:
        aris_strat = foo['aris_strat']
        num_strat = foo['num_strat']
        plot_single(2, 'Acc (strat)', aris_strat, methods, percs, n_src, n_trg, desc)
        plot_single(3, '#Samples in Overlapping Cluster', num_strat, methods, percs, n_src, n_trg, desc)

    if 'aucs_nmf' in foo:
        aucs_nmf = foo['aucs_nmf']
        aucs_nmf_desc = foo['aucs_nmf_desc']
        if len(aucs_nmf.shape) > 0:
            plot_single(4, 'Acc Reject Option', aucs_nmf, aucs_nmf_desc, percs, n_src, n_trg, aucs_nmf_desc)

    plt.show()


def plot_single(fig_num, title, aris, methods, percs, n_src, n_trg, desc):
    plt.figure(fig_num)
    plt.subplot(1, 2, 1)
    np.random.seed(8)
    cols = np.random.rand(3, len(methods))
    cols[:, 0] = cols[:, 0] / np.max(cols[:, 0]) * 0.3
    for i in range(len(methods) - 1):
        cols[:, i + 1] = cols[:, i + 1] / np.max(cols[:, i + 1]) * np.max([(0.2 + np.float(i) * 0.1), 1.0])

    legend = []
    aucs = np.zeros(len(methods))
    for m in range(len(methods)):
        res = np.mean(aris[:, :, m], axis=0)
        res_stds = np.std(aris[:, :, m], axis=0)

        if m > 0:
            plt.plot(percs, res, '-', color=cols[:, m], linewidth=4)
            # plt.errorbar(percs, res, res_stds, fmt='-', color=cols[:, m], linewidth=4, elinewidth=1)
        else:
            plt.plot(percs, res, '--', color=cols[:, m], linewidth=4)
            # plt.errorbar(percs, res, res_stds, fmt='--', color=cols[:, m], linewidth=4, elinewidth=1)
        aucs[m] = np.trapz(res, percs)

    plt.title('{2}\n#Src={0}, #Reps={1}'.format(n_src, aris.shape[0], title))
    plt.xlabel('Fraction of target samples ('
               '1.0={0} samples)'.format(n_trg), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim([4e-2, 1.3])
    plt.ylim([0., 1.])

    plt.legend(desc, loc=4)
    plt.semilogx()

    plt.subplot(1, 2, 2)
    plt.title('Overall performance')
    plt.bar(np.arange(aucs.size), aucs, color=cols.T)
    plt.xticks(np.arange(len(desc))-0.0, desc, rotation=45)
    plt.ylabel('Area under curve', fontsize=14)


def exp_percs_accs(methods, n_src=800, n_trg=800, mode=2, reps=10, cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]]):
    flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    n_cluster = len(flatten(cluster_spec))
    print 'Number of cluster is ', n_cluster
    fname = 'res_mtl_m{0}_r{1}.npz'.format(mode, reps)

    percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 9, 11]]
    # percs = [0.1, 0.4, 0.8]

    aris = np.zeros((reps, len(percs), len(methods)))
    aris_strat = np.zeros((reps, len(percs), len(methods)))
    num_strat = np.zeros((reps, len(percs), len(methods)))
    aucs_nmf = None
    aucs_nmf_desc = None
    res_desc = []
    for r in range(reps):
        # 1. Generate scRNA data
        data, labels = generate_toy_data(num_genes=1000,
                                         num_cells=n_trg + n_src,
                                         cluster_spec=cluster_spec)
        # 2. Split source and target according to specified mode/setting
        src, trg, src_labels, trg_labels = split_source_target(data, labels,
                                                               target_ncells=n_trg, source_ncells=n_src,
                                                               mode=mode, source_clusters=None,
                                                               noise_target=False, noise_sd=0.0)
        trg_labels = np.array(trg_labels, dtype=np.int)
        src_labels = np.array(src_labels, dtype=np.int)
        # 3.a. Subsampling order for target
        inds = np.random.permutation(trg_labels.size)
        # 3.b. Use perfect number of latent states for nmf and sc3
        src_lbl_set = np.unique(src_labels)
        n_trg_cluster = np.unique(trg_labels).size
        n_src_cluster = src_lbl_set.size
        # 3.c. Target data subsampling loop
        for i in range(len(percs)):
            n_trg_perc = np.int(n_trg * percs[i])
            p_trg = trg[:, inds[:n_trg_perc]].copy()
            p_trg_labels = trg_labels[inds[:n_trg_perc]].copy()
            # 4. MTL/DA mixing parameter loop
            res_desc = list()
            reject = None
            for m in range(len(methods)):
                desc, lbls, reject = methods[m](src.copy(), src_labels.copy(),
                                                p_trg.copy(), p_trg_labels.copy(),
                                                n_src_cluster=n_src_cluster, n_trg_cluster=n_trg_cluster)
                res_desc.append(desc)

                # evaluate
                aris[r, i, m] = metrics.adjusted_rand_score(p_trg_labels, lbls)
                lbls_inds = []
                for n in range(p_trg_labels.size):
                    if p_trg_labels[n] in src_lbl_set:
                        lbls_inds.append(n)
                aris_strat[r, i, m] = metrics.adjusted_rand_score(
                    p_trg_labels[lbls_inds], lbls[lbls_inds])
                num_strat[r, i, m] = np.float(len(lbls_inds)) / np.float(lbls.size)

                # handle reject options
                if reject is not None and lbls.size > len(lbls_inds):
                    # setup reject measures
                    if aucs_nmf is None:
                        aucs_nmf = np.zeros((reps, len(percs), len(reject)))
                        aucs_nmf_desc = list()
                        for n in range(len(reject)):
                            aucs_nmf_desc.append(reject[n][0])

                    bin_lbls = np.zeros(lbls.size, dtype=np.int)
                    bin_lbls[lbls_inds] = 1
                    for n in range(len(reject)):
                        fpr, tpr, thresholds = metrics.roc_curve(bin_lbls, reject[n][1], pos_label=1)
                        aucs_nmf[r, i, n] = metrics.auc(fpr, tpr)
                        print 'Aucs ', aucs_nmf_desc[n], ' = ', aucs_nmf[r, i, n]

    # save the result and then plot
    np.savez(fname, methods=methods, aris=aris, percs=percs, reps=reps, n_src=n_src, n_trg=n_trg,
             desc=res_desc, mode=mode, num_strat=num_strat, aris_strat=aris_strat,
             aucs_nmf=aucs_nmf, aucs_nmf_desc=aucs_nmf_desc)
    print('Done.')


if __name__ == "__main__":

    methods = list()
    methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    # methods.append(partial(method_sc3, mix=0.2, metric='euclidean'))
    # methods.append(partial(method_sc3, mix=0.8, metric='euclidean'))
    methods.append(partial(method_da_nmf, use_strat=True))
    methods.append(partial(method_da_nmf, use_strat=False))

    exp_percs_accs(methods, mode=4, reps=5, cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]])

    fname = 'res_mtl_m4_r5.npz'
    plot_percs_accs(fname)
