import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from functools import partial

import sc3_clustering_impl as sc
from sc3_clustering import SC3Clustering
from nmf_clustering import DaNmfClustering, NmfClustering
from simulation import *


def method_sc3(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster,
               mix=0.5, consensus_mode=0, metric='euclidean'):
    num_cells = trg.shape[1]
    max_pca_comp = np.ceil(num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(num_cells * 0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    cp = SC3Clustering(trg, pc_range=[min_pca_comp, max_pca_comp],
                       consensus_mode=consensus_mode, sub_sample=True)

    src = NmfClustering(src, np.arange(src.shape[0]), num_cluster=n_src_cluster)
    cp.add_distance_calculation(partial(sc.da_nmf_distances, src=src, metric=metric, mixture=mix))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_trg_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_trg_cluster))
    cp.apply()

    desc = 'SC3*'
    if mix > 0.0:
        desc = 'DA-SC3* ({0}%, {1})'.format(np.int(mix*100.), metric[:3])
    return desc, cp.cluster_labels, None


def method_da_nmf(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster, mix=0.0):
    src = NmfClustering(src, np.arange(src.shape[0]), num_cluster=n_src_cluster)
    cp = DaNmfClustering(src, trg, np.arange(trg.shape[0]), num_cluster=n_trg_cluster)
    cp.apply(mix=mix)
    lbls = cp.cluster_labels
    desc = 'DA-NMF-Mix {0}'.format(np.int(mix*100.))
    return desc, lbls, cp.reject


def method_nmf(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster):
    ids = np.arange(trg.shape[0])
    cp = NmfClustering(trg, ids, num_cluster=n_trg_cluster)
    cp.apply()
    return 'NMF', cp.cluster_labels, None


def plot_results(fname):
    foo = np.load(fname)
    accs = foo['accs']
    accs_desc = foo['accs_desc']
    methods = foo['methods']
    methods_desc = foo['desc']
    percs = foo['percs']
    n_trg = foo['n_trg']
    n_src = foo['n_src']

    for i in range(accs.shape[0]):
        title, _ = accs_desc[i]
        plot_single(i, title, accs[i, :, :, :], methods, percs, n_src, n_trg, methods_desc)

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


def acc_reject(X, lbls_true, lbls_pred, reject, strat_lbl_inds, reject_name='KTA kurt1'):
    desc = ('Rejection {0}'.format(reject_name), 'AUC')
    if reject is None:
        return 0.0, desc
    bin_lbls = np.zeros(lbls_true.size, dtype=np.int)
    bin_lbls[strat_lbl_inds] = 1
    for n in range(len(reject)):
        name, value = reject[n]
        if reject_name in name:
            fpr, tpr, thresholds = metrics.roc_curve(bin_lbls, value, pos_label=1)
            auc = metrics.auc(fpr, tpr)
    return auc, desc


def acc_ari(X, lbls_true, lbls_pred, reject, strat_lbl_inds, use_strat=False):
    if use_strat:
        ari = metrics.adjusted_rand_score(lbls_true[strat_lbl_inds], lbls_pred[strat_lbl_inds])
        perc = np.int(np.float(len(strat_lbl_inds))/np.float(lbls_true.size) * 100.0)
        desc = ('ARI (strat={0})'.format(perc), 'ARI')
    else:
        ari = metrics.adjusted_rand_score(lbls_true, lbls_pred)
        desc = ('ARI', 'ARI')
    return ari, desc


def acc_silhouette(X, lbls_true, lbls_pred, reject, strat_lbl_inds, use_strat=False):
    if use_strat:
        sil = metrics.silhouette_score(X[:, strat_lbl_inds].T, lbls_pred[strat_lbl_inds])
        perc = np.int(np.float(len(strat_lbl_inds))/np.float(lbls_true.size) * 100.0)
        desc = ('Silhouette (strat={0})'.format(perc), 'Silhouette')
    else:
        sil = metrics.silhouette_score(X.T, lbls_pred)
        desc = ('Silhouette', 'Silhouette')
    return sil, desc


def experiment_loop(methods, acc_funcs, n_src=800, n_trg=800, mode=2, reps=10,
                    cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]], percs=[0.1, 0.4, 0.8]):
    flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    n_cluster = len(flatten(cluster_spec))
    print 'Number of cluster is ', n_cluster
    fname = 'res_mtl_m{0}_r{1}.npz'.format(mode, reps)

    accs = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
    accs_desc = [''] * len(acc_funcs)

    num_strat = np.zeros((reps, len(percs), len(methods)))
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
            for m in range(len(methods)):
                desc, lbls, reject = methods[m](src.copy(), src_labels.copy(),
                                                p_trg.copy(), p_trg_labels.copy(),
                                                n_src_cluster=n_src_cluster, n_trg_cluster=n_trg_cluster)
                res_desc.append(desc)

                # evaluation
                strat_lbl_inds = []
                for n in range(p_trg_labels.size):
                    if p_trg_labels[n] in src_lbl_set:
                        strat_lbl_inds.append(n)

                for f in range(len(acc_funcs)):
                    accs[f, r, i, m], desc = acc_funcs[f](p_trg.copy(), p_trg_labels.copy(),
                                                          lbls.copy(), reject, strat_lbl_inds)
                    accs_desc[f] = desc

    # save the result and then plot
    np.savez(fname, methods=methods, acc_funcs=acc_funcs, accs=accs, accs_desc=accs_desc,
             percs=percs, reps=reps, n_src=n_src, n_trg=n_trg, desc=res_desc, mode=mode, num_strat=num_strat)
    print('Done.')


if __name__ == "__main__":
    percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 9, 11]]
    #percs = [0.1, 0.4, 0.8, 1.0]

    acc_funcs = list()
    acc_funcs.append(partial(acc_ari, use_strat=False))
    acc_funcs.append(partial(acc_ari, use_strat=True))
    acc_funcs.append(partial(acc_silhouette, use_strat=False))
    acc_funcs.append(partial(acc_silhouette, use_strat=True))
    acc_funcs.append(partial(acc_reject, reject_name='KTA kurt1'))
    acc_funcs.append(partial(acc_reject, reject_name='KTA kurt2'))
    acc_funcs.append(partial(acc_reject, reject_name='KTA kurt3'))
    acc_funcs.append(partial(acc_reject, reject_name='Kurtosis'))
    acc_funcs.append(partial(acc_reject, reject_name='Entropy'))
    acc_funcs.append(partial(acc_reject, reject_name='Diffs'))

    methods = list()
    methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    methods.append(partial(method_sc3, mix=0.5, metric='euclidean'))
    methods.append(partial(method_sc3, mix=1.0, metric='euclidean'))
    # methods.append(partial(method_nmf))
    methods.append(partial(method_da_nmf, mix=0.0))
    # methods.append(partial(method_da_nmf, mix=0.25))
    methods.append(partial(method_da_nmf, mix=0.5))
    # methods.append(partial(method_da_nmf, mix=0.75))
    methods.append(partial(method_da_nmf, mix=1.0))
    # methods.append(partial(method_da_nmf, use_strat=False))

    experiment_loop(methods, acc_funcs, mode=4, reps=20,
                    cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]], percs=percs)

    fname = 'res_mtl_m4_r20.npz'
    plot_results(fname)
