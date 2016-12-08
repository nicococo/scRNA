from functools import partial

import numpy as np
from sklearn import metrics as metrics

import sc3_clustering_impl as sc
from nmf_clustering import NmfClustering, DaNmfClustering
from sc3_clustering import SC3Clustering
from simulation import generate_toy_data, split_source_target
from utils import *

def method_sc3(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster,
               mix=0.5, consensus_mode=0, metric='euclidean', limit_pc_range=-1):
    num_cells = trg.shape[1]
    if num_cells > limit_pc_range > 0:
        print('Limit PC range to :'.format(limit_pc_range))
        num_cells = limit_pc_range
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


def acc_reject_ari(X, lbls_true, lbls_pred, reject, strat_lbl_inds, reject_name='KTA kurt1', threshold=0.1):
    desc = ('Rejection-ARI {0}-{1}'.format(reject_name, threshold), 'ARI')
    if reject is None:
        return 0.0, desc
    ari = 0.0
    for n in range(len(reject)):
        name, values = reject[n]
        if reject_name in name:
            thres_ind = np.int(np.float(values.size)*threshold)
            inds = np.argsort(values)
            inds = inds[thres_ind:]
            ari = metrics.adjusted_rand_score(lbls_true[inds], lbls_pred[inds])
    return ari, desc


def acc_ari(X, lbls_true, lbls_pred, reject, strat_lbl_inds, use_strat=False):
    if use_strat:
        ari = metrics.adjusted_rand_score(lbls_true[strat_lbl_inds], lbls_pred[strat_lbl_inds])
        perc = np.int(np.float(len(strat_lbl_inds))/np.float(lbls_true.size) * 100.0)
        desc = ('ARI (strat={0})'.format(perc), 'ARI')
    else:
        ari = metrics.adjusted_rand_score(lbls_true, lbls_pred)
        desc = ('ARI', 'ARI')
    return ari, desc


def acc_silhouette(X, lbls_true, lbls_pred, reject, strat_lbl_inds, use_strat=False, metric='euclidean'):
    if use_strat:
        dists = sc.distances(X[:, strat_lbl_inds], gene_ids=np.arange(strat_lbl_inds.size), metric=metric )
        sil = metrics.silhouette_score(dists, lbls_pred[strat_lbl_inds], metric='precomputed')
        perc = np.int(np.float(len(strat_lbl_inds))/np.float(lbls_true.size) * 100.0)
        desc = ('Silhouette (strat={0},{1})'.format(perc, metric), 'Silhouette ({0})'.format(metric))
    else:
        dists = sc.distances(X, gene_ids=np.arange(X.shape[1]), metric=metric )
        sil = metrics.silhouette_score(dists, lbls_pred, metric='precomputed')
        desc = ('Silhouette ({0})'.format(metric), 'Silhouette ({0})'.format(metric))
    return sil, desc


def acc_kta(X, lbls_true, lbls_pred, reject, strat_lbl_inds, kernel='linear', param=1.0):
    Ky = np.zeros((lbls_pred.size, np.max(lbls_pred)+1))
    for i in range(len(lbls_pred)):
        Ky[i, lbls_pred[i]] = 1.

    if kernel == 'rbf':
        Kx = get_kernel(X, X, type='rbf', param=param)
        Ky = get_kernel(Ky.T, Ky.T, type='rbf', param=param)
    else:
        Kx = X.T.dot(X)
        Ky = Ky.dot(Ky.T)

    Kx = center_kernel(Kx)
    Ky = center_kernel(Ky)
    Kx = normalize_kernel(Kx)
    Ky = normalize_kernel(Ky)

    sil = kta_align_general(Kx, Ky)
    desc = ('KTA ({0})'.format(kernel), 'KTA ({0})'.format(kernel))
    return sil, desc


def experiment_loop(fname, methods, acc_funcs, n_src=800, n_trg=800, n_genes=1000, mode=2, reps=10,
                    cluster_mode=False, cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]], percs=[0.1, 0.4, 0.8]):
    flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    n_cluster = len(flatten(cluster_spec))
    print 'Number of cluster is ', n_cluster

    accs = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
    accs_desc = [''] * len(acc_funcs)

    num_strat = np.zeros((reps, len(percs), len(methods)))
    res_desc = []
    for r in range(reps):
        # 1. Generate scRNA data
        data, labels = generate_toy_data(num_genes=n_genes,
                                         num_cells=2.*(n_trg + n_src),  # generate more data
                                         cluster_spec=cluster_spec)
        # 2. Split source and target according to specified mode/setting
        src, trg, src_labels, trg_labels = split_source_target(data, labels,
                                                               target_ncells=n_trg, source_ncells=n_src,
                                                               mode=mode, source_clusters=None,
                                                               noise_target=False, noise_sd=0.1)
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
            if cluster_mode:
                n_trg_cluster = percs[i]
                p_trg = trg[:, inds[:n_trg]].copy()
                p_trg_labels = trg_labels[inds[:n_trg]].copy()
            else:
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
             percs=percs, reps=reps, n_genes=n_genes, n_src=n_src, n_trg=n_trg,
             desc=res_desc, mode=mode, num_strat=num_strat, cluster_mode=cluster_mode)
    print('Done.')