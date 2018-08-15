from functools import partial

import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression

from scRNA.nmf_clustering import NmfClustering, DaNmfClustering
from scRNA.sc3_clustering import SC3Clustering
from scRNA.simulation import generate_toy_data, split_source_target
from scRNA.utils import *


def method_hub(src_nmf, trg, trg_labels, n_trg_cluster, method_list=None, func=None):
    aris = np.zeros(len(method_list))
    desc = {}
    trg_nmfs = list()
    trg_lbls = np.zeros((len(method_list), trg_labels.size), dtype=np.int)
    for i in range(len(method_list)):
        desc, trg_nmf, trg_lbls[i, :] = method_list[i](src_nmf, trg, trg_labels, n_trg_cluster)
        trg_nmfs.append(trg_nmf)
        aris[i] = metrics.adjusted_rand_score(trg_labels, trg_lbls[i, :])
    ind = func(aris)
    if np.array_equal(ind, aris):
        trg_nmfs_out = trg_nmfs
        trg_lbls_out = trg_lbls
    else:
        trg_nmfs_out = trg_nmfs[ind]
        trg_lbls_out = trg_lbls[ind, :]
    desc['hub'] = func
    desc['stats'] = (np.max(aris), np.min(aris), np.mean(aris))
    return desc, trg_nmfs_out, trg_lbls_out


def method_random(src_nmf, trg, trg_labels, n_trg_cluster):
    return {'method': 'random'}, \
           DaNmfClustering(src_nmf, trg, np.arange(trg.shape[0]), num_cluster=n_trg_cluster), \
           np.random.randint(0, n_trg_cluster, size=trg_labels.size)


def method_sc3_combined(src_nmf, trg, trg_labels, n_trg_cluster, consensus_mode=0, metric='euclidean'):
    lbls = np.hstack([trg_labels, src_nmf.cluster_labels])
    n_cluster = np.unique(lbls).size
    num_cells = trg.shape[1] + src_nmf.data.shape[1]
    max_pca_comp = np.ceil(num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(num_cells * 0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    cp = SC3Clustering(np.hstack([trg, src_nmf.data]), pc_range=[min_pca_comp, max_pca_comp],
                       consensus_mode=consensus_mode, sub_sample=True)
    cp.add_distance_calculation(partial(sc.distances, metric=metric))
    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
    cp.apply()
    return {'method': 'SC3-comb', 'metric': metric}, \
           DaNmfClustering(src_nmf, trg, np.arange(trg.shape[0]), num_cluster=n_trg_cluster), \
           cp.cluster_labels[:trg_labels.size]


def method_sc3(src_nmf, trg, trg_labels, n_trg_cluster,
               limit_pc_range=-1, metric='euclidean', consensus_mode=0,
               mix=0.5, use_da_dists=True, calc_transferability=True):
    print [mix, 'Mixture Parameter']
    num_cells = trg.shape[1]
    if num_cells > limit_pc_range > 0:
        print('Limit PC range to :'.format(limit_pc_range))
        num_cells = limit_pc_range
    max_pca_comp = np.ceil(num_cells * 0.07).astype(np.int)
    min_pca_comp = np.max([1,np.floor(num_cells * 0.04).astype(np.int)])
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    trg_nmf = DaNmfClustering(src_nmf, trg, np.arange(trg.shape[0]), num_cluster=n_trg_cluster)
    mixed_data, _, _ = trg_nmf.get_mixed_data(mix=mix, calc_transferability=calc_transferability)

    # use mixed data are mixed distances
    cp = SC3Clustering(trg, pc_range=[min_pca_comp, max_pca_comp],
                       consensus_mode=consensus_mode, sub_sample=True)
    if not use_da_dists:
        cp = SC3Clustering(mixed_data, pc_range=[min_pca_comp, max_pca_comp],
                           consensus_mode=consensus_mode, sub_sample=True)
        cp.add_distance_calculation(partial(sc.distances, metric=metric))
    else:
        cp.add_distance_calculation(partial(sc.da_nmf_distances,
                                            da_model=trg_nmf.intermediate_model,
                                            metric=metric, mixture=mix, reject_ratio=0.))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_trg_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_trg_cluster))
    cp.apply()
    return {'method': 'SC3', 'metric': metric, 'mix': mix, 'use_da_dists': use_da_dists}, trg_nmf, cp.cluster_labels


def acc_transferability(trg_nmf, X_trg, trg_labels, lbls_pred):
    return trg_nmf.transferability_score, 'Transferability'


def acc_classification(trg_nmf, X_trg, trg_labels, lbls_pred):
    if trg_nmf.intermediate_model is None:
        return 0.0, 'MixARI'

    include_inds = []
    pred_lbls = np.unique(lbls_pred)
    for p in pred_lbls:
        inds = np.where(lbls_pred == p)[0]
        if inds.size >= 2:
            include_inds.extend(inds)

    W, H, H2 = trg_nmf.intermediate_model
    mixed_data = W.dot(H)
    cls = OneVsRestClassifier(LinearRegression(fit_intercept=True, normalize=False, copy_X=False, n_jobs=1)).fit(
        mixed_data[:, include_inds].T, lbls_pred[include_inds])
    ret = cls.predict(X_trg[:, include_inds].T)
    return metrics.adjusted_rand_score(ret, lbls_pred[include_inds]), 'MixARI'


def acc_ari(trg_nmf, X_trg, trg_labels, lbls_pred, use_strat=False):
    if len(lbls_pred.shape) == 1:
        if use_strat:
            stratify = lambda s, t, i: stratify(s, [t[0]], i) + stratify(s, t[1:], i + 1) if len(t) > 1 else [i] if t[0] in s else []
            strat_lbl_inds = stratify(np.unique(trg_nmf.src.cluster_labels), trg_labels, 0)
            perc = np.int(np.float(len(strat_lbl_inds)) / np.float(trg_labels.size) * 100.0)
            desc = 'ARI (strat={0})'.format(perc)
            ari = metrics.adjusted_rand_score(trg_labels[strat_lbl_inds], lbls_pred[strat_lbl_inds])
        else:
            desc = 'ARI'
            ari = metrics.adjusted_rand_score(trg_labels, lbls_pred)
    else:
        ari = np.zeros(lbls_pred.shape[0])
        if use_strat:
            stratify = lambda s, t, i: stratify(s, [t[0]], i) + stratify(s, t[1:], i + 1) if len(t) > 1 else [i] if t[0] in s else []
            strat_lbl_inds = stratify(np.unique(trg_nmf.src.cluster_labels), trg_labels, 0)
            perc = np.int(np.float(len(strat_lbl_inds)) / np.float(trg_labels.size) * 100.0)
            desc = 'ARI (strat={0})'.format(perc)
            for ind in range(lbls_pred.shape[0]):
                ari[ind] = metrics.adjusted_rand_score(trg_labels[strat_lbl_inds], lbls_pred[ind, strat_lbl_inds])
        else:
            desc = 'ARI'
            for ind in range(lbls_pred.shape[0]):
                # print ind, lbls_pred.shape[1]
                ari[ind] = metrics.adjusted_rand_score(trg_labels, lbls_pred[ind,:])
    return ari, desc


def acc_silhouette(trg_nmf, X_trg, trg_labels, lbls_pred, metric='euclidean'):
    dists = sc.distances(X_trg, gene_ids=np.arange(X_trg.shape[1]), metric=metric)
    if np.unique(lbls_pred).size <= 1:
        return 1.0, 'Silhouette ({0})'.format(metric)
    return metrics.silhouette_score(dists, lbls_pred, metric='precomputed'), 'Silhouette ({0})'.format(metric)


def acc_kta(target_nmf, X_trg, trg_labels, trg_lbls_pred, kernel='linear', param=1.0, center=True, normalize=True,
            mode=0):
    Ky = np.zeros((trg_lbls_pred.size, np.max(trg_lbls_pred) + 1))
    for i in range(len(trg_lbls_pred)):
        Ky[i, trg_lbls_pred[i]] = 1.

    fmt = ''
    if mode == 2 and target_nmf.intermediate_model is not None:
        # dense reconstructed data
        fmt = '-WH2'
        W, H, H2 = target_nmf.intermediate_model
        X_trg = W.dot(H2)
    if mode == 3 and target_nmf.intermediate_model is not None:
        # sparse reconstructed data
        fmt = '-WH'
        W, H, H2 = target_nmf.intermediate_model
        X_trg = W.dot(H)

    if kernel == 'rbf':
        Kx = get_kernel(X_trg, X_trg, type='rbf', param=param)
        Ky = get_kernel(Ky.T, Ky.T, type='rbf', param=param)
    else:
        Kx = X_trg.T.dot(X_trg)
        Ky = Ky.dot(Ky.T)

    if center:
        Kx = center_kernel(Kx)
        Ky = center_kernel(Ky)
    if normalize:
        Kx = normalize_kernel(Kx)
        Ky = normalize_kernel(Ky)

    sil = kta_align_general(Kx, Ky)
    return sil, 'KTA{1} ({0})'.format(kernel, fmt)


def experiment_loop(fname, methods, acc_funcs,
                    n_src=800,
                    n_trg=800,
                    n_genes=1000,
                    mode=2,
                    reps=10,
                    n_common_cluster=2,
                    cluster_mode=False,
                    cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]],
                    percs=[0.1, 0.4, 0.8],
                    min_src_cluster_ari=0.94):
    flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    n_cluster = len(flatten(cluster_spec))
    print 'Number of cluster is ', n_cluster

    source_aris = np.zeros(reps)
    accs = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
    accs_desc = list()

    num_strat = np.zeros((reps, len(percs), len(methods)))
    res_desc = []
    r = 0
    while r < reps:
        print 'running experiment: r, fname'
        print r, fname
        # 1. Generate scRNA data
        data, labels = generate_toy_data(num_genes=n_genes, num_cells=10. * (n_trg + n_src), cluster_spec=cluster_spec)
        # 2. Split source and target according to specified mode/setting
        src, trg, src_labels, trg_labels = split_source_target(data, labels,
                                                               target_ncells=n_trg,
                                                               source_ncells=n_src,
                                                               mode=mode,
                                                               source_clusters=None,
                                                               noise_target=False,
                                                               noise_sd=0.1,
                                                               common=n_common_cluster,
                                                               cluster_spec=cluster_spec)
        trg_labels = np.array(trg_labels, dtype=np.int)
        src_labels = np.array(src_labels, dtype=np.int)
        # 3.a. Subsampling order for target

        inds = np.random.permutation(trg_labels.size)
        # 3.b. Use perfect number of latent states for nmf and sc3
        src_lbl_set = np.unique(src_labels)
        n_trg_cluster = np.unique(trg_labels).size
        n_src_cluster = src_lbl_set.size
        # 3.c. train source once per repetition
        source_nmf = NmfClustering(src, np.arange(src.shape[0]), num_cluster=n_src_cluster)
        source_nmf.apply(k=n_src_cluster, max_iter=4000, rel_err=1e-3)
        source_aris[r] = metrics.adjusted_rand_score(src_labels, source_nmf.cluster_labels)
        print 'ITER(', r, '): SOURCE ARI = ', source_aris[r]
        if source_aris[r] < min_src_cluster_ari:
            continue

        # 3.d. Target data subsampling loop
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
                desc, target_nmf, trg_lbls_pred = methods[m](source_nmf, p_trg.copy(), p_trg_labels.copy(),
                                                             n_trg_cluster=n_trg_cluster)
                res_desc.append(desc)
                accs_desc = list()
                for f in range(len(acc_funcs)):
                    accs[f, r, i, m], accs_descr = acc_funcs[f](target_nmf, p_trg.copy(), p_trg_labels.copy(),
                                                                trg_lbls_pred.copy())
                    accs_desc.append(accs_descr)
        r += 1

    # save the result and then plot
    np.savez(fname, methods=methods, acc_funcs=acc_funcs, accs=accs, accs_desc=accs_desc,
             percs=percs, reps=reps, n_genes=n_genes, n_src=n_src, n_trg=n_trg,
             desc=res_desc, mode=mode, n_common_cluster=n_common_cluster,
             num_strat=num_strat, cluster_mode=cluster_mode, source_aris=source_aris)
    print('Done.')
    return source_aris, accs, accs_desc, res_desc
