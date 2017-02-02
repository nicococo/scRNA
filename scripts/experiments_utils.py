from functools import partial

from sklearn import metrics as metrics

import scRNA.sc3_clustering_impl as sc
from scRNA.nmf_clustering import NmfClustering, DaNmfClustering
from scRNA.sc3_clustering import SC3Clustering
from scRNA.simulation import generate_toy_data, split_source_target
from scRNA.utils import *


def method_hub(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster,
               method_list=None, func=None):
    aris = np.zeros(len(method_list))
    reject = list()
    src_lbls = np.zeros((len(method_list), src_labels.size))
    trg_lbls = np.zeros((len(method_list), trg_labels.size))
    for i in range(len(method_list)):
        desc, src_lbls[i, :], trg_lbls[i, :], r = method_list[i](src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster)
        reject.append(r)
        aris[i] = metrics.adjusted_rand_score(trg_labels, trg_lbls[i, :])
    ind = func(aris)
    # print aris
    # print ind, np.argmax(aris), np.argmin(aris)
    desc['hub'] = func
    desc['stats'] = (np.max(aris), np.min(aris), np.mean(aris))
    return desc, src_lbls[ind, :], trg_lbls[ind, :], reject[ind]


def method_random(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster):
    desc = {}
    desc['method'] = 'random'
    return desc, np.random.randint(0, n_src_cluster, size=src_labels.size), \
           np.random.randint(0, n_trg_cluster, size=src_labels.size), None


def method_sc3_combined(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster,
                        consensus_mode=0, metric='euclidean'):
    lbls = np.hstack([trg_labels, src_labels])
    n_cluster = np.unique(lbls).size
    num_cells = trg.shape[1] + src.shape[1]
    max_pca_comp = np.ceil(num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(num_cells * 0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    cp = SC3Clustering(np.hstack([trg, src]), pc_range=[min_pca_comp, max_pca_comp],
                       consensus_mode=consensus_mode, sub_sample=True)
    cp.add_distance_calculation(partial(sc.distances, metric=metric))
    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
    cp.apply()

    # add some description
    desc = {}
    desc['method'] = 'SC3-comb'
    desc['metric'] = metric
    return desc, cp.cluster_labels[trg_labels.size:], cp.cluster_labels[:trg_labels.size], None


def method_sc3(src, src_labels, trg, trg_labels, n_src_cluster, n_trg_cluster,
               limit_pc_range=-1, metric='euclidean', consensus_mode=0,
               mix=0.5, use_da_dists=True, reject_ratio=0.0):

    num_cells = trg.shape[1]
    if num_cells > limit_pc_range > 0:
        print('Limit PC range to :'.format(limit_pc_range))
        num_cells = limit_pc_range
    max_pca_comp = np.ceil(num_cells * 0.07).astype(np.int)
    min_pca_comp = np.floor(num_cells * 0.04).astype(np.int)
    print 'Min and max PCA components: ', min_pca_comp, max_pca_comp

    nmf_src = NmfClustering(src, np.arange(src.shape[0]), num_cluster=n_src_cluster)
    nmf_trg = DaNmfClustering(nmf_src, trg, np.arange(trg.shape[0]), num_cluster=n_trg_cluster)
    mixed_data, _, _ = nmf_trg.get_mixed_data(mix=mix, reject_ratio=reject_ratio)

    # use mixed data are mixed distances
    cp = SC3Clustering(trg, pc_range=[min_pca_comp, max_pca_comp],
                       consensus_mode=consensus_mode, sub_sample=True)
    if not use_da_dists:
        cp = SC3Clustering(mixed_data, pc_range=[min_pca_comp, max_pca_comp],
                           consensus_mode=consensus_mode, sub_sample=True)
        cp.add_distance_calculation(partial(sc.distances, metric=metric))
    else:
        cp.add_distance_calculation(partial(sc.da_nmf_distances,
                                            da_model=nmf_trg.intermediate_model,
                                            metric=metric, mixture=mix, reject_ratio=reject_ratio))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_trg_cluster))
    cp.set_build_consensus_matrix(sc.build_consensus_matrix)
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_trg_cluster))
    cp.apply()

    # add some description
    desc = {}
    desc['method'] = 'SC3'
    desc['metric'] = metric
    desc['reject'] = reject_ratio
    desc['mix'] = mix
    desc['use_da_dists'] = use_da_dists
    return desc, nmf_src.cluster_labels, cp.cluster_labels, nmf_trg.reject


def get_strat_lbl_inds(src_labels, trg_labels):
    src_lbl_set = np.unique(src_labels)
    strat_lbl_inds = []
    for n in range(trg_labels.size):
        if trg_labels[n] in src_lbl_set:
            strat_lbl_inds.append(n)
    return strat_lbl_inds


def acc_reject_auc(X_src, src_labels, X_trg, trg_labels, src_lbls_pred, lbls_pred, reject, reject_name='KTA kurt1'):
    desc = ('Rejection-AUC {0}'.format(reject_name), 'AUC')
    if reject is None:
        return 0.0, desc
    bin_lbls = np.zeros(trg_labels.size, dtype=np.int)
    bin_lbls[get_strat_lbl_inds(src_labels, trg_labels)] = 1
    for n in range(len(reject)):
        name, value = reject[n]
        if reject_name in name:
            fpr, tpr, thresholds = metrics.roc_curve(bin_lbls, value, pos_label=1)
            auc = metrics.auc(fpr, tpr)
    return auc, desc


def acc_transferability(X_src, src_labels, X_trg, trg_labels, src_lbls_pred, lbls_pred, reject):
    desc = ('Transferability', 'Transferability')
    if reject is None:
        return 0.0, desc
    name, value = reject[-1]
    if 'Transferability' not in name:
        value = 0.0
    return value, desc


def acc_reject_ari(X_src, src_labels, X_trg, trg_labels, src_lbls_pred, lbls_pred, reject, reject_name='KTA kurt1', threshold=0.1):
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
            ari = metrics.adjusted_rand_score(trg_labels[inds], lbls_pred[inds])
    return ari, desc


def acc_ari(X_src, src_labels, X_trg, trg_labels, src_lbls_pred, lbls_pred, reject,
            use_strat=False, test_src_lbls=False):
    if use_strat:
        strat_lbl_inds = get_strat_lbl_inds(src_labels, trg_labels)
        ari = metrics.adjusted_rand_score(trg_labels[strat_lbl_inds], lbls_pred[strat_lbl_inds])
        perc = np.int(np.float(len(strat_lbl_inds)) / np.float(trg_labels.size) * 100.0)
        desc = ('ARI (strat={0})'.format(perc), 'ARI')
    else:
        desc = ('ARI', 'ARI')
        ari = metrics.adjusted_rand_score(trg_labels, lbls_pred)
        if test_src_lbls:
            desc = ('ARI (src)', 'ARI')
            ari = metrics.adjusted_rand_score(src_labels, src_lbls_pred)
    return ari, desc


def acc_silhouette(X_src, src_labels, X_trg, trg_labels, src_lbls_pred, lbls_pred, reject, metric='euclidean'):
    dists = sc.distances(X_trg, gene_ids=np.arange(X_trg.shape[1]), metric=metric )
    num_lbls = np.unique(lbls_pred).size
    sil = 1.0
    if num_lbls > 1:
        sil = metrics.silhouette_score(dists, lbls_pred, metric='precomputed')
    desc = ('Silhouette ({0})'.format(metric), 'Silhouette ({0})'.format(metric))
    return sil, desc


def acc_kta(X_src, src_labels, X_trg, trg_labels, src_lbls_pred, trg_lbls_pred, reject, kernel='linear', param=1.0):
    Ky = np.zeros((trg_lbls_pred.size, np.max(trg_lbls_pred) + 1))
    for i in range(len(trg_lbls_pred)):
        Ky[i, trg_lbls_pred[i]] = 1.

    if kernel == 'rbf':
        Kx = get_kernel(X_trg, X_trg, type='rbf', param=param)
        Ky = get_kernel(Ky.T, Ky.T, type='rbf', param=param)
    else:
        Kx = X_trg.T.dot(X_trg)
        Ky = Ky.dot(Ky.T)

    Kx = center_kernel(Kx)
    Ky = center_kernel(Ky)
    Kx = normalize_kernel(Kx)
    Ky = normalize_kernel(Ky)

    sil = kta_align_general(Kx, Ky)
    desc = ('KTA ({0})'.format(kernel), 'KTA ({0})'.format(kernel))
    return sil, desc


def experiment_loop(fname, methods, acc_funcs, n_src=800, n_trg=800, n_genes=1000, mode=2,
                    reps=10,
                    n_common_cluster=2,
                    cluster_mode=False,
                    cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]],
                    percs=[0.1, 0.4, 0.8]):

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
                                         num_cells=10.*(n_trg + n_src),  # generate more data
                                         cluster_spec=cluster_spec)
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
                desc, src_lbls_pred, trg_lbls_pred, reject = methods[m](src.copy(), src_labels.copy(),
                                                p_trg.copy(), p_trg_labels.copy(),
                                                n_src_cluster=n_src_cluster,
                                                n_trg_cluster=n_trg_cluster)
                res_desc.append(desc)
                for f in range(len(acc_funcs)):
                    accs[f, r, i, m], desc = acc_funcs[f](src.copy(), src_labels.copy(),
                                                          p_trg.copy(), p_trg_labels.copy(),
                                                          src_lbls_pred.copy(), trg_lbls_pred.copy(),
                                                          reject)
                    accs_desc[f] = desc

    # save the result and then plot
    np.savez(fname, methods=methods, acc_funcs=acc_funcs, accs=accs, accs_desc=accs_desc,
             percs=percs, reps=reps, n_genes=n_genes, n_src=n_src, n_trg=n_trg,
             desc=res_desc, mode=mode, n_common_cluster=n_common_cluster,
             num_strat=num_strat, cluster_mode=cluster_mode)
    print('Done.')
    return accs, accs_desc, res_desc