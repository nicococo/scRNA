import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.metrics as metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Lars
from sklearn.svm import SVR, LinearSVR
from functools import partial

from nmf_clustering import NmfClustering, DaNmfClustering
from sc3_clustering import SC3Clustering
from utils import *


def plot_results(fname):
    foo = np.load(fname)
    src = foo['source_ari'][()]
    trg = foo['target_ari'][()]
    n_source_cluster = 16
    n_target_cluster = foo['n_target_cluster'][()]
    n_source = foo['n_source'][()]
    n_target = foo['n_target'][()]
    n_mix = foo['n_mix'][()]
    names = ['ARI', 'KTA-target', 'KTA-Reconstr(Sparse)', 'KTA-Reconstr(Dense)','Sil-Euc', 'Sil-Spe', 'Sil-Pea', 'MixAri(Lasso)', 'MixAri(LS)']
    names = ['ARI', 'KTA-target', 'KTA-Reconstr(Sparse)', 'KTA-Reconstr(Dense)','Sil-Euc', 'MixAri (Least Squares)']
    reps = src.size

    plt.figure(1)
    plt.subplot(len(n_target), 1, 1)
    plt.title('Source ARI mean={0:1.2f}, min={1:1.2f}'.format(np.mean(src), np.min(src)))
    for n in range(len(n_target)):

        ari_mean = np.mean(trg[0, :, n, : ].reshape(reps, len(n_mix)), axis=0)
        measures = list(range(1, 5))
        for a in measures:
            plt.subplot(len(n_target), len(measures), n*len(measures)+a)
            res_mean = np.mean(trg[a, :, n, : ].reshape(reps, len(n_mix)), axis=0)
            plt.plot(res_mean, ari_mean, '.b', linewidth=2)

            # for r in range(reps):
            #     res = trg[a, :, n, : ].reshape(reps, len(n_mix))
            #     plt.plot(res[r, :], ari_mean, '.', color=np.random.rand(3), linewidth=2)

            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.grid()
            plt.plot([0, 1], [0, 1], '--k', linewidth=1)
            plt.xlabel(names[a])
            plt.ylabel('{0} samples\n{1}'.format(n_target[n], names[0]))


        # for a in range(trg.shape[0]):
        #     res = trg[a, :, n, : ].reshape(reps, len(n_mix))
        #     print res.shape
        #     res_mean = np.mean(res, axis=0)
        #     res_std = np.std(res, axis=0)
        #     if a == 0:
        #         plt.errorbar(n_mix, res_mean, res_std, linewidth=2)
        #     else:
        #         plt.errorbar(n_mix, res_mean, res_std)
        #
        # plt.legend(names, loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.ylim([-0.01, 1.01])

    #----------------- ARI PLOT -------------
    plt.figure(2)
    plt.title('{0} Reps, Source ARI={1:1.2f}, #Samples={2}'.format(reps, np.mean(src), n_source))
    for n in range(len(n_target)):
        res = trg[0, :, n, : ].reshape(reps, len(n_mix))
        res_mean = np.mean(res, axis=0)
        res_std = np.std(res, axis=0)
        plt.errorbar(n_mix, res_mean, res_std, linewidth=2, elinewidth=1)
        plt.ylim([-0.01, 1.01])

    plt.legend(n_target, loc='lower left')
    plt.xlabel('Mixture levels')
    plt.ylabel('Ground truth accuracy [ARI]')

    #----------------- UNSUPERVISED PLOT -------------
    plt.figure(3)
    for n in range(0, 5):
        plt.subplot(1, 5, n+1)
        plt.title(names[n])
        res = trg[n, :, :, : ].reshape(reps, len(n_target), len(n_mix))
        res_mean = np.mean(res, axis=0).reshape(len(n_target), len(n_mix))
        plt.pcolor(res_mean)
        plt.colorbar()

        plt.xticks(list(range(len(n_mix))), n_mix, rotation=30)
        plt.xlim([0, len(n_mix)])
        plt.yticks(list(range(len(n_target))), n_target)
        plt.xlabel('Mixture levels')
        plt.ylabel('# Target data')

    #----------------- CLASSIFIER PLOT -------------
    plt.figure(4)
    for n in range(5, 6):
        plt.subplot(1, 1, n-4)
        plt.title(names[n])
        res = trg[n, :, :, : ].reshape(reps, len(n_target), len(n_mix))
        res_mean = np.mean(res, axis=0).reshape(len(n_target), len(n_mix))
        plt.pcolor(res_mean)
        plt.colorbar()

        plt.xticks(list(range(len(n_mix))), n_mix, rotation=30)
        plt.xlim([0, 21])
        plt.yticks(list(range(len(n_target))), n_target)
        plt.xlabel('Mixture levels')
        plt.ylabel('# Target data')

    plt.show()


def get_classifier_scores(pred_labels, mixed_data, target_nmf_pp_data, target_data, use_lasso=True, n_nonzeros=10):
    aris = np.zeros(5)
    active_genes = list(range(mixed_data.shape[0]))
    include_inds = []
    pred_lbls = np.unique(pred_labels)
    for p in pred_lbls:
        inds = np.where(pred_labels == p)[0]
        if inds.size >= 2:
            include_inds.extend(inds)
    if len(include_inds) > 2:
        if use_lasso:
            cls = OneVsRestClassifier(Lars(fit_intercept=True, normalize=True, copy_X=True,
                                           n_nonzero_coefs=50)).fit(mixed_data[:, include_inds].T.copy(),
                                                                    pred_labels[include_inds].copy())
            # collect active indices
            active_genes = []
            for e in range(len(cls.estimators_)):
                active_genes.extend(cls.estimators_[e].active_)
            active_genes = np.unique(active_genes)
            print(active_genes)
            print(active_genes.shape)
        else:
            cls = OneVsRestClassifier(LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)).fit(
                mixed_data[:, include_inds].T.copy(), pred_labels[include_inds].copy())

        ret = cls.predict(target_data[:, include_inds].T.copy())
        aris[4] = metrics.adjusted_rand_score(ret, pred_labels[include_inds].copy())

    aris[0] = unsupervised_acc_kta(target_data[active_genes, :].copy(), pred_labels.copy(), kernel='linear')
    aris[1] = unsupervised_acc_silhouette(target_data[active_genes, :].copy(), pred_labels.copy(), metric='euclidean')
    aris[2] = unsupervised_acc_silhouette(target_data[active_genes, :].copy(), pred_labels.copy(), metric='pearson')
    aris[3] = unsupervised_acc_silhouette(target_data[active_genes, :].copy(), pred_labels.copy(), metric='spearman')
    return aris



if __name__ == "__main__":
    path = "/Users/nicococo/Downloads/mouse_vis_cortex 2/all/"
    fname = 'res_gout_v1.npz'
    reps = 10
    n_source_cluster = 16
    n_target_cluster = 16
    n_source = 1024
    n_target = [30, 50, 100, 200, 400]
    # n_mix = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_mix = np.linspace(0., 1., 21)

    # reps = 5
    # n_mix = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    # n_target = [30, 50, 100]

    do_plot_results = True
    if do_plot_results:
        plot_results(fname)
        exit(0)

    data = np.load('{0}all_c16.npz'.format(path))
    data = data['src'][()]
    gene_ids = data.gene_ids[data.remain_gene_inds]

    print(data)
    print(np.unique(data.cluster_labels))

    source_ari = np.zeros(reps)
    target_ari = np.zeros((7, reps, len(n_target), len(n_mix)))
    n = 0
    while n < reps:
        source_perc = n_source / float(data.cluster_labels.size)
        sss = StratifiedShuffleSplit(n_splits=2, test_size=source_perc)
        for split_1, split_2 in sss.split(data.pp_data.T, data.cluster_labels):
            print(split_1.size, split_2.size)

        source_data = data.pp_data[:, split_2]
        source_labels = data.cluster_labels[split_2]

        # train source and test performance
        source_nmf = NmfClustering(source_data, gene_ids, num_cluster=n_source_cluster)
        source_nmf.apply(k=n_source_cluster, max_iter=4000, rel_err=1e-3)
        source_ari[n] = metrics.adjusted_rand_score(source_labels, source_nmf.cluster_labels)
        print('ITER(', n,'): SOURCE ARI = ', source_ari[n])
        if source_ari[n] < 0.94:
            continue

        for i in range(len(n_target)):
            target_perc = n_target[i] / float(split_1.size)
            ttt = StratifiedShuffleSplit(n_splits=2, test_size=target_perc)
            for split_11, split_22 in ttt.split(data.pp_data[:, split_1].T, data.cluster_labels[split_1]):
                print(split_11.size, split_22.size)

            # shuffle the gene ids for testing
            perm_inds = np.random.permutation(data.pp_data.shape[0])
            target_gene_ids = gene_ids[perm_inds].copy()
            target_data = data.pp_data[:, split_1[split_22]]
            target_data = target_data[perm_inds, :]
            target_labels = data.cluster_labels[split_1[split_22]]

            for m in range(len(n_mix)):
                target_nmf = DaNmfClustering(source_nmf, target_data.copy(), target_gene_ids, num_cluster=n_target_cluster)
                # target_nmf.apply(k=n_target_cluster, mix=n_mix[m], calc_transferability=False)


                mixed_data, rec_trg_data, _ = target_nmf.get_mixed_data(mix=n_mix[m], use_H2=True, calc_transferability=False)
                W, H, H2 = target_nmf.intermediate_model

                num_cells = target_data.shape[1]
                max_pca_comp = np.ceil(num_cells*0.07).astype(np.int)
                min_pca_comp = np.floor(num_cells*0.04).astype(np.int)
                sc3_mix = SC3Clustering(mixed_data, target_gene_ids, pc_range=[min_pca_comp, max_pca_comp], sub_sample=True, consensus_mode=0)
                sc3_mix.add_distance_calculation(partial(sc.distances, metric='euclidean'))
                sc3_mix.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
                sc3_mix.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_target_cluster))
                sc3_mix.set_build_consensus_matrix(sc.build_consensus_matrix)
                sc3_mix.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_target_cluster))
                sc3_mix.apply()

                pred_labels = sc3_mix.cluster_labels
                target_ari[0, n, i, m] = metrics.adjusted_rand_score(target_labels, pred_labels.copy())

                target_ari[1, n, i, m] = unsupervised_acc_kta(target_data.copy(), pred_labels.copy(), kernel='linear')
                target_ari[2, n, i, m] = unsupervised_acc_kta(rec_trg_data, pred_labels.copy(), kernel='linear')
                target_ari[3, n, i, m] = unsupervised_acc_kta(W.dot(H), pred_labels.copy(), kernel='linear')

                target_ari[[4,5], n, i, m] = get_classifier_scores(pred_labels, mixed_data, target_nmf.pp_data, target_data, use_lasso=False)[[1,4]]

                # target_ari[1:6, n, i, m] = get_classifier_scores(pred_labels, mixed_data, target_nmf.pp_data, target_data, use_lasso=True)
                # target_ari[[1,6], n, i, m] = get_classifier_scores(pred_labels, mixed_data, target_nmf.pp_data, target_data, use_lasso=False)[[0,4]]

                # target_ari[[1,6], n, i, m] = get_classifier_scores(target_labels, mixed_data, target_nmf.pp_data, target_data, use_lasso=False)[[0,4]]
                # target_ari[6, n, i, m] = np.sum(np.sum(mixed_data - target_data)) / float(mixed_data.size)
        n += 1

    print(source_ari)
    print(target_ari)

    np.savez(fname, source_ari=source_ari, target_ari=target_ari, n_mix=n_mix,
             n_source=n_source, n_target=n_target, n_source_cluster=n_source_cluster, n_target_cluster=n_target_cluster)
    plot_results(fname)
