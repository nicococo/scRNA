import numpy as np

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.manifold as manifold
import sklearn.preprocessing as pp
from functools import partial

import mtl
import sc3_pipeline_impl as sc
from sc3_pipeline import SC3Pipeline
from utils import *


if __name__ == "__main__":
    data, gene_ids = load_dataset_by_name('Usoskin')

    mixture_coeffs = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    mixture_coeffs = [0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
    all_labels = []
    all_ktas = []
    for i in range(len(mixture_coeffs)):
        cp = SC3Pipeline(data, gene_ids)
        np.random.seed(1)
        n_cluster = 11
        max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
        min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)

        cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=2000))
        cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=2))
        cp.set_data_transformation(sc.data_transformation)

        cp.add_distance_calculation(partial(mtl.mtl_distance, metric='euclidean', mixture=mixture_coeffs[i]))

        # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
        cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

        cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
        cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
        cp.apply(pc_range=[min_pca_comp, max_pca_comp])
        print cp

        Y = cp.filtered_transf_data
        Y = pp.scale(Y, axis=1, with_mean=True, with_std=False)

        model = manifold.TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=1234567, init='pca')
        out = model.fit_transform(Y.T)

        plt.figure(5)
        plt.subplot(2, 3, i+1)
        plt.title('t-SNE (mixture={0})'.format(mixture_coeffs[i]))
        model = manifold.TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=1234567, metric='precomputed')
        out = model.fit_transform(cp.dists)
        plt.scatter(out[:, 0], out[:, 1], s=20, c=cp.cluster_labels)
        plt.grid('on')
        plt.xticks([])
        plt.yticks([])

        K1 = Y.T.dot(Y)
        K2 = cp.cluster_labels.reshape((Y.shape[1],1)).dot(cp.cluster_labels.reshape((1, Y.shape[1])))
        all_ktas.append(kta_align_general(K1, K2))
        all_labels.append(cp.cluster_labels)
        print 'KTA scores CP: ', kta_align_general(K1, K2)

    print 'KTAs: ', all_ktas

    lbl = np.array(all_labels, dtype=np.int)
    print 'Results: Lbl-mtx=', lbl.shape

    comp = np.zeros((len(all_labels), len(all_labels)))
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            comp[i, j] = metrics.adjusted_rand_score(lbl[i, :], lbl[j, :])
    print comp
    plt.show()

    print('Done.')
