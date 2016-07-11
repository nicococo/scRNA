import numpy as np

from functools import partial

import sc3_pipeline_impl as sc
from cluster_pipeline import ClusterPipeline
from utils import *


if __name__ == "__main__":
    dataset = 'Ting'
    # dataset = 'Pfizer'
    # dataset = 'Usoskin'

    data, gene_ids = load_dataset(dataset)

    # X = rpm_data[:, filtered_inds]
    # X = data
    # print foo.keys()
    # transcr_names = foo["transcripts"]
    # cell_names = foo['cells']
    # print X.shape, transcr_names.shape
    # X = np.hstack([transcr_names.reshape(X.shape[0],1), X])
    # cell_names = np.array2string(cell_names, separator=',', prefix='')[1:-1].replace('\'','').replace('\n','')
    # print cell_names
    # print X.shape
    # fmt=list(['%s'])
    # fmt.extend(['%1.2f']*(X.shape[1]-1))
    # # np.savetxt('ting.csv', X, fmt=fmt, delimiter=',', newline='\n', header=cell_names, comments='')
    # # np.savetxt('ting.csv', X, fmt=fmt, delimiter=',', newline='\n', comments='')
    # np.savetxt('ting.csv', X[:,1:], fmt='%1.2f', delimiter=',', newline='\n', comments='')

    cp = ClusterPipeline(data, gene_ids)

    np.random.seed(1)
    ks = range(3, 7+1)
    n_cluster = 7
    max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)

    if dataset != 'Ting':
        cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=2000))
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=2))

    cp.set_data_transformation(sc.data_transformation)

    # cp.add_distance_calculation(partial(sc.mtl_distance, metric='euclidean', mixture=0.75))
    cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))
    # cp.add_distance_calculation(partial(sc.distances, metric='pearson'))
    # cp.add_distance_calculation(partial(sc.distances, metric='spearman'))
    # cp.add_distance_calculation(partial(sc.distances, metric='chebychev'))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    # cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=np.min(ks)-1))
    # for k in ks:
    #     cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=k-1))

    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))

    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    print cp

    import matplotlib.pyplot as plt
    import sklearn.manifold as manifold
    import sklearn.metrics as metrics
    import sklearn.preprocessing as pp


    Y = cp.filtered_transf_data
    Y = pp.scale(Y, axis=1, with_mean=True, with_std=False)

    print Y.shape

    model = manifold.TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=1234567, init='pca')
    out = model.fit_transform(Y.T)
    # model = manifold.TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=0, metric='precomputed')
    # out = model.fit_transform(sc.distances(Y, [], metric='euclidean'))
    print out.shape
    plt.figure(5)

    plt.subplot(1, 2, 1)
    plt.title('t-SNE (filtered, transformed data)')
    plt.scatter(out[:, 0], out[:, 1], s=20, c=cp.cluster_labels)
    plt.grid('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    if dataset == 'Ting':
        plt.title('t-SNE (SC3 labels)')
        plt.scatter(out[:, 0], out[:, 1], s=20, c=sc.get_sc3_Ting_labels())
    else:
        plt.title('t-SNE (distance matrix)')
        model = manifold.TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=1234567, metric='precomputed')
        out = model.fit_transform(cp.dists)
        plt.scatter(out[:, 0], out[:, 1], s=20, c=cp.cluster_labels)

    plt.grid('on')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    K1 = Y.T.dot(Y)
    K2 = cp.cluster_labels.reshape((Y.shape[1],1)).dot(cp.cluster_labels.reshape((1, Y.shape[1])))
    print 'KTA scores CP: ', kta_align_general(K1, K2)
    if dataset == 'Ting':
        K3 = sc.get_sc3_Ting_labels().reshape((Y.shape[1],1)).dot(sc.get_sc3_Ting_labels().reshape((1, Y.shape[1])))
        print 'KTA scores SC3:', kta_align_general(K1, K3)
        print 'Result SC3-CP: ', \
            metrics.adjusted_rand_score(sc.get_sc3_Ting_labels(), cp.cluster_labels)

    print('Done.')
