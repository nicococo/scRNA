import numpy as np

from functools import partial

import sc3_pipeline_impl as sc
from cluster_pipeline import ClusterPipeline


if __name__ == "__main__":
    foo = np.load('pfizer_data.npz')
    foo = np.load('/Users/nicococo/Documents/scRNA-data/Usoskin.npz')
    # data = foo['data']  # transcripts x cells

    # filtered_inds=foo['filtered_inds']
    data  = foo['data']
    # rpm_data = foo['rpm_data']
    # transcripts = foo['transcripts']
    # transcripts_header = foo['transcripts_header']
    # xlsx_data = foo['xlsx_data']
    # xlsx_header = foo['xlsx_header']

    # X = rpm_data[:, filtered_inds]
    # X = data
    # print X.shape
    # np.savetxt('test.csv', X, fmt='%i', delimiter=',', newline='\n')

    cp = ClusterPipeline(data)

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=2000))
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.94, non_zero_threshold=2))

    cp.set_data_transformation(sc.data_transformation)

    cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))
    # cp.add_distance_calculation(partial(sc.distances, metric='pearson'))
    # cp.add_distance_calculation(partial(sc.distances, metric='spearman'))
    # cp.add_distance_calculation(partial(sc.distances, metric='chebychev'))

    cp.add_dimred_calculation(partial(sc.transformations, components=50, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=12, method='spectral'))

    cp.set_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=7))

    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=11))

    cp.apply(pc_range=[24, 36])

    print('Done.')
