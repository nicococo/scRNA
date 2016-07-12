import numpy as np

from functools import partial

import sc3_pipeline_impl as sc
from cluster_pipeline import ClusterPipeline
from utils import *


if __name__ == "__main__":

    src_data = np.random.rand(1000, 100)*10000.
    trgt_data = np.random.rand(1000, 100)*10000.

    cp = ClusterPipeline(trgt_data)

    np.random.seed(1)
    ks = range(3, 7+1)
    n_cluster = 7
    max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)

    # cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=20))
    # cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.98, non_zero_threshold=0))

    cp.set_data_transformation(sc.data_transformation)
    cp.add_distance_calculation(partial(sc.mtl_toy_distance, src_data=src_data, metric='euclidean', mixture=0.75))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    print('Done.')
