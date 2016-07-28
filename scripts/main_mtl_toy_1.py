from functools import partial

from sklearn.metrics import adjusted_rand_score

import sc3_pipeline_impl as sc
from sc3_pipeline import SC3Pipeline
from simulation import generate_toy_data
from utils import *

if __name__ == "__main__":
    # Data generation parameters
    num_genes = 5000  # 20000
    num_cells = 400  # 400
    true_num_clusters = 10  # 5
    dirichlet_parameter_cluster_size = 0.1  # 1, between 0 and inf, smaller values make cluster sizes more similar
    shape_power_law = 0.01  # 0.1, shape parameter of the power law -  between 0 and 1, the smaller this value the more extreme the power law
    upper_bound_counts = 1000000  # 1000000
    dirichlet_parameter_counts = 0.1  # 1, between 0 and inf, noise parameter: smaller values make counts within cluster more similar (splitting the total
    # abundances in more equal parts for each cell)

    # Generate toy data
    data, labels = generate_toy_data(num_genes=num_genes, num_cells=num_cells,num_clusters=true_num_clusters,
                                        dirichlet_parameter_cluster_size=dirichlet_parameter_cluster_size, shape_power_law=shape_power_law,
                                        upper_bound_counts=upper_bound_counts,dirichlet_parameter_counts=dirichlet_parameter_counts)

    cp = SC3Pipeline(data)

    # np.random.seed(1)
    ks = range(3, 7+1)
    n_cluster = 7
    max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
    min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)

    cp.add_cell_filter(partial(sc.cell_filter, non_zero_threshold=2, num_expr_genes=100))
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=0.98, non_zero_threshold=0))

    cp.set_data_transformation(sc.data_transformation)
    cp.add_distance_calculation(partial(sc.distances, metric='euclidean'))

    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='pca'))
    # cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method='spectral'))

    cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
    cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
    cp.apply(pc_range=[min_pca_comp, max_pca_comp])

    print labels
    print adjusted_rand_score(labels, cp.cluster_labels)

    print('Done.')
