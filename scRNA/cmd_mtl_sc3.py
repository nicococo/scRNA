import argparse, sys

from functools import partial
from sklearn.metrics import adjusted_rand_score

import sc3_pipeline_impl as sc
from sc3_pipeline import SC3Pipeline
from utils import *


# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="Target dataset filename", required=True, type=str)
parser.add_argument("--fmtl", help="MTL source dataset filename", required=True, type=str)
parser.add_argument("--fout", help="Result filename", default='out.npz', type=str)

parser.add_argument("--cf_min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--cf_non_zero_threshold", help="(Cell filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)

parser.add_argument("--gf_perc_consensus_genes", help="(Gene filter) Filter genes that have a consensus greater than this value across all cells (default 0.98)", default=0.98, type = float)
parser.add_argument("--gf_non_zero_threshold", help="(Gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)

parser.add_argument("--sc3_k", help="(SC3) Number of latent components (default 10)", default=10, type = int)
parser.add_argument("--sc3_dists", help="(SC3) Comma-separated MTL distances (default euclidean)", default='euclidean', type = str)
parser.add_argument("--sc3_transf", help="(SC3) Comma-separated transformations (default pca)", default='pca', type = str)

parser.add_argument("--mtl_mixture", help="(MTL) Convex combination mixture coefficient (0.=no transfer, default 0.1)", default=0.1, type = float)

parser.add_argument("--nmf_k", help="(NMF) Number of latent components (default 10)", default=10, type = int)
parser.add_argument("--nmf_alpha", help="(NMF) Regularization strength (default 1.0)", default=1.0, type = float)
parser.add_argument("--nmf_l1", help="(NMF) L1 regularization impact [0,1] (default 0.75)", default=0.75, type = float)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print arguments

# 1. LOAD DATA
print("\nLoading target dataset ({0}).".format(arguments.fname))
dataset = arguments.fname
data, gene_ids, labels = load_dataset(dataset)
print('Found {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1]))

# 2. BUILD SC3 PIPELINE
print('\n')
cp = SC3Pipeline(data, gene_ids)

n_cluster = arguments.sc3_k
max_pca_comp = np.ceil(cp.num_cells*0.07).astype(np.int)
min_pca_comp = np.floor(cp.num_cells*0.04).astype(np.int)
print('(Max/Min) PCA components: ({0}/{1})'.format(max_pca_comp, min_pca_comp))

cp.add_cell_filter(partial(sc.cell_filter, num_expr_genes=arguments.cf_min_expr_genes, non_zero_threshold=arguments.cf_non_zero_threshold))
cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=arguments.gf_perc_consensus_genes, non_zero_threshold=arguments.gf_non_zero_threshold))

cp.set_data_transformation(sc.data_transformation)

dist_list = arguments.sc3_dists.split(",")
print('\nThere are {0} distances given.'.format(len(dist_list)))
for ds in dist_list:
    print('- Adding MTL distance {0}'.format(ds))
    cp.add_distance_calculation(partial(sc.mtl_distance, fmtl=arguments.fmtl, metric=ds, mixture=arguments.mtl_mixture,
        nmf_alpha=arguments.nmf_alpha, nmf_k=arguments.nmf_k, nmf_l1=arguments.nmf_l1))

transf_list = arguments.sc3_transf.split(",")
print('\nThere are {0} transformations given.'.format(len(transf_list)))
for ts in transf_list:
    print('- Adding transformation {0}'.format(ts))
    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method=ts))

cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
cp.apply(pc_range=[min_pca_comp, max_pca_comp])

print cp

# Check if labels are available:
if not labels is None:
    print('\nLabels are available!')
    print 'ARI for max-assignment: ', adjusted_rand_score(labels[cp.remain_cell_inds], cp.cluster_labels)

# 4. SAVE RESULTS
print('\nSaving results to \'{0}\'.'.format(arguments.fout))
np.savez(arguments.fout, type='SC3-single', sc3_pipeline=cp, args=arguments)

np.savetxt('{0}.labels'.format(arguments.fout), (cp.cluster_labels, cp.remain_cell_inds), fmt='%u')


print('Done.')
