import argparse, sys

from functools import partial
from sklearn.metrics import adjusted_rand_score

import sc3_clustering_impl as sc
from sc3_clustering import SC3Clustering
from utils import *


# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="Target TSV dataset filename", required=True, type=str)
parser.add_argument("--flabels", help="Target TSV labels filename", default=None, type=str)
parser.add_argument("--fout", help="Result filename (no extension)", default='out', type=str)

parser.add_argument("--min_expr_genes", help="(Target cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--non_zero_threshold", help="(Target cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)
parser.add_argument("--perc_consensus_genes", help="(Target gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type = float)

parser.add_argument("--sc3_k", help="(SC3) Number of latent components (default 10)", default=10, type = int)
parser.add_argument("--sc3_dists", help="(SC3) Comma-separated distances (default euclidean)", default='euclidean', type = str)
parser.add_argument("--sc3_transf", help="(SC3) Comma-separated transformations (default pca)", default='pca', type = str)

parser.add_argument(
    "--cell-filter",
    help = "Enable cell filter for source and target datasets.",
    dest = "use_cell_filter",
    action = 'store_true')
parser.add_argument(
    "--no-cell-filter",
    help = "Disable cell filter for source and target datasets.",
    dest = "use_cell_filter",
    action = 'store_false')
parser.set_defaults(use_cell_filter = True)

parser.add_argument(
    "--gene-filter",
    help = "Enable gene filter for source and target datasets.",
    dest = "use_gene_filter",
    action = 'store_true')
parser.add_argument(
    "--no-gene-filter",
    help = "Disable gene filter for source and target datasets.",
    dest = "use_gene_filter",
    action = 'store_false')
parser.set_defaults(use_gene_filter = True)

parser.add_argument(
    "--transform",
    help="Transform data to log2(x+1)",
    dest="transform",
    action='store_true'
)
parser.add_argument(
    "--no-transform",
    help="Disable transform data to log2(x+1)",
    dest="transform",
    action='store_false'
)
parser.set_defaults(transform=True)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print arguments

# 1. LOAD DATA
print("\nLoading target dataset ({0}).".format(arguments.fname))
dataset = arguments.fname
data, _, labels = load_dataset_tsv(dataset, flabels=arguments.flabels)
print('Found {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1]))

# 2. BUILD SC3 PIPELINE
print('\n')
n_cluster = arguments.sc3_k
num_cells = data.shape[1]
max_pca_comp = np.ceil(num_cells*0.07).astype(np.int)
min_pca_comp = np.floor(num_cells*0.04).astype(np.int)
print('(Max/Min) PCA components: ({0}/{1})'.format(max_pca_comp, min_pca_comp))

cp = SC3Clustering(data, np.arange(data.shape[0]), pc_range=[min_pca_comp, max_pca_comp], sub_sample=True, consensus_mode=0)

if arguments.use_cell_filter:
    cp.add_cell_filter(partial(sc.cell_filter, num_expr_genes=arguments.min_expr_genes, non_zero_threshold=arguments.non_zero_threshold))
if arguments.use_gene_filter:
    cp.add_gene_filter(partial(sc.gene_filter, perc_consensus_genes=arguments.perc_consensus_genes, non_zero_threshold=arguments.non_zero_threshold))
if arguments.transform:
    cp.set_data_transformation(sc.data_transformation_log2)

dist_list = arguments.sc3_dists.split(",")
print('\nThere are {0} distances given.'.format(len(dist_list)))
for ds in dist_list:
    print('- Adding distance {0}'.format(ds))
    cp.add_distance_calculation(partial(sc.distances, metric=ds))

transf_list = arguments.sc3_transf.split(",")
print('\nThere are {0} transformations given.'.format(len(transf_list)))
for ts in transf_list:
    print('- Adding transformation {0}'.format(ts))
    cp.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method=ts))

cp.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=n_cluster))
cp.set_build_consensus_matrix(sc.build_consensus_matrix)
cp.set_consensus_clustering(partial(sc.consensus_clustering, n_components=n_cluster))
cp.apply()

print cp

# Check if labels are available:
if labels is not None:
    print('\nLabels are available!')
    print 'ARI for max-assignment: ', adjusted_rand_score(labels[cp.remain_cell_inds], cp.cluster_labels)

# 4. SAVE RESULTS
cp.cell_filter_list = None
cp.gene_filter_list = None
cp.data_transf = None
cp.dists_list = None
print('\nSaving data structures and results to \'{0}.npz\'.'.format(arguments.fout))
np.savez('{0}.npz'.format(arguments.fout), type='SC3-single', sc3_pipeline=cp, args=arguments)

print('\nSaving inferred labeling as TSV file to \'{0}.labels.tsv\'.'.format(arguments.fout))
np.savetxt('{0}.labels.tsv'.format(arguments.fout), (cp.cluster_labels, cp.remain_cell_inds), fmt='%u', delimiter='\t')


print('Done.')
