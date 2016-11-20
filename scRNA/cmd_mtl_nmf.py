import argparse, sys

from functools import partial
from sklearn.metrics import adjusted_rand_score

import mtl
import sc3_clustering_impl as sc
from sc3_clustering import SC3Clustering
from utils import *


# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="Target TSV dataset filename", required=True, type=str)
parser.add_argument("--flabels", help="Target TSV labels filename", default=None, type=str)
parser.add_argument("--fgeneids", help="Target TSV gene ids filename", required=True, type=str)
parser.add_argument("--fmtl", help="MTL source TSV dataset filename", required=True, type=str)
parser.add_argument("--fmtl_geneids", help="MTL source TSV gene ids filename", required=True, type=str)
parser.add_argument("--fout", help="Result filename", default='out', type=str)

parser.add_argument("--min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--non_zero_threshold", help="Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)
parser.add_argument("--perc_consensus_genes", help="(Gene filter) Filter genes that have a consensus greater than this value across all cells (default 0.98)", default=0.96, type = float)

parser.add_argument("--src_min_expr_genes", help="(Source cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--src_non_zero_threshold", help="(Source cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)
parser.add_argument("--src_perc_consensus_genes", help="(Source gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type = float)

parser.add_argument("--nmf_k", help="(NMF) Number of latent components (default 10)", default=7, type = int)
parser.add_argument("--nmf_alpha", help="(NMF) Regularization strength (default 1.0)", default=1., type = float)
parser.add_argument("--nmf_l1", help="(NMF) L1 regularization impact [0,1] (default 0.75)", default=0.75, type = float)

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
    help = "Transform data to log2(x+1)",
    dest = "transform",
    action = 'store_true'
)
parser.add_argument(
    "--no-transform",
    help = "Disable transform data to log2(x+1)",
    dest = "transform",
    action = 'store_false'
)
parser.set_defaults(transform = True)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print arguments

# 1. LOAD DATA
print("\nLoading target dataset ({0} with {1} gene ids).".format(arguments.fname, arguments.fgeneids))
dataset = arguments.fname
data, gene_ids, labels = load_dataset_tsv(dataset, arguments.fgeneids, flabels=arguments.flabels)
print('Found {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1]))

# 2. MTL NMF
print('\n')

num_transcripts, num_cells = data.shape

remain_cell_inds = np.arange(0, num_cells)
src_cell_filter_fun = lambda x: np.arange(x.shape[1]).tolist()
if arguments.use_cell_filter:
    src_cell_filter_fun = partial(sc.cell_filter, num_expr_genes=arguments.src_min_expr_genes, non_zero_threshold=arguments.src_non_zero_threshold)
    res = sc.cell_filter(data, num_expr_genes=arguments.min_expr_genes, non_zero_threshold=arguments.non_zero_threshold)
    remain_cell_inds = np.intersect1d(remain_cell_inds, res)
A = data[:, remain_cell_inds]

remain_gene_inds = np.arange(0, num_transcripts)
src_gene_filter_fun = lambda x: np.arange(x.shape[0]).tolist()
if arguments.use_gene_filter:
    src_gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=arguments.src_perc_consensus_genes, non_zero_threshold=arguments.src_non_zero_threshold)
    res = sc.gene_filter(data, perc_consensus_genes=arguments.perc_consensus_genes, non_zero_threshold=arguments.non_zero_threshold)
    remain_gene_inds = np.intersect1d(remain_gene_inds, res)
X = A[remain_gene_inds, :]

src_data_transf_fun = lambda x: x
if arguments.transform:
    src_data_transf_fun = sc.data_transformation_log2
    X = sc.data_transformation_log2(X)

gene_ids = gene_ids[remain_gene_inds]
print X.shape, np.min(X), np.max(X)

# 3. Do magic
print('\nNMF MTL:')
W, H, H2, Hsrc, reject, src_gene_inds, trg_gene_inds = mtl.nmf_mtl_full(X, gene_ids,
    fmtl=arguments.fmtl,
    fmtl_geneids=arguments.fmtl_geneids,
    nmf_alpha=arguments.nmf_alpha,
    nmf_k=arguments.nmf_k,
    nmf_l1=arguments.nmf_l1,
    data_transf_fun=src_data_transf_fun,
    cell_filter_fun=src_cell_filter_fun,
    gene_filter_fun=src_gene_filter_fun,
    max_iter=2000,
    rel_err=1e-3)

pred_lbls = np.argmax(H2, axis=0)

# decode reject
rej = np.zeros((remain_cell_inds.size, len(reject)))
rej_desc = list()
for n in range(len(reject)):
    rej[:, n] = reject[n][1]
    rej_desc.append(reject[n][0])

# Check if labels are available:
if labels is not None:
    print('\nLabels are available!')
    print 'Labels in truth: ', np.unique(labels[remain_cell_inds])
    print 'Labels in pred: ', np.unique(pred_lbls)
    print 'ARI for max-assignment: ', adjusted_rand_score(labels[remain_cell_inds], pred_lbls)

# 4. SAVE RESULTS
print('\nSaving data structures and results to \'{0}.npz\'.'.format(arguments.fout))
np.savez('{0}.npz'.format(arguments.fout), type='NMF-mtl',
         W=W, H=H, H2=H2, Hsrc=Hsrc, reject=rej, reject_desc=rej_desc,
         src_gene_inds=src_gene_inds, trg_gene_inds=trg_gene_inds,
         args=arguments)

print reject[0][1]
print reject[0][0]

print('\nSaving inferred labeling as TSV file to \'{0}.labels.tsv\'.'.format(arguments.fout))
np.savetxt('{0}.labels.tsv'.format(arguments.fout), (pred_lbls.T, remain_cell_inds.T, reject[0][1].T), fmt='%g', delimiter='\t')

print('Done.')
