import argparse, sys

import sklearn.decomposition as decomp
from sklearn.metrics import adjusted_rand_score

import sc3_pipeline_impl as sc
from utils import *

# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="Target TSV dataset filename", required=True, type=str)
parser.add_argument("--flabels", help="Target TSV labels filename", default=None, type=str)
parser.add_argument("--fout", help="Result filename (no extension)", default='out', type=str)

parser.add_argument("--cf_min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--cf_non_zero_threshold", help="(Cell filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)

parser.add_argument("--gf_perc_consensus_genes", help="(Gene filter) Filter genes that have a consensus greater than this value across all cells (default 0.98)", default=0.98, type = float)
parser.add_argument("--gf_non_zero_threshold", help="(Gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)

parser.add_argument("--nmf_k", help="(NMF) Number of latent components (default 10)", default=10, type = int)
parser.add_argument("--nmf_alpha", help="(NMF) Regularization strength (default 1.0)", default=1.0, type = float)
parser.add_argument("--nmf_l1", help="(NMF) L1 regularization impact [0,1] (default 0.75)", default=0.75, type = float)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print arguments

# 1. LOAD DATA
print("\nLoading target dataset ({0}).".format(arguments.fname))
dataset = arguments.fname
data, _, labels = load_dataset_tsv(dataset, flabels=arguments.flabels)
print('Found {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1]))

num_transcripts, num_cells = data.shape
remain_cell_inds = np.arange(0, num_cells)

# 2. FILTER DATA (GENES AND TRANSCRIPTS)
res = sc.cell_filter(data, num_expr_genes=arguments.cf_min_expr_genes, non_zero_threshold=arguments.cf_non_zero_threshold)
remain_cell_inds = np.intersect1d(remain_cell_inds, res)
A = data[:, remain_cell_inds]

remain_gene_inds = np.arange(0, num_transcripts)
res = sc.gene_filter(data, perc_consensus_genes=0.98, non_zero_threshold=arguments.gf_non_zero_threshold)
remain_gene_inds = np.intersect1d(remain_gene_inds, res)
X = sc.data_transformation(A[remain_gene_inds, :])
print X.shape, np.min(X), np.max(X)
num_transcripts, num_cells = X.shape

# 3. NMF
print('\nNon-negative matrix factorization (k={0})'.format(arguments.nmf_k))
nmf = decomp.NMF(alpha=arguments.nmf_alpha, init='nndsvdar', l1_ratio=arguments.nmf_l1, max_iter=1000,
    n_components=arguments.nmf_k, random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
W = nmf.fit_transform(X)
H = nmf.components_

print('Some NMF result statistics:')
print nmf
print 'Elementwise absolute reconstruction error: ', np.sum(np.abs(X - W.dot(H)))/np.float(X.size)
print 'Frobenius-norm reconstruction error: ', np.sqrt(np.sum((X - W.dot(H))*(X - W.dot(H))))
print 'dim(W): ', W.shape
print 'dim(H): ', H.shape

# Check if labels are available:
if not labels is None:
    print('\nLabels are available!')
    print 'ARI for max-assignment: ', adjusted_rand_score(labels[remain_cell_inds], np.argmax(H, axis=0))

# 4. SAVE RESULTS
print('\nSaving data structures and results to \'{0}.npz\'.'.format(arguments.fout))
np.savez('{0}.npz'.format(arguments.fout), type='NMF-single', X=X, W=W, H=H, labels=np.argmax(H, axis=0),
         args=arguments, remain_cell_inds=remain_cell_inds, remain_gene_inds=remain_gene_inds)

print('\nSaving inferred labeling as TSV file to \'{0}.labels.tsv\'.'.format(arguments.fout))
np.savetxt('{0}.labels.tsv'.format(arguments.fout), (np.argmax(H, axis=0), remain_cell_inds), fmt='%u', delimiter='\t')


print('Done.')