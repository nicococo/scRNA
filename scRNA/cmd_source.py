import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import argparse, sys

from functools import partial
from sklearn.manifold import TSNE

from nmf_clustering import NmfClustering
from utils import *

# --------------------------------------------------
# PARSE COMMAND LINE ARGUMENTS
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="Source data (TSV file)", required=True, type=str, default=None)
parser.add_argument("--fgene-ids", help="Source data gene ids (TSV file)", dest='fgene_ids', required=True, type=str, default=None)
parser.add_argument("--fout", help="Result files will use this prefix.", default='src', type=str)
parser.add_argument("--flabels", help="[optional] Cluster labels (TSV file)", required=False, type=str, default=None)

parser.add_argument("--min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type=int)
parser.add_argument("--non_zero_threshold", help="(Cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type=float)
parser.add_argument("--perc_consensus_genes", help="(Gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type=float)

parser.add_argument("--cluster-range", help="Comma separated list of clusters (default 6,7,8)", dest='cluster_range', default='6,7,8', type=str)

parser.add_argument("--nmf_alpha", help="(NMF) Regularization strength (default 1.0)", default=1.0, type = float)
parser.add_argument("--nmf_l1", help="(NMF) L1 regularization impact [0,1] (default 0.75)", default=0.75, type = float)
parser.add_argument("--nmf_max_iter", help="(NMF) Maximum number of iterations (default 4000).", default=4000, type = int)
parser.add_argument("--nmf_rel_err", help="(NMF) Relative error threshold must be reached before convergence (default 1e-3)", default=1e-3, type=float)

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
    action = 'store_true')
parser.add_argument(
    "--no-transform",
    help = "Disable transform data to log2(x+1)",
    dest = "transform",
    action = 'store_false')
parser.set_defaults(transform = True)

parser.add_argument(
    "--tsne",
    help = "Enable t-SNE plots.",
    dest = "tsne",
    action = 'store_true')
parser.add_argument(
    "--no-tsne",
    help = "Disable t-SNE plots.",
    dest = "tsne",
    action = 'store_false')
parser.set_defaults(tsne=True)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print("\nLoading  dataset (data={0} and gene_ids={1}).".format(arguments.fname, arguments.fgene_ids))
data, gene_ids, labels, labels_2_ids = load_dataset_tsv(arguments.fname, arguments.fgene_ids, flabels=arguments.flabels)
print('Data  {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1]))
print np.unique(labels)

print('Number of genes/transcripts in data and gene-ids must coincide.')
assert(data.shape[0] == gene_ids.shape[0])

# --------------------------------------------------
# 2. CELL and GENE FILTER
# --------------------------------------------------
cell_filter_fun = lambda x: np.arange(x.shape[1]).tolist()
if arguments.use_cell_filter:
    cell_filter_fun = partial(sc.cell_filter, num_expr_genes=arguments.min_expr_genes, non_zero_threshold=arguments.non_zero_threshold)

gene_filter_fun = lambda x: np.arange(x.shape[0]).tolist()
if arguments.use_gene_filter:
    gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=arguments.perc_consensus_genes, non_zero_threshold=arguments.non_zero_threshold)

data_transf_fun = lambda x: x
if arguments.transform:
    data_transf_fun = sc.data_transformation_log2

# --------------------------------------------------
# 3. CLUSTERING
# --------------------------------------------------
num_cluster = map(np.int, arguments.cluster_range.split(","))

accs_names = ['KTA (linear)', 'Silhouette (euc)', 'Silhouette (pearson)', 'Silhouette (spearman)', 'ARI']
accs = np.zeros((5, len(num_cluster)))

for i in range(len(num_cluster)):
    k = num_cluster[i]
    print('Iteration {0}, num-cluster={0}'.format(i, k))

    # --------------------------------------------------
    # 3.1. SETUP SOURCE DATA NMF CLUSTERING
    # --------------------------------------------------
    nmf = None
    nmf = NmfClustering(data, gene_ids, num_cluster=k)
    nmf.add_cell_filter(cell_filter_fun)
    nmf.add_gene_filter(gene_filter_fun)
    nmf.set_data_transformation(data_transf_fun)
    nmf.apply(k=k, alpha=arguments.nmf_alpha, l1=arguments.nmf_l1, max_iter=arguments.nmf_max_iter, rel_err=arguments.nmf_rel_err)

    # --------------------------------------------------
    # 3.2. EVALUATE CLUSTER ASSIGNMENT
    # --------------------------------------------------
    print('\nUnsupervised evaluation:')
    accs[0, i] = unsupervised_acc_kta(nmf.pp_data, nmf.cluster_labels, kernel='linear')
    accs[1, i] = unsupervised_acc_silhouette(nmf.pp_data, nmf.cluster_labels, metric='euclidean')
    accs[2, i] = unsupervised_acc_silhouette(nmf.pp_data, nmf.cluster_labels, metric='pearson')
    accs[3, i] = unsupervised_acc_silhouette(nmf.pp_data, nmf.cluster_labels, metric='spearman')
    print '  -KTA (linear)     : ', accs[0, i]
    print '  -Silhouette (euc) : ', accs[1, i]
    print '  -Silhouette (pear): ', accs[2, i]
    print '  -Silhouette (spea): ', accs[3, i]
    if labels is not None:
        print('\nSupervised evaluation:')
        accs[4, i] = metrics.adjusted_rand_score(
            labels[nmf.remain_cell_inds], nmf.cluster_labels)
        print '  -ARI: ', accs[4, i]

    # --------------------------------------------------
    # 3.3. SAVE RESULTS
    # --------------------------------------------------
    nmf.cell_filter_list = None
    nmf.gene_filter_list = None
    nmf.data_transf = None
    print('\nSaving data structures and results to file with prefix \'{0}_c{1}\'.'.format(arguments.fout, k))
    np.savez('{0}_c{1}.npz'.format(arguments.fout, k), src=nmf, args=arguments)
    np.savetxt('{0}_c{1}.labels.tsv'.format(arguments.fout, k),
               (nmf.cluster_labels, nmf.remain_cell_inds), fmt='%u', delimiter='\t')

    # --------------------------------------------------
    # 3.4. T-SNE PLOT
    # --------------------------------------------------
    if arguments.tsne:
        model = TSNE(n_components=2, random_state=0, init='pca', method='exact', metric='euclidean', perplexity=30)
        ret = model.fit_transform(nmf.pp_data.T)
        plt.title('{0} cluster (Euclidean)'.format(k))
        plt.scatter(ret[:, 0], ret[:, 1], 20, nmf.cluster_labels)
        plt.xticks([])
        plt.yticks([])

        plt.savefig('{0}_c{1}.tsne.png'.format(arguments.fout, k), format='png', bbox_inches=None, pad_inches=0.1)
        # plt.show()

# --------------------------------------------------
# 6. SUMMARIZE RESULTS
# --------------------------------------------------
print '\n------------------------------ Summary:'
print 'Cluster:', num_cluster
print 'Accuracy measures: ', accs_names
print 'Accuracies:'
print accs

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

print '\n\n\n'
print '================================================================================'
print '\n\n\n'
print 'SUMMARY'
print '\n\n\n'
print 'Parameters'
print '-------------'
print ' - Output prefix: ', arguments.fout
print ' - Source file name: ', arguments.fname
print ' - Cluster:', num_cluster
if labels is not None:
    print ' - Class 2 label conversion (class {0:1d}-{1:1d}): '.format(
        np.int(np.min(labels)), np.int(np.max(labels))), labels_2_ids
print ''

print 'Results'
print '-------------'
print ' - Accuracies: ', accs_names
for i in range(accs.shape[0]):
    print('\n{0} (cluster({1})):'.format(accs_names[i], len(num_cluster)))
    print accs[i, :]

plt.figure(0, figsize=(20,5), dpi=100)
fig, axes = plt.subplots(nrows=1, ncols=accs.shape[0])
fig.tight_layout(h_pad=1.08, pad=2.2) # Or equivalently,  "plt.tight_layout()"
for i in range(accs.shape[0]):
    plt.subplot(1, accs.shape[0], i+1)

    if i % 2 == 0:
        plt.title(accs_names[i] + '\n', fontsize=12, fontweight='bold')
    else:
        plt.title('\n' + accs_names[i], fontsize=12, fontweight='bold')

    if i == accs.shape[0]-1:
        plt.bar(range(len(num_cluster)), accs[i, :], color='red')
    else:
        plt.bar(range(len(num_cluster)), accs[i, :])

    if i == 0:
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)

    plt.yticks(fontsize=8)
    plt.xticks(np.array(range(len(num_cluster)), dtype=np.float)+0.5, num_cluster, fontsize=8)
    plt.grid('on')

plt.savefig('{0}.accs.png'.format(arguments.fout), format='png',
            bbox_inches=None, pad_inches=0.1, dpi=100)
# plt.show()

print('\nDone.')
