import matplotlib.pyplot as plt
import argparse, sys

from functools import partial
import sklearn.metrics as metrics

import sc3_clustering_impl as sc
from sc3_clustering import SC3Clustering
from nmf_clustering import DaNmfClustering, NmfClustering
from utils import *

# --------------------------------------------------
# PARSE COMMAND LINE ARGUMENTS
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src_name", help="Target TSV dataset filename", required=False, type=str, default=None)
parser.add_argument("--src_geneids", help="Target TSV gene ids filename", required=False, type=str, default=None)
parser.add_argument("--trg_name", help="MTL source TSV dataset filename", required=True, type=str)
parser.add_argument("--trg_labels", help="Target TSV ground truth labels filename", required=False, default=None, type=str)
parser.add_argument("--trg_geneids", help="MTL source TSV gene ids filename", required=True, type=str)

parser.add_argument("--fout", help="Result files will use this prefix.", default='out', type=str)

parser.add_argument("--trg_min_expr_genes", help="(Target cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--trg_non_zero_threshold", help="(Target cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)
parser.add_argument("--trg_perc_consensus_genes", help="(Target gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type = float)

parser.add_argument("--src_min_expr_genes", help="(Source cell filter) Minimum number of expressed genes (default 2000)", default=2000, type = int)
parser.add_argument("--src_non_zero_threshold", help="(Source cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type = float)
parser.add_argument("--src_perc_consensus_genes", help="(Source gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type = float)

parser.add_argument("--src_k", help="(Source) Number of latent components (default 10)", default=7, type = int)
parser.add_argument("--trg_ks", help="(Target)Comma separated list of latent components (default 10)", default="2,4,7,8", type = str)

parser.add_argument("--mixtures", help="Comma separated list of convex combination src-trg mixture coefficient (0.=no transfer, default 0.1)", default="0.0,0.25,0.5,0.75,1.0", type = str)
parser.add_argument("--method", help="Clustering method ('SC3' or 'NMF')", default="SC3", type = str)

parser.add_argument("--nmf_alpha", help="(NMF) Regularization strength (default 1.0)", default=1.0, type = float)
parser.add_argument("--nmf_l1", help="(NMF) L1 regularization impact [0,1] (default 0.75)", default=0.75, type = float)
parser.add_argument("--nmf_max_iter", help="(NMF) Maximum number of iterations (default 4000).", default=4000, type = int)
parser.add_argument("--nmf_rel_err", help="(NMF) Relative error threshold must be reached before convergence (default 1e-4)", default=1e-3, type = float)

parser.add_argument("--sc3_dists", help="(SC3) Comma-separated MTL distances (default euclidean)", default='euclidean', type = str)
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
    help = "Transform data to log2(x+1)",
    dest = "transform",
    action = 'store_true')
parser.add_argument(
    "--no-transform",
    help = "Disable transform data to log2(x+1)",
    dest = "transform",
    action = 'store_false')
parser.set_defaults(transform = True)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print("\nLoading target dataset (data={0} and gene_ids={1}).".format(arguments.trg_name, arguments.trg_geneids))
trg_data, trg_gene_ids, trg_labels = load_dataset_tsv(arguments.trg_name, arguments.trg_geneids, flabels=arguments.trg_labels)
print('Target data  {1} cells and {0} genes/transcripts.'.format(trg_data.shape[0], trg_data.shape[1]))
print np.unique(trg_labels)

src_data = None
if arguments.src_name is not None:
    print("\nLoading source dataset (data={0} and gene_ids={1}).".format(arguments.src_name, arguments.src_geneids))
    src_data, src_gene_ids, _ = load_dataset_tsv(arguments.src_name, arguments.src_geneids)
    print('Source data  {1} cells and {0} genes/transcripts.'.format(src_data.shape[0], src_data.shape[1]))

# --------------------------------------------------
# 2. CELL and GENE FILTER
# --------------------------------------------------
trg_cell_filter_fun = lambda x: np.arange(x.shape[1]).tolist()
src_cell_filter_fun = lambda x: np.arange(x.shape[1]).tolist()
if arguments.use_cell_filter:
    trg_cell_filter_fun = partial(sc.cell_filter, num_expr_genes=arguments.trg_min_expr_genes, non_zero_threshold=arguments.trg_non_zero_threshold)
    src_cell_filter_fun = partial(sc.cell_filter, num_expr_genes=arguments.src_min_expr_genes, non_zero_threshold=arguments.src_non_zero_threshold)

trg_gene_filter_fun = lambda x: np.arange(x.shape[0]).tolist()
src_gene_filter_fun = lambda x: np.arange(x.shape[0]).tolist()
if arguments.use_gene_filter:
    trg_gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=arguments.trg_perc_consensus_genes, non_zero_threshold=arguments.trg_non_zero_threshold)
    src_gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=arguments.src_perc_consensus_genes, non_zero_threshold=arguments.src_non_zero_threshold)

trg_data_transf_fun = lambda x: x
src_data_transf_fun = lambda x: x
if arguments.transform:
    trg_data_transf_fun = sc.data_transformation_log2
    src_data_transf_fun = sc.data_transformation_log2

# --------------------------------------------------
# 3. CLUSTERING
# --------------------------------------------------
mixtures = map(np.float, arguments.mixtures.split(","))
num_cluster = map(np.int, arguments.trg_ks.split(","))

accs_names = ['Calinski-Harabaz', 'Silhouette (euc)', 'Silhouette (corr)', 'Silhouette (jacc)', 'ARI']
accs = np.zeros((5, len(mixtures), len(num_cluster)))

for i in range(len(num_cluster)):
    for j in range(len(mixtures)):
        print('Iteration k={0} mix={1}')
        trg_k = num_cluster[i]
        mix = mixtures[j]

        # --------------------------------------------------
        # 3.1. SETUP SOURCE DATA NMF CLUSTERING
        # --------------------------------------------------
        src_clustering = None
        if src_data is not None:
            src_clustering = NmfClustering(src_data, src_gene_ids, num_cluster=arguments.src_k)
            src_clustering.add_cell_filter(src_cell_filter_fun)
            src_clustering.add_gene_filter(src_gene_filter_fun)
            src_clustering.set_data_transformation(src_data_transf_fun)

        # --------------------------------------------------
        # 3.2. SETUP TARGET DATA CLUSTERING
        # --------------------------------------------------
        if arguments.method is 'NMF' and src_data is not None:
            print('Transfer learning method is NMF.')
            trg_clustering = DaNmfClustering(src_clustering, trg_data, trg_gene_ids, num_cluster=trg_k)
            trg_clustering.add_cell_filter(trg_cell_filter_fun)
            trg_clustering.add_gene_filter(trg_gene_filter_fun)
            trg_clustering.set_data_transformation(trg_data_transf_fun)
            trg_clustering.apply(mix=mix, alpha=arguments.nmf_alpha, l1=arguments.nmf_l1,
                                 max_iter=arguments.nmf_max_iter, rel_err=arguments.nmf_rel_err)

        if arguments.method is 'NMF' and src_data is None:
            print('Single task clustering method is NMF.')
            trg_clustering = NmfClustering(trg_data, trg_gene_ids, num_cluster=trg_k)
            trg_clustering.add_cell_filter(trg_cell_filter_fun)
            trg_clustering.add_gene_filter(trg_gene_filter_fun)
            trg_clustering.set_data_transformation(trg_data_transf_fun)
            trg_clustering.apply(alpha=arguments.nmf_alpha, l1=arguments.nmf_l1,
                                 max_iter=arguments.nmf_max_iter, rel_err=arguments.nmf_rel_err)

        if arguments.method is 'SC3':
            print('Clustering method is SC3.')
            num_cells = trg_data.shape[1]
            max_pca_comp = np.ceil(num_cells*0.07).astype(np.int)
            min_pca_comp = np.floor(num_cells*0.04).astype(np.int)
            print('(Max/Min) PCA components: ({0}/{1})'.format(max_pca_comp, min_pca_comp))
            trg_clustering = SC3Clustering(trg_data, trg_gene_ids,
                                           pc_range=[min_pca_comp, max_pca_comp], sub_sample=True, consensus_mode=0)

            dist_list = arguments.sc3_dists.split(",")
            print('\nThere are {0} distances given.'.format(len(dist_list)))
            for ds in dist_list:
                if src_data is not None:
                    print('- Adding transfer learning distance {0}'.format(ds))
                    trg_clustering.add_distance_calculation(partial(sc.da_nmf_distances, metric=ds,
                                                                    src=src_clustering, mixture=mix))
                else:
                    print('- Adding distance {0}'.format(ds))
                    trg_clustering.add_distance_calculation(partial(sc.distances, metric=ds))

            transf_list = arguments.sc3_transf.split(",")
            print('\nThere are {0} transformations given.'.format(len(transf_list)))
            for ts in transf_list:
                print('- Adding transformation {0}'.format(ts))
                trg_clustering.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method=ts))

            trg_clustering.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=trg_k))
            trg_clustering.set_build_consensus_matrix(sc.build_consensus_matrix)
            trg_clustering.set_consensus_clustering(partial(sc.consensus_clustering, n_components=trg_k))
            trg_clustering.apply()

        # --------------------------------------------------
        # 4. EVALUATE CLUSTER ASSIGNMENT
        # --------------------------------------------------
        print('\nUnsupervised evaluation:')
        accs[0, j, i] = metrics.calinski_harabaz_score(
            trg_clustering.pp_data.T, trg_clustering.cluster_labels)
        accs[1, j, i] = metrics.silhouette_score(
            trg_clustering.pp_data.T, trg_clustering.cluster_labels, metric='euclidean')
        accs[2, j, i] = metrics.silhouette_score(
            trg_clustering.pp_data.T, trg_clustering.cluster_labels, metric='correlation')
        accs[3, j, i] = metrics.silhouette_score(
            trg_clustering.pp_data.T, trg_clustering.cluster_labels, metric='jaccard')
        print '  -Calinski-Harabaz : ', accs[0, j, i]
        print '  -Silhouette (euc) : ', accs[1, j, i]
        print '  -Silhouette (corr): ', accs[2, j, i]
        print '  -Silhouette (jacc): ', accs[3, j, i]
        if trg_labels is not None:
            print('\nSupervised evaluation:')
            accs[4, j, i] = metrics.adjusted_rand_score(
                trg_labels[trg_clustering.remain_cell_inds], trg_clustering.cluster_labels)
            print '  -ARI: ', accs[4, j, i]

        # --------------------------------------------------
        # 5. SAVE RESULTS
        # --------------------------------------------------
        trg_clustering.cell_filter_list = None
        trg_clustering.gene_filter_list = None
        trg_clustering.data_transf = None
        if arguments.method is 'SC3':
            trg_clustering.dists_list = None
            trg_clustering.dimred_list = None
            trg_clustering.intermediate_clustering_list = None
        if src_clustering is not None:
            src_clustering.cell_filter_list = None
            src_clustering.gene_filter_list = None
            src_clustering.data_transf = None
        print('\nSaving data structures and results to file with prefix \'{0}_m{1}_c{2}\'.'.format(arguments.fout, mix, trg_k))
        np.savez('{0}_m{1}_c{2}.npz'.format(arguments.fout, mix, trg_k), src=src_clustering, trg=trg_clustering, args=arguments)
        np.savetxt('{0}_m{1}_c{2}.labels.tsv'.format(arguments.fout, mix, trg_k),
                   (trg_clustering.cluster_labels, trg_clustering.remain_cell_inds), fmt='%u', delimiter='\t')

# --------------------------------------------------
# 6. SUMMARIZE RESULTS
# --------------------------------------------------
print 'Mixtures:', mixtures
print 'Cluster:', num_cluster

plt.figure(0)
for i in range(accs.shape[0]):
    plt.subplot(2, 3, i+1)
    print('\n{0} (mixtures x cluster):'.format(accs_names[i]))
    print accs[i, :, :].reshape(len(mixtures), len(num_cluster))

    plt.title(accs_names[i])
    plt.pcolor(accs[i, :, :], cmap=plt.get_cmap('Reds'))
    plt.xlabel('Cluster')
    plt.ylabel('Mixture')
    plt.xticks(np.array(range(len(num_cluster)), dtype=np.float)+0.5, num_cluster)
    plt.yticks(np.array(range(len(mixtures)), dtype=np.float)+0.5, mixtures)
    plt.colorbar()
    if i>0:
        plt.clim(0.,+1.)
    if i==accs.shape[0]-1:
        plt.clim(0.,+1.)

plt.savefig('{0}.{1}.png'.format(arguments.fout, 'accs'), format='png',
            bbox_inches=None, pad_inches=0.1)
plt.show()

print('\nDone.')
