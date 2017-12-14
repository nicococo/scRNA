import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import argparse, sys

from functools import partial
from sklearn.manifold import TSNE

from sc3_clustering import SC3Clustering
from nmf_clustering import DaNmfClustering, NmfClustering
from utils import *


# --------------------------------------------------
# PARSE COMMAND LINE ARGUMENTS
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src-fname", help="Source *.npz result filename", dest='src_fname', required=False, type=str, default=None)
parser.add_argument("--fname", help="Target data (TSV file)", required=True, type=str, default=None)
parser.add_argument("--fgene-ids", help="Target gene ids (TSV file)", dest='fgene_ids', required=True, type=str, default=None)
parser.add_argument("--fout", help="Result files will use this prefix.", default='trg', type=str)
parser.add_argument("--flabels", help="[optional] Target cluster labels (TSV file)", required=False, type=str, default=None)

parser.add_argument("--min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type=int)
parser.add_argument("--non_zero_threshold", help="(Cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type=float)
parser.add_argument("--perc_consensus_genes", help="(Gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type=float)

parser.add_argument("--cluster-range", help="Comma separated list of clusters (default 6,7,8)", dest='cluster_range', default='6,7,8', type=str)
parser.add_argument("--mixtures", help="Comma separated list of convex combination src-trg mixture coefficient (0.=no transfer, default 0.0,0.1,0.2)", default="0.0,0.1,0.2", type = str)

parser.add_argument("--sc3-dists", dest='sc3_dists', help="(SC3) Comma-separated MTL distances (default euclidean)", default='euclidean', type = str)
parser.add_argument("--sc3-transf", dest='sc3_transf', help="(SC3) Comma-separated transformations (default pca)", default='pca', type = str)

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
parser.set_defaults(use_cell_filter=True)

parser.add_argument("--gene-filter",
    help="Enable gene filter for source and target datasets.",
    dest="use_gene_filter",
    action='store_true')
parser.add_argument("--no-gene-filter",
    help="Disable gene filter for source and target datasets.",
    dest="use_gene_filter",
    action='store_false')
parser.set_defaults(use_gene_filter=True)

parser.add_argument("--transform",
    help="Transform data to log2(x+1)",
    dest="transform",
    action='store_true')
parser.add_argument("--no-transform",
    help="Disable transform data to log2(x+1)",
    dest="transform",
    action='store_false')
parser.set_defaults(transform=True)

parser.add_argument("--tsne",
    help="Enable t-SNE plots.",
    dest="tsne",
    action='store_true')
parser.add_argument("--no-tsne",
    help="Disable t-SNE plots.",
    dest="tsne",
    action='store_false')
parser.set_defaults(tsne=True)

parser.add_argument("--transferability",
    help="Enable transferability score calculation.",
    dest="calc_transf",
    action='store_true')
parser.add_argument("--no-transferability",
    help="Disable transferability score calculation.",
    dest="calc_transf",
    action='store_false')
parser.set_defaults(calc_transf=True)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print("\nLoading target dataset (data={0} and gene_ids={1}).".format(arguments.fname, arguments.fgene_ids))
data, gene_ids, labels, labels_2_ids = load_dataset_tsv(arguments.fname, arguments.fgene_ids, flabels=arguments.flabels)

# inds = np.random.permutation(data.shape[1])[:80]
# data = data[:, inds]
# if labels is not None:
#     labels = labels[inds]
# inds = np.random.permutation(data.shape[1])
# data = data[:, inds]

# inds = np.random.permutation(data.shape[1])[:100]
# data = data[:, inds]
# if labels is not None:
#     labels = labels[inds]

print np.histogram(labels, bins=np.unique(labels).size)

print('Target data {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1]))
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
mixtures = map(np.float, arguments.mixtures.split(","))

accs_names = ['KTA', 'KTA (WH2)', 'KTA (WH)', 'Sil (euc)', 'Sil (pearson)', 'Sil (spearman)', 'MixARI', 'ARI']

accs_dist = np.zeros((len(accs_names), len(mixtures), len(num_cluster)))
accs_mix = np.zeros((len(accs_names), len(mixtures), len(num_cluster)))

transferability = 0.0
transferability_percs = None

for i in range(len(num_cluster)):
    for j in range(len(mixtures)):
        k = num_cluster[i]
        mix = mixtures[j]
        print('Iteration k={0} mix={1}'.format(k, mix))

        # --------------------------------------------------
        # 3.1. MIX TARGET & SOURCE DATE
        # --------------------------------------------------
        src_data = np.load(arguments.src_fname)  # src data gets while applying da_nmf ...
        src_nmf = src_data['src'][()]
        print type(src_nmf)
        src_nmf.cell_filter_list = list()
        src_nmf.gene_filter_list = list()
        # source data is already filtered and transformed ...
        src_nmf.add_cell_filter(lambda x: np.arange(x.shape[1]).tolist())
        src_nmf.add_gene_filter(lambda x: np.arange(x.shape[0]).tolist())
        src_nmf.set_data_transformation(lambda x: x)

        print '\n\n+++++++++++++++++++++++++++++++++++'
        print np.max(data)
        print '+++++++++++++++++++++++++++++++++++\n\n'

        da_nmf = DaNmfClustering(src_nmf, data.copy(), gene_ids.copy(), k)
        da_nmf.add_cell_filter(cell_filter_fun)
        da_nmf.add_gene_filter(gene_filter_fun)
        da_nmf.set_data_transformation(data_transf_fun)
        calc_transf = False
        if i == 0 and j == 0:
            calc_transf = True
        if not arguments.calc_transf:
            calc_transf = False
        mix_data, mix_new_trg_data, mix_trg_data = \
            da_nmf.get_mixed_data(mix=mix, reject_ratio=0., calc_transferability=calc_transf, max_iter=2000)
        # mix_gene_ids = da_nmf.common_ids
        mix_gene_ids = da_nmf.gene_ids
        if calc_transf:
            transferability = da_nmf.transferability_score
            transferability_percs = da_nmf.transferability_percs

        print '\n\n+++++++++++++++++++++++++++++++++++------'
        print np.max(data)
        print '+++++++++++++++++++++++++++++++++++------\n\n'

        # --------------------------------------------------
        # 3.2. TARGET DATA CLUSTERING
        # --------------------------------------------------
        print('Clustering method is SC3.')
        num_cells = data.shape[1]
        max_pca_comp = np.ceil(num_cells*0.07).astype(np.int)
        min_pca_comp = np.floor(num_cells*0.04).astype(np.int)
        print('(Max/Min) PCA components: ({0}/{1})'.format(max_pca_comp, min_pca_comp))
        sc3_dist = SC3Clustering(data, gene_ids,
                                pc_range=[min_pca_comp, max_pca_comp], sub_sample=True, consensus_mode=0)
        sc3_mix = SC3Clustering(mix_data, mix_gene_ids,
                                pc_range=[min_pca_comp, max_pca_comp], sub_sample=True, consensus_mode=0)

        sc3_dist.add_cell_filter(cell_filter_fun)
        sc3_dist.add_gene_filter(gene_filter_fun)
        sc3_dist.set_data_transformation(data_transf_fun)

        # INFO: mix_data is already filtered and transformed.
        # The lines below are for documentation. Do not uncomment these.
        # sc3_mix.add_cell_filter(cell_filter_fun)
        # sc3_mix.add_gene_filter(gene_filter_fun)
        # sc3_mix.set_data_transformation(data_transf_fun)

        dist_list = arguments.sc3_dists.split(",")
        print('\nThere are {0} distances given.'.format(len(dist_list)))
        for ds in dist_list:
            print('- Adding transfer learning distance {0}'.format(ds))
            sc3_dist.add_distance_calculation(partial(sc.da_nmf_distances,
                                                            da_model=da_nmf.intermediate_model,
                                                            reject_ratio=0.,
                                                            metric=ds,
                                                            mixture=mix))
            print('- Adding distance {0}'.format(ds))
            sc3_mix.add_distance_calculation(partial(sc.distances, metric=ds))

        transf_list = arguments.sc3_transf.split(",")
        print('\nThere are {0} transformations given.'.format(len(transf_list)))
        for ts in transf_list:
            print('- Adding transformation {0}'.format(ts))
            sc3_dist.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method=ts))
            sc3_mix.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method=ts))

        sc3_dist.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=k))
        sc3_dist.set_build_consensus_matrix(sc.build_consensus_matrix)
        sc3_dist.set_consensus_clustering(partial(sc.consensus_clustering, n_components=k))
        sc3_dist.apply()

        sc3_mix.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=k))
        sc3_mix.set_build_consensus_matrix(sc.build_consensus_matrix)
        sc3_mix.set_consensus_clustering(partial(sc.consensus_clustering, n_components=k))
        sc3_mix.apply()

        # --------------------------------------------------
        # 3.3. EVALUATE CLUSTER ASSIGNMENT
        # --------------------------------------------------
        W, H, H2 = da_nmf.intermediate_model

        print('\nSC3 dist evaluation:')
        accs_dist[0, j, i] = unsupervised_acc_kta(sc3_dist.pp_data, sc3_dist.cluster_labels, kernel='linear')
        accs_dist[1, j, i] = unsupervised_acc_kta(W.dot(H2), sc3_dist.cluster_labels, kernel='linear')
        accs_dist[2, j, i] = unsupervised_acc_kta(W.dot(H), sc3_dist.cluster_labels, kernel='linear')
        accs_dist[3, j, i] = unsupervised_acc_silhouette(sc3_dist.pp_data, sc3_dist.cluster_labels, metric='euclidean')
        accs_dist[4, j, i] = unsupervised_acc_silhouette(sc3_dist.pp_data, sc3_dist.cluster_labels, metric='pearson')
        accs_dist[5, j, i] = unsupervised_acc_silhouette(sc3_dist.pp_data, sc3_dist.cluster_labels, metric='spearman')
        # accs_dist[6, j, i] is not used
        if labels is not None:
            accs_dist[7, j, i] = metrics.adjusted_rand_score(labels[sc3_dist.remain_cell_inds], sc3_dist.cluster_labels)
        print accs_dist[:, j, i]
        print('\nSC3 mix evaluation:')

        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LinearRegression
        include_inds = []
        pred_lbls = np.unique(sc3_mix.cluster_labels)
        for p in pred_lbls:
            inds = np.where(sc3_mix.cluster_labels == p)[0]
            if inds.size >= 2:
                include_inds.extend(inds)

        accs_mix[0, j, i] = unsupervised_acc_kta(mix_trg_data.copy(), sc3_mix.cluster_labels.copy(), kernel='linear')
        accs_mix[1, j, i] = unsupervised_acc_kta(W.dot(H2), sc3_mix.cluster_labels.copy(), kernel='linear')
        accs_mix[2, j, i] = unsupervised_acc_kta(W.dot(H), sc3_mix.cluster_labels.copy(), kernel='linear')
        accs_mix[3, j, i] = unsupervised_acc_silhouette(mix_trg_data.copy(), sc3_mix.cluster_labels.copy(), metric='euclidean')
        accs_mix[4, j, i] = unsupervised_acc_silhouette(mix_trg_data.copy(), sc3_mix.cluster_labels.copy(), metric='pearson')
        accs_mix[5, j, i] = unsupervised_acc_silhouette(mix_trg_data.copy(), sc3_mix.cluster_labels.copy(), metric='spearman')

        if len(include_inds) > 5:
            cls = OneVsRestClassifier(LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)).fit(sc3_mix.pp_data[:, include_inds].T.copy(), sc3_mix.cluster_labels[include_inds].copy())
            ret = cls.predict(mix_trg_data[:, include_inds].T.copy())
            accs_mix[6, j, i] = metrics.adjusted_rand_score(ret, sc3_mix.cluster_labels[include_inds].copy())

        if labels is not None:
            accs_mix[7, j, i] = metrics.adjusted_rand_score(labels[da_nmf.remain_cell_inds], sc3_mix.cluster_labels)
        print accs_mix[:, j, i]

        # --------------------------------------------------
        # 3.4. SAVE RESULTS
        # --------------------------------------------------
        print('\nSaving data structures and results to file with prefix \'{0}_m{1}_c{2}\'.'.format(arguments.fout, mix, k))
        np.savetxt('{0}_m{1}_c{2}.labels.sc3_dist.tsv'.format(arguments.fout, mix, k),
                   (sc3_dist.cluster_labels, sc3_dist.remain_cell_inds), fmt='%u', delimiter='\t')
        np.savetxt('{0}_m{1}_c{2}.labels.sc3_mix.tsv'.format(arguments.fout, mix, k),
                   (sc3_mix.cluster_labels, sc3_mix.remain_cell_inds), fmt='%u', delimiter='\t')
        np.savetxt('{0}_m{1}_c{2}.data.tsv'.format(arguments.fout, mix, k),
                   (sc3_mix.pp_data), fmt='%u', delimiter='\t')
        np.savetxt('{0}_m{1}_c{2}.geneids.tsv'.format(arguments.fout, mix, k),
                   (mix_gene_ids), fmt='%s', delimiter='\t')

        # --------------------------------------------------
        # 3.5. T-SNE PLOT
        # --------------------------------------------------
        if arguments.tsne:
            print 'Data:'
            print sc3_dist.data[sc3_dist.remain_gene_inds[0], :10]
            print sc3_mix.data[sc3_mix.remain_gene_inds[0], :10]

            plt.clf()
            plt.subplot(1, 3, 1)
            # plt.title('SC3-Dist(l), SC3-mix(r) cluster={0}, mix={1}, (match={2:0.2f})'.format(k, mix, match), fontsize=10)
            plt.title('SC3-Dist')
            model = TSNE(n_components=2, random_state=0, init='pca', method='exact', metric='euclidean', perplexity=30)
            ret = model.fit_transform(sc3_dist.pp_data.T)
            plt.scatter(ret[:, 0], ret[:, 1], 20, sc3_dist.cluster_labels)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 3, 2)
            plt.title('SC3-Mix (mixed target)', fontsize=10)
            model = TSNE(n_components=2, random_state=0, init='pca', method='exact', metric='euclidean', perplexity=30)
            ret = model.fit_transform(sc3_mix.pp_data.T)
            plt.scatter(ret[:, 0], ret[:, 1], 20, sc3_mix.cluster_labels)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 3, 3)
            plt.title('SC3-Mix (target)', fontsize=10)
            model = TSNE(n_components=2, random_state=0, init='pca', method='exact', metric='euclidean', perplexity=30)
            ret = model.fit_transform(mix_trg_data.T)
            plt.scatter(ret[:, 0], ret[:, 1], 20, sc3_mix.cluster_labels)
            plt.xticks([])
            plt.yticks([])

            plt.savefig('{0}_m{1}_c{2}_tsne.png'.format(arguments.fout, mix, k), format='png', bbox_inches=None, pad_inches=0.1)
            # plt.show()


# --------------------------------------------------
# 6. SUMMARIZE RESULTS
# --------------------------------------------------
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
print ' - Source file name: ', arguments.src_fname
print ' - Mixtures:', mixtures
print ' - Cluster:', num_cluster
if labels is not None:
    print ' - Class 2 label conversion (class {0:1d}-{1:1d}): '.format(
        np.int(np.min(labels)), np.int(np.max(labels))), labels_2_ids
print ''

print 'Results'
print '-------------'
print ' - Estimated transferability t (0 <= t <= 1):', transferability
print ' - Estimated non-transferability percentiles:', transferability_percs
print ' - Accuracies: ', accs_names
for i in range(accs_mix.shape[0]):
    print('\n{0} (mixtures({1}) x cluster({2})) [sc3-dist(left), sc3-mix(right)]:'.format(
        accs_names[i], len(mixtures), len(num_cluster)))
    for m in range(accs_mix.shape[1]):
        print accs_dist[i, m, :], '   ', accs_mix[i, m, :]


plt.figure(0)
n = accs_mix.shape[0]
fig, axes = plt.subplots(nrows=4, ncols=n / 2)
fig.suptitle("Mixture (y-axis) vs Clusters (x-axis)", fontsize=6)
fig.tight_layout(h_pad=2.08, pad=2.2) # Or equivalently,  "plt.tight_layout()"
for i in range(n):
    plt.subplot(4, n/2, i+1)
    plt.title(accs_names[i], fontsize=8)
    plt.pcolor(accs_dist[i, :, :], cmap=plt.get_cmap('Reds'))
    # plt.xlabel('Cluster', fontsize=12)
    if i == 0:
        plt.ylabel('SC3-dist results', fontsize=10)
    plt.xticks(np.array(range(len(num_cluster)), dtype=np.float)+0.5, num_cluster, fontsize=8)
    plt.yticks(np.array(range(len(mixtures)), dtype=np.float)+0.5, mixtures, fontsize=8)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=5)
    # plt.clim(0.,+1.)
    # if i == accs_mix.shape[0]-1:
    # plt.clim(0.,+1.)

    plt.subplot(4, n/2, i+1+n)
    plt.title(accs_names[i], fontsize=8)
    plt.pcolor(accs_mix[i, :, :], cmap=plt.get_cmap('Blues'))
    # plt.xlabel('Cluster', fontsize=12)
    if i == 0:
        plt.ylabel('SC3-mix results', fontsize=10)
    plt.xticks(np.array(range(len(num_cluster)), dtype=np.float)+0.5, num_cluster, fontsize=8)
    plt.yticks(np.array(range(len(mixtures)), dtype=np.float)+0.5, mixtures, fontsize=8)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=5)
    # plt.clim(0.,+1.)
    # if i == accs_mix.shape[0]-1:
    # plt.clim(0.,+1.)

plt.savefig('{0}_{1}.png'.format(arguments.fout, 'accs'), format='png',
            bbox_inches=None, pad_inches=0.1, dpi=1000)
# plt.show()


print('\nDone.')
