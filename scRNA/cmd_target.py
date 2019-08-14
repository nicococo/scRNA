import matplotlib as mpl
mpl.use('Agg')

import argparse, sys

from functools import partial
from sklearn.manifold import TSNE

from scRNA.sc3_clustering import SC3Clustering
from scRNA.nmf_clustering import DaNmfClustering
from scRNA.utils import *


# --------------------------------------------------
# PARSE COMMAND LINE ARGUMENTS
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src-fname", help="Source *.npz result filename", dest='src_fname', type=str, default='src_c8.npz')
parser.add_argument("--fname", help="Source data (TSV file)", type=str, default='fout_source_data_T1_100_S1_1000.tsv')
# parser.add_argument("--fname", help="Source data (TSV file)", type=str, default='C:\Users\Bettina\PycharmProjects2\scRNA_new\data\Usoskin\usoskin_m_fltd_mat.tsv')
parser.add_argument("--fname-trg", help="Target data (TSV file)", type=str, default='fout_target_data_T1_100_S1_1000.tsv')
# parser.add_argument("--fname-trg", help="Target data (TSV file)", type=str, default='C:\Users\Bettina\PycharmProjects2\scRNA_new\data\Jim\Visceraltpm_m_fltd_mat.tsv')
parser.add_argument("--fgene-ids", help="Source gene ids (TSV file)", dest='fgene_ids', type=str, default='fout_geneids.tsv')
# parser.add_argument("--fgene-ids", help="Source gene ids (TSV file)", dest='fgene_ids', type=str, default='C:\Users\Bettina\PycharmProjects2\scRNA_new\data\Usoskin\usoskin_m_fltd_row.tsv')
parser.add_argument("--fgene-ids-trg", help="Target data gene ids (TSV file)", dest='fgene_ids_trg', type=str, default='fout_geneids.tsv')
# parser.add_argument("--fgene-ids-trg", help="Target data gene ids (TSV file)", dest='fgene_ids_trg', type=str, default='C:\Users\Bettina\PycharmProjects2\scRNA_new\data\Jim\Visceraltpm_m_fltd_row.tsv')
parser.add_argument("--fout", help="Result files will use this prefix.", default='trg', type=str)
parser.add_argument("--flabels-trg", help="[optional] True target cluster labels (TSV file)", dest='flabels_trg', required=False, type=str, default='fout_target_labels_T1_100_S1_1000.tsv')
# parser.add_argument("--flabels-trg", help="[optional] True target cluster labels (TSV file)", dest='flabels_trg', required=False, type=str)

parser.add_argument("--min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type=int)
parser.add_argument("--non_zero_threshold", help="(Cell/gene filter) Threshold for zero expression per gene (default 2.0)", default=2.0, type=float)
parser.add_argument("--perc_consensus_genes", help="(Gene filter) Filter genes that coincide across a percentage of cells (default 0.94)", default=0.94, type=float)

parser.add_argument("--cluster-range", help="Comma separated list of clusters (default 8)", dest='cluster_range', default='8', type=str)
parser.add_argument("--mixtures", help="Comma separated list of convex combination src-trg mixture coefficient (0.=no transfer, default 0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0)", default="0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0", type = str)

parser.add_argument("--sc3-dists", dest='sc3_dists', help="(SC3) Comma-separated MTL distances (default euclidean)", default='euclidean', type = str)
parser.add_argument("--sc3-transf", dest='sc3_transf', help="(SC3) Comma-separated transformations (default pca)", default='pca', type = str)

parser.add_argument("--nmf_alpha", help="(NMF) Regularization strength (default 10.0)", default=10.0, type = float)
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
parser.set_defaults(use_cell_filter=False)

parser.add_argument("--gene-filter",
    help="Enable gene filter for source and target datasets.",
    dest="use_gene_filter",
    action='store_true')
parser.add_argument("--no-gene-filter",
    help="Disable gene filter for source and target datasets.",
    dest="use_gene_filter",
    action='store_false')
parser.set_defaults(use_gene_filter=False)

parser.add_argument("--transform",
    help="Transform data to log2(x+1)",
    dest="transform",
    action='store_true')
parser.add_argument("--no-transform",
    help="Disable transform data to log2(x+1)",
    dest="transform",
    action='store_false')
parser.set_defaults(transform=False)

parser.add_argument("--tsne",
    help="Enable t-SNE plots.",
    dest="tsne",
    action='store_true')
parser.add_argument("--no-tsne",
    help="Disable t-SNE plots.",
    dest="tsne",
    action='store_false')
parser.set_defaults(tsne=True)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print(arguments)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print(("\nLoading target dataset (data={0} and gene_ids={1}).".format(arguments.fname_trg, arguments.fgene_ids_trg)))
data, gene_ids, labels, labels_2_ids = load_dataset_tsv(arguments.fname_trg, arguments.fgene_ids_trg, flabels=arguments.flabels_trg)

# labels can be empty
if labels is not None:
    #print np.histogram(labels, bins=np.unique(labels).size)
    print(('Target data {1} cells and {0} genes/transcripts.'.format(data.shape[0], data.shape[1])))
    print('Cell types of real labels:')
    print((np.unique(labels)))
    np.savetxt('{0}.labels2ids.tsv'.format(arguments.fout), (np.arange(labels_2_ids.size), labels_2_ids), fmt='%s', delimiter='\t')

#print('Number of genes/transcripts in data and gene-ids must coincide.')
assert(data.shape[0] == gene_ids.shape[0]), 'Number of genes/transcripts in data and gene-ids must coincide.'

# --------------------------------------------------
# 2. CELL and GENE FILTER
# --------------------------------------------------
# Preprocessing Target Data
print(("Target data dimensions before preprocessing: genes x cells", data.shape))
# Cell and gene filter and transformation before the whole procedure
if arguments.use_cell_filter:
    cell_inds = sc.cell_filter(data, num_expr_genes=arguments.min_expr_genes, non_zero_threshold=arguments.non_zero_threshold)
    data = data[:, cell_inds]
    if labels is not None:
        labels = labels[cell_inds]
        labels_2_ids = labels_2_ids[cell_inds]
else:
    cell_inds = list(range(data.shape[1]))
if arguments.use_gene_filter:
    gene_inds = sc.gene_filter(data, perc_consensus_genes=arguments.perc_consensus_genes, non_zero_threshold=arguments.non_zero_threshold)
    data = data[gene_inds, :]
    gene_ids = gene_ids[gene_inds]
if arguments.transform:
    data = sc.data_transformation_log2(data)
cell_filter_fun = partial(sc.cell_filter, num_expr_genes=0, non_zero_threshold=-1)
gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=1, non_zero_threshold=-1)
data_transf_fun = sc.no_data_transformation
print(("Target data dimensions after preprocessing: genes x cells: ", data.shape))

# --------------------------------------------------
# Gene subset between source and target
# --------------------------------------------------
data_src, gene_ids_src, _, _ = load_dataset_tsv(arguments.fname, arguments.fgene_ids)

# Preprocessing Source Data
print(("Source data dimensions before preprocessing: genes x cells", data_src.shape))
# Cell and gene filter and transformation before the whole procedure
if arguments.use_cell_filter:
    cell_inds_src = sc.cell_filter(data_src, num_expr_genes=arguments.min_expr_genes, non_zero_threshold=arguments.non_zero_threshold)
    data_src = data_src[:, cell_inds_src]
if arguments.use_gene_filter:
    gene_inds = sc.gene_filter(data_src, perc_consensus_genes=arguments.perc_consensus_genes, non_zero_threshold=arguments.non_zero_threshold)
    data_src = data_src[gene_inds, :]
    gene_ids_src = gene_ids_src[gene_inds]

print(("Source data dimensions after preprocessing: genes x cells: ", data_src.shape))

# print('Genes/transcripts in source and target data must coincide.')
# Find gene subset
gene_intersection = list(set(x for x in gene_ids).intersection(set(x for x in gene_ids_src)))
assert (gene_intersection is not None), 'Genes/transcripts in source and target data must coincide.'

# Adjust target data to only include overlapping genes
data_target_indices = list(list(gene_ids).index(x) for x in gene_intersection)
data = data[data_target_indices,]

print(("Target data dimensions after taking target intersection: genes x cells: ", data.shape))

# --------------------------------------------------
# 3. CLUSTERING
# --------------------------------------------------
num_cluster = list(map(np.int, arguments.cluster_range.split(",")))
mixtures = list(map(np.float, arguments.mixtures.split(",")))

accs_names = ['KTA', 'ARI']
accs = np.zeros((len(accs_names), len(num_cluster),len(mixtures)))

opt_mix_ind = np.zeros(len(num_cluster))
opt_mix_aris = np.zeros(len(num_cluster))
opt_mix_ktas = np.zeros(len(num_cluster))

trg_labels = np.zeros((data.shape[1],len(num_cluster), len(mixtures)))

for i in range(len(num_cluster)):
    for j in range(len(mixtures)):
        k = num_cluster[i]
        mix = mixtures[j]
        print(('\n Iteration k={0} mix={1}'.format(k, mix)))

        # --------------------------------------------------
        # 3.1. MIX TARGET & SOURCE DATE
        # --------------------------------------------------
        src_data = np.load(arguments.src_fname)  # src data gets while applying da_nmf ...
        src_nmf = src_data['src'][()]
        # print type(src_nmf)
        src_nmf.cell_filter_list = list()
        src_nmf.gene_filter_list = list()
        # source data is already filtered and transformed ...
        src_nmf.add_cell_filter(lambda x: np.arange(x.shape[1]).tolist())
        src_nmf.add_gene_filter(lambda x: np.arange(x.shape[0]).tolist())
        src_nmf.set_data_transformation(lambda x: x)

        #print '\n\n+++++++++++++++++++++++++++++++++++'
        #print np.max(data)
        #print '+++++++++++++++++++++++++++++++++++\n\n'

        da_nmf = DaNmfClustering(src_nmf, data.copy(), gene_ids.copy(), k)
        da_nmf.add_cell_filter(cell_filter_fun)
        da_nmf.add_gene_filter(gene_filter_fun)
        da_nmf.set_data_transformation(data_transf_fun)
        mixed_data, _, _ = \
            da_nmf.get_mixed_data(mix=mix, calc_transferability=False)
        # mix_gene_ids = da_nmf.common_ids
        mix_gene_ids = da_nmf.gene_ids


        #print '\n\n+++++++++++++++++++++++++++++++++++------'
        #print np.max(data)
        #print '+++++++++++++++++++++++++++++++++++------\n\n'

        # --------------------------------------------------
        # 3.2. TARGET DATA CLUSTERING
        # --------------------------------------------------
        print('TARGET DATA CLUSTERING.')
        num_cells = mixed_data.shape[1]
        max_pca_comp = np.ceil(num_cells*0.07).astype(np.int)
        min_pca_comp = np.max([1, np.floor(num_cells * 0.04).astype(np.int)])

        #print('(Max/Min) PCA components: ({0}/{1})'.format(max_pca_comp, min_pca_comp))

        sc3_mix = SC3Clustering(mixed_data, pc_range=[min_pca_comp, max_pca_comp], sub_sample=True, consensus_mode=0)

        dist_list = arguments.sc3_dists.split(",")
        #print('\nThere are {0} distances given.'.format(len(dist_list)))
        for ds in dist_list:
        #    print('- Adding distance {0}'.format(ds))
            sc3_mix.add_distance_calculation(partial(sc.distances, metric=ds))

        transf_list = arguments.sc3_transf.split(",")
        #print('\nThere are {0} transformations given.'.format(len(transf_list)))
        for ts in transf_list:
        #    print('- Adding transformation {0}'.format(ts))
            sc3_mix.add_dimred_calculation(partial(sc.transformations, components=max_pca_comp, method=ts))

        sc3_mix.add_intermediate_clustering(partial(sc.intermediate_kmeans_clustering, k=k))
        sc3_mix.set_build_consensus_matrix(sc.build_consensus_matrix)
        sc3_mix.set_consensus_clustering(partial(sc.consensus_clustering, n_components=k))
        sc3_mix.apply()

        # Save predicted labels
        trg_labels[:, i,j] = sc3_mix.cluster_labels

        # --------------------------------------------------
        # 3.3. EVALUATE CLUSTER ASSIGNMENT
        # --------------------------------------------------
        print('Transfer Target clustering evaluation:'
              '[unsupervised KTA score, supervised ARI]')
        # mixed_data_now, _, _ = da_nmf.get_mixed_data(mix=mix, calc_transferability=False)
        accs[0, i, j] = unsupervised_acc_kta(mixed_data.copy(), sc3_mix.cluster_labels.copy(), kernel='linear')
        if labels is not None:
            accs[1, i, j] = metrics.adjusted_rand_score(labels[da_nmf.remain_cell_inds], sc3_mix.cluster_labels)
        print((accs[:, i, j]))

    # pdb.set_trace()
    opt_mix_ind[i] = np.argmax(accs[0, i, :])
    print('\n optimal mixture parameter (via KTA score selection):')
    print((mixtures[int(opt_mix_ind[i])]))

    opt_mix_ktas[i] = accs[0, i, int(opt_mix_ind[i])]
    print('\n KTA score of optimal mixture parameter (via KTA score selection):')
    print((opt_mix_ktas[i]))

    opt_mix_aris[i] = accs[1, i, int(opt_mix_ind[i])]
    print('\n ARI of optimal mixture parameter (via KTA score selection):')
    print((opt_mix_aris[i]))

    trg_labels_pred = trg_labels[:, i, int(opt_mix_ind[i])]

    # --------------------------------------------------
    # 3.4. SAVE RESULTS
    # --------------------------------------------------
    print(('\nSaving data structures and results to file with prefix \'{0}_c{1}\'.'.format(arguments.fout, k)))
    np.savetxt('{0}_c{1}.labels.transfercluster.tsv'.format(arguments.fout, k),(trg_labels_pred.astype(np.int64), cell_inds), fmt='%u', delimiter='\t')
    np.savetxt('{0}_c{1}.data.tsv'.format(arguments.fout, k), data, fmt='%u', delimiter='\t')
    np.savetxt('{0}_c{1}.geneids.tsv'.format(arguments.fout, k), gene_ids, fmt='%s', delimiter='\t')

    # --------------------------------------------------
    # 3.5. T-SNE PLOT with predicted labels
    # --------------------------------------------------
    if arguments.tsne:
        # print 'Data:'
        # print data[0, :10]

        plt.clf()

        plt.title('TransferCluster', fontsize=10)
        model = TSNE(n_components=2, random_state=0, init='pca', method='exact', metric='euclidean', perplexity=30)
        ret = model.fit_transform(data.T)
        plt.scatter(ret[:, 0], ret[:, 1], 20, trg_labels_pred)
        plt.xticks([])
        plt.yticks([])

        plt.savefig('{0}_c{1}.tsne.png'.format(arguments.fout, k), format='png', bbox_inches=None, pad_inches=0.1)
        # plt.show()

# --------------------------------------------------
# 3.5. T-SNE PLOT with real labels
# --------------------------------------------------
if arguments.tsne & (labels is not None):
    # print 'Data:'
    # print data[0, :10]

    plt.clf()

    plt.title('TransferCluster', fontsize=10)
    model = TSNE(n_components=2, random_state=0, init='pca', method='exact', metric='euclidean', perplexity=30)
    ret = model.fit_transform(data.T)
    plt.scatter(ret[:, 0], ret[:, 1], 20, labels)
    plt.xticks([])
    plt.yticks([])

    plt.savefig('{0}_c{1}.tsne_reallabels.png'.format(arguments.fout, k), format='png', bbox_inches=None, pad_inches=0.1)
    # plt.show()

# --------------------------------------------------
# 6. SUMMARIZE RESULTS
# --------------------------------------------------
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

print('\n\n\n')
print('================================================================================')
print('\n\n\n')
print('SUMMARY')
print('\n\n\n')
print('Parameters')
print('-------------')
print((' - Output prefix: ', arguments.fout))
print((' - Source file name: ', arguments.src_fname))
print((' - Mixtures:', mixtures))
print((' - Cluster:', num_cluster))
if labels is not None:
    print((' - Class 2 label conversion (class {0:1d}-{1:1d}): '.format(
        np.int(np.min(labels)), np.int(np.max(labels))), labels_2_ids))
print('')

print('Results')
print('-------------')
print((' - Accuracies: ', accs_names))
for i in range(accs.shape[0]):
    print(('\n{0} (mixtures({1}) x cluster({2})):'.format(
        accs_names[i], len(mixtures), len(num_cluster))))
    for m in range(accs.shape[1]):
        print((accs[i, m, :]))

for i in range(len(num_cluster)):
    print(('\n Optimal mixture parameters (via KTA score selection) for k = {0}'.format(num_cluster[i])))
    print((mixtures[int(opt_mix_ind[i])]))

    print(('\n KTA scores of optimal mixture parameter (via KTA score selection) for k = {0}'.format(num_cluster[i])))
    print((opt_mix_ktas[i]))

    print(('\n ARI of optimal mixture parameter (via KTA score selection)for k = {0}'.format(num_cluster[i])))
    print((opt_mix_aris[i]))


plt.figure(0)
n = accs.shape[0]
fig, axes = plt.subplots(nrows=4, ncols=np.int(n / 2))
fig.suptitle("Mixture (y-axis) vs Clusters (x-axis)", fontsize=6)
fig.tight_layout(h_pad=2.08, pad=2.2) # Or equivalently,  "plt.tight_layout()"
for i in range(n):
    plt.subplot(2, np.int(n/2), i+1)
    plt.title(accs_names[i], fontsize=8)
    plt.pcolor(accs[i, :, :].T, cmap=plt.get_cmap('Blues'))
    # plt.xlabel('Cluster', fontsize=12)
    if i == 0:
        plt.ylabel('SC3-mix results', fontsize=10)
    plt.xticks(np.array(list(range(len(num_cluster))), dtype=np.float)+0.5, num_cluster, fontsize=8)
    plt.yticks(np.array(list(range(len(mixtures))), dtype=np.float)+0.5, mixtures, fontsize=8)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=5)
    # plt.clim(0.,+1.)
    # if i == accs_mix.shape[0]-1:
    # plt.clim(0.,+1.)

plt.savefig('{0}.{1}.png'.format(arguments.fout, 'accs'), format='png',
            bbox_inches=None, pad_inches=0.1, dpi=1000)
# plt.show()


print('\nDone.')
