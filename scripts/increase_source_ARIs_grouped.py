import logging
from functools import partial
from nmf_clustering import NmfClustering
from utils import *
import datetime
from simulation import split_source_target
import pandas as pd

logging.basicConfig()
now1 = datetime.datetime.now()
print "Current date and time:"
print now1.strftime("%Y-%m-%d %H:%M")

# Data location
fname_data = 'C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\matrix'
fname_results = 'increase_src_ari_experiment_grouped_3.npz'
fname_labels = 'C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\cell_labels_primary_grouped_3'

# Parameters
reps = 10   # 10, number of repetitions, 100
n_src = [1000]  # [20,100, 300, 600, 1000, 1300, 1600, 1670] number of source data points, 1000
min_expr_genes = 2000
non_zero_threshold = 1
perc_consensus_genes = 0.98
num_cluster = 3 # 18
nmf_alpha = 1.0
nmf_l1 = 0.75
nmf_max_iter = 4000
nmf_rel_err = 1e-3
preprocessing_first = True

data = pd.read_csv(fname_data, sep='\t').values
print "data dimensions before preprocessing: genes x cells: ", data.shape

# Loading labels
labels = np.loadtxt(fname_labels,  delimiter='\t')
label_names, label_counts = np.unique(labels, return_counts = True)
print "Labels NMF: ", label_names
print "Counts NMF: ", label_counts

# Cell and gene filter and transformation before the whole procedure
cell_inds = sc.cell_filter(data, num_expr_genes=min_expr_genes, non_zero_threshold=non_zero_threshold)
data = data[:,cell_inds]
labels = labels[cell_inds]

# labels = labels[cell_inds]
gene_inds = sc.gene_filter(data, perc_consensus_genes=perc_consensus_genes, non_zero_threshold=non_zero_threshold)
data = data[gene_inds, :]
data = sc.data_transformation_log2(data)
cell_filter_fun = partial(sc.cell_filter, num_expr_genes=0, non_zero_threshold=-1)
gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=1, non_zero_threshold=-1)
data_transf_fun = sc.no_data_transformation
print "data dimensions after preprocessing: genes x cells: ", data.shape


genes = data.shape[0]  # number of genes
n_all = data.shape[1]
n_trg = n_all - np.max(n_src)    # overall number of target data points
if data.shape[1] < np.max(n_src) | num_cluster > np.min(n_src) | n_trg < num_cluster:
    print "Not enough cells left!"

# Create results matrix
source_aris = np.zeros((len(n_src), reps))

# create empty job vector
jobs = []
params = []
exp_counter = 1

# Run jobs
for s in range(len(n_src)):
#for s in [5,6]:
    r = 0
    while r < reps:
        # Split data in source and target randomly (mode =1) or randomly stratified (mode = 2)
        if n_src[s]==data.shape[1]:
            src = data
            src_labels = labels
        else:
            src, trg, src_labels, trg_labels = split_source_target(data, labels, mode=2,
                                                               target_ncells=n_trg,
                                                               source_ncells=n_src[s])

        src_labels = np.array(src_labels, dtype=np.int)
        #src_labels_SC3 = np.array(src_labels_SC3, dtype=np.int)

        # 3.c. train source once per repetition
        print "Train source data of rep {0}".format(r+1)
        source_nmf = None
        source_nmf = NmfClustering(src, np.arange(src.shape[0]), num_cluster=num_cluster)
        source_nmf.add_cell_filter(cell_filter_fun)
        source_nmf.add_gene_filter(gene_filter_fun)
        source_nmf.set_data_transformation(data_transf_fun)
        source_nmf.apply(k=num_cluster, alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)

        # Calculate ARIs and KTAs
        source_aris[s,r] = metrics.adjusted_rand_score(src_labels[source_nmf.remain_cell_inds], source_nmf.cluster_labels)


        print 'SOURCE ARI Labels NMF, Method NMF = ', source_aris[s, r]


        r += 1

print 'Mean ARIs', np.mean(source_aris)
np.savez(fname_results, source_aris=source_aris, min_expr_genes=min_expr_genes,
         non_zero_threshold=non_zero_threshold, perc_consensus_genes=perc_consensus_genes, num_cluster=num_cluster,
         nmf_alpha=nmf_alpha, nmf_l1=nmf_l1, nmf_max_iter=nmf_max_iter, nmf_rel_err=nmf_rel_err,
         reps=reps, genes=genes, n_src=n_src, n_trg=n_trg)

now2 = datetime.datetime.now()
print "Current date and time:"
print now2.strftime("%Y-%m-%d %H:%M")
print "Time passed:"
print now2-now1

print('Done.')

