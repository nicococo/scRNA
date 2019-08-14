import logging
from functools import partial
from .experiments_utils import (method_sc3_filter, method_hub, method_sc3_combined_filter,
                               acc_ari, acc_kta, acc_transferability)
from nmf_clustering import NmfClustering
from utils import *
import datetime
from simulation import split_source_target
import pandas as pd
import sys


logging.basicConfig()
now1 = datetime.datetime.now()
print("Current date and time:")
print(now1.strftime("%Y-%m-%d %H:%M"))

# Data location
fname_data = 'C:\\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\matrix'

# Parameters
reps = 10   # number of repetitions, 100
n_src = [1668]  # number of source data points, 1000
min_expr_genes = 2000
non_zero_threshold = 1
perc_consensus_genes = 0.98
num_cluster = 16
nmf_alpha = 1.0
nmf_l1 = 0.75
nmf_max_iter = 4000
nmf_rel_err = 1e-3
preprocessing_first = True

data = pd.read_csv(fname_data, sep='\t').values
print("data dimensions before preprocessing: genes x cells: ", data.shape)

# Cell and gene filter and transformation before the whole procedure
cell_inds = sc.cell_filter(data, num_expr_genes=min_expr_genes, non_zero_threshold=non_zero_threshold)
data = data[:,cell_inds]
# labels = labels[cell_inds]
gene_inds = sc.gene_filter(data, perc_consensus_genes=perc_consensus_genes, non_zero_threshold=non_zero_threshold)
data = data[gene_inds, :]
data = sc.data_transformation_log2(data)
cell_filter_fun = partial(sc.cell_filter, num_expr_genes=0, non_zero_threshold=-1)
gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=1, non_zero_threshold=-1)
data_transf_fun = sc.no_data_transformation
print("data dimensions after preprocessing: genes x cells: ", data.shape)


# Generating labels from complete dataset
print("Train complete data")
complete_nmf = None
complete_nmf = NmfClustering(data, np.arange(data.shape[0]), num_cluster=num_cluster)
complete_nmf.add_cell_filter(cell_filter_fun)
complete_nmf.add_gene_filter(gene_filter_fun)
complete_nmf.set_data_transformation(data_transf_fun)
complete_nmf.apply(k=num_cluster, alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)
# Get labels
desc, target_nmf, trg_lbls_pred, mixed_data = method_sc3_filter(complete_nmf, data, [], cell_filter=cell_filter_fun, gene_filter=gene_filter_fun, transformation=data_transf_fun, mix=0.0, metric='euclidean', use_da_dists=False, n_trg_cluster=num_cluster)
labels = trg_lbls_pred
label_names, label_counts = np.unique(labels, return_counts = True)
print("Labels: ", label_names)
print("Counts: ", label_counts)

data = data[:, complete_nmf.remain_cell_inds]

print("Data dimensions after complete training: ", data.shape)
genes = data.shape[0]  # number of genes
n_all = data.shape[1]
n_trg = n_all - n_src[0]    # overall number of target data points

# Create results matrix
source_aris = np.zeros(reps)

# create empty job vector
jobs = []
params = []
exp_counter = 1

# Run jobs

r = 0
while r < reps:
    # Split data in source and target randomly (mode =1) or randomly stratified (mode = 2)
    src, trg, src_labels, trg_labels = split_source_target(data, labels, mode=1,
                                                           target_ncells=n_trg,
                                                           source_ncells=n_src[0])

    trg_labels = np.array(trg_labels, dtype=np.int)
    src_labels = np.array(src_labels, dtype=np.int)

    # 3.a. Subsampling order for target
    inds = np.random.permutation(trg_labels.size)

    # 3.b. Use perfect number of latent states for nmf and sc3
    src_lbl_set = np.unique(src_labels)
    n_trg_cluster = np.unique(trg_labels).size
    n_src_cluster = src_lbl_set.size

    # 3.c. train source once per repetition
    print("Train source data of rep {0}".format(r+1))
    source_nmf = None
    source_nmf = NmfClustering(src, np.arange(src.shape[0]), num_cluster=num_cluster)
    source_nmf.add_cell_filter(cell_filter_fun)
    source_nmf.add_gene_filter(gene_filter_fun)
    source_nmf.set_data_transformation(data_transf_fun)
    source_nmf.apply(k=num_cluster, alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)

    # Calculate ARIs and KTAs
    source_aris[r] = metrics.adjusted_rand_score(src_labels[source_nmf.remain_cell_inds], source_nmf.cluster_labels)
    print('SOURCE ARI = ', source_aris[r])

    r += 1

print(source_aris)

now2 = datetime.datetime.now()
print("Current date and time:")
print(now2.strftime("%Y-%m-%d %H:%M"))
print("Time passed:")
print(now2-now1)
print('Done.')