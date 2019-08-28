###################################################
###						###
###   Complete Experiment on Hockley data       ###
###  written by Bettina Mieth, Nico Görnitz,    ###
###   Marina Vidovic and Alex Gutteridge        ###
###                                             ###
###################################################

# Please change all directories to yours!

import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import logging
logging.basicConfig()
from functools import partial
from experiments_utils import (method_sc3_ours, method_sc3_combined_ours, method_transfer_ours, acc_ari, acc_kta)
from nmf_clustering import NmfClustering_initW
from utils import *
import datetime
import pandas as pd
import numpy as np

# Running times
now1 = datetime.datetime.now()
print("Current date and time:")
print(now1.strftime("%Y-%m-%d %H:%M"))

# Data location - Please change directories to yours!
fname_data_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_mat.tsv'
fname_gene_names_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_row.tsv'
fname_cell_names_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_col.tsv'
fname_data_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_mat.tsv'
fname_gene_names_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_row.tsv'
fname_cell_ids_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_col.tsv'
fname_labels_source = '/home/bmieth/scRNAseq/data/usoskin/Usoskin_labels_only.xlsx'

# Result file
fname_final = '/home/bmieth/scRNAseq/results/jims_data/final_for_pub/jimtarget_usoskinsource_level3labels_k7.npz'

# Pre-processing parameters for gene and cell filter
min_expr_genes = 2000		
non_zero_threshold_source = 1
non_zero_threshold_target = 4
perc_consensus_genes_source = 0.94
perc_consensus_genes_target = 0.94

# Number of clusters to obtain
num_cluster = 7

# Source labels are taken at which level of the original Usoskin publication 
labels_level_ind = 3 # 1,2 or 3 

# NMF parameters
nmf_alpha = 10.0
nmf_l1 = 0.75
nmf_max_iter = 4000
nmf_rel_err = 1e-3

# Transfer learning parameters
mixes = np.arange(0,0.75,0.05)   # range of mixture parameters to use for transfer learning 

# List of accuracy functions to be used
acc_funcs = list()
acc_funcs.append(partial(acc_ari, use_strat=False))
acc_funcs.append(partial(acc_kta, mode=0))

# Read source data
data_source = pd.read_csv(fname_data_source, sep='\t', header=None).values
gene_names_source = pd.read_csv(fname_gene_names_source, sep='\t', header=None).values
cell_ids_source = pd.read_csv(fname_cell_ids_source, sep='\t', header=None).values

# Read source labels
print("Load source labels")
df = pd.read_excel(io=fname_labels_source, sheet_name='Tabelle1')
df_cell_ids = df.columns[1:]
df_cell_ids = list(x.encode('ascii','replace') for x in df_cell_ids)
src_labels = df.values[labels_level_ind-1,1:] 
src_labels = list(x.encode('ascii','replace') for x in src_labels)

label_source_names, label_source_counts = np.unique(src_labels, return_counts = True)
print("Source labels: ", label_source_names)
print("Source label counts: ", label_source_counts)

# Find cell subset/order
cell_intersection = list(set(x[0] for x in cell_ids_source.tolist()).intersection(set(x for x in df_cell_ids)))

# Adjust source data to only include cells with labels
data_indices = list(list(cell_ids_source).index(x) for x in cell_intersection)
data_source = data_source[:,data_indices]
cell_ids_source = cell_ids_source[data_indices]

# Adjust order of labels
labels_indices = list(list(df_cell_ids).index(x) for x in cell_intersection)
src_labels = np.asarray(src_labels)[labels_indices]
df_cell_ids = np.asarray(df_cell_ids)[labels_indices]

# Preprocessing source data
print("Source data dimensions before preprocessing: genes x cells", data_source.shape)
# Cell and gene filter and transformation before the whole procedure
cell_inds = sc.cell_filter(data_source, num_expr_genes=min_expr_genes, non_zero_threshold=non_zero_threshold_source)
data_source = data_source[:,cell_inds]
cell_ids_source = cell_ids_source[cell_inds]
src_labels = src_labels[cell_inds]
gene_inds = sc.gene_filter(data_source, perc_consensus_genes=perc_consensus_genes_source, non_zero_threshold=non_zero_threshold_source)
data_source = data_source[gene_inds, :]
gene_names_source = gene_names_source[gene_inds,:]
data_source = sc.data_transformation_log2(data_source)
# data is no filtered and transformed, don't do it again:
cell_filter_fun = partial(sc.cell_filter, num_expr_genes=0, non_zero_threshold=-1)
gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=1, non_zero_threshold=-1)
data_transf_fun = sc.no_data_transformation
print("source data dimensions after preprocessing: genes x cells: ", data_source.shape)

# Read target data
data_target = pd.read_csv(fname_data_target, sep='\t',header=None ).values
# reverse log2 for now (dataset is saved in log-format, so we have to undo this)
data_target = np.power(2,data_target)-1
gene_names_target = pd.read_csv(fname_gene_names_target, sep='\t', header=None).values
cell_names_target = pd.read_csv(fname_cell_names_target, sep='\t', header=None).values

# Preprocessing target data
print("Target data dimensions before preprocessing: genes x cells", data_target.shape)
# Cell and gene filter and transformation before the whole procedure
cell_inds = sc.cell_filter(data_target, num_expr_genes=min_expr_genes, non_zero_threshold=non_zero_threshold_target)
data_target = data_target[:,cell_inds]
cell_names_target = cell_names_target[cell_inds]
gene_inds = sc.gene_filter(data_target, perc_consensus_genes=perc_consensus_genes_target, non_zero_threshold=non_zero_threshold_target)
data_target = data_target[gene_inds, :]
gene_names_target = gene_names_target[gene_inds,:]
data_target = sc.data_transformation_log2(data_target)
print("Target data dimensions after preprocessing: genes x cells: ", data_target.shape)

# Find gene subset of genees that appear in both source and target
gene_intersection = list(set(x[0] for x in gene_names_target).intersection(set(x[0] for x in gene_names_source)))

# Adjust source and target data to only include overlapping genes
data_target_indices = list(list(gene_names_target).index(x) for x in gene_intersection)
data_target = data_target[data_target_indices,]
gene_names_target = gene_names_target[data_target_indices]

data_source_indices = list(list(gene_names_source).index(x) for x in gene_intersection)
data_source = data_source[data_source_indices,]
gene_names_source = gene_names_source[data_source_indices]

print("Target data dimensions after taking source intersection: genes x cells: ", data_target.shape)
print("source data dimensions after taking target intersection: genes x cells: ", data_source.shape)

# Specify dataset sizes
genes = len(gene_intersection)  # number of genes
n_src = data_source.shape[1]
n_trg = data_target.shape[1]

# List of methods to be applied
methods = list()
# original SC3 (SC3 on target data, TargetCluster)
methods.append(partial(method_sc3_ours))
# combined baseline SC3 (SC3 on combined source and target data, ConcatenateCluster)
methods.append(partial(method_sc3_combined_ours))
# transfer via mixing (Transfer learning via mixing source and target before SC3, TransferCluster)
# Experiment for all mixture_parameters
for m in mixes:
    methods.append(partial(method_transfer_ours, mix=m, calc_transferability=False))	

# Create results matrix
res = np.zeros((len(acc_funcs), len(methods)))
exp_counter = 1
num_exps = len(methods)
accs = np.zeros((len(acc_funcs), len(methods)))
trg_labels = np.zeros((n_trg, len(methods)))

# Use perfect number of latent states for nmf and sc3
src_lbl_set = np.unique(src_labels)
n_trg_cluster = num_cluster
n_src_cluster = src_lbl_set.size

## Train source 
source_nmf = NmfClustering_initW(data_source, np.arange(data_source.shape[0]), num_cluster=n_src_cluster, labels=src_labels)
source_nmf.apply(k=n_src_cluster, alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)

## Calculate ARIs and KTAs
source_aris = metrics.adjusted_rand_score(src_labels[source_nmf.remain_cell_inds], source_nmf.cluster_labels)
print('SOURCE ARI = ', source_aris)

# MTL/DA mixing parameter loop
res_desc = list()
for m in range(len(methods)):
	print(('Running experiment {0} of {1}: Train target data - {2} source cells, {3} genes, {4} target cells and the {5}th method'.format(exp_counter, num_exps,  n_src, genes, n_trg, m+1)))
	source_nmf.cell_filter_list = list()
	source_nmf.gene_filter_list = list()
	# source data is already filtered and transformed ...
	source_nmf.add_cell_filter(lambda x: np.arange(x.shape[1]).tolist())
	source_nmf.add_gene_filter(lambda x: np.arange(x.shape[0]).tolist())
	source_nmf.set_data_transformation(lambda x: x)

        # Run method
	desc, target_nmf, data_for_SC3,trg_lbls_pred = methods[m](source_nmf, data_target.copy(), num_cluster=n_trg_cluster)
	trg_labels[:,m] = trg_lbls_pred
	res_desc.append(desc)
	
        # Evaluate results
	print("Evaluation of target results")
	accs_desc = list()
	if m >=2:
		mixed_data, _, _ = target_nmf.get_mixed_data(mix=mixes[m-2], calc_transferability=False)
	for f in range(len(acc_funcs)):
		if f != 1 or m <= 1:
			accs[f,m] = accs[f,m]
			accs_descr = "No labels, no ARIs."
		else:
			accs[f, m], accs_descr = acc_funcs[f]([], mixed_data, [], trg_lbls_pred.copy())
		accs_desc.append(accs_descr)
		print(('Accuracy: {0} ({1})'.format(accs[f, m], accs_descr)))
	perc_done = round(np.true_divide(exp_counter, num_exps)*100, 4)
	print(('{0}% of experiments done.'.format(perc_done)))
	exp_counter += 1
	
        # Identify optimal mixture parameter
	opt_mix_ind = np.argmax(accs[1, 2:])
	opt_mix_aris = accs[0, int(opt_mix_ind+2)]

# Save results
res[:, :] = accs
res_opt_mix_ind = opt_mix_ind
res_opt_mix_aris = opt_mix_aris

np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc, trg_labels = trg_labels, data_target = data_target, method_desc=res_desc, source_aris=source_aris, min_expr_genes=min_expr_genes, non_zero_threshold_target=non_zero_threshold_target, non_zero_threshold_source=non_zero_threshold_source, perc_consensus_genes_source=perc_consensus_genes_source, perc_consensus_genes_target=perc_consensus_genes_target, num_cluster=num_cluster, nmf_alpha=nmf_alpha, nmf_l1=nmf_l1, nmf_max_iter=nmf_max_iter, nmf_rel_err=nmf_rel_err, genes=genes, n_src=n_src, n_trg=n_trg, mixes=mixes, res_opt_mix_ind=res_opt_mix_ind, res_opt_mix_aris=res_opt_mix_aris, labels_source = src_labels, gene_intersection=gene_intersection, cell_names_source=cell_ids_source, cell_names_target=cell_names_target, gene_names_target=gene_names_target, gene_names_source=gene_names_source)

# Print running times
now2 = datetime.datetime.now()
print("Current date and time:")
print(now2.strftime("%Y-%m-%d %H:%M"))
print("Time passed:")
print(now2-now1)
print('Done.')
