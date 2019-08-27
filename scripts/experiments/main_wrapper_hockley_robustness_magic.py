import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import logging
logging.basicConfig()
from functools import partial
#from experiments_utils import get_nn_code
from experiments_utils import (method_sc3_ours, method_sc3_combined_ours, method_transfer_ours, acc_ari, acc_kta)
from nmf_clustering import NmfClustering_initW
from utils import *
import datetime
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as spc
import scipy.spatial.distance as dist


def build_consensus_here(X):
    """
    :param X: n x cells label matrix
    :return: cells x cells consensus matrix
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    n, cells = X.shape
    consensus = np.zeros((cells, cells), dtype=np.float)
    for i in range(n):
        t = dist.squareform(dist.pdist(X[i, :].reshape(cells, 1)))
        t = np.array(t, dtype=np.int)
        # print np.unique(t)
        t[t != 0] = -1
        t[t == 0] = +1
        t[t == -1] = 0
        # print np.unique(t)
        consensus += np.array(t, dtype=np.float)
    consensus /= np.float(n)
    return consensus
	
	

def consensus_clustering_here(consensus, n_components=5):
    """
    :param consensus: cells x cells consensus matrix
    :param n_components: number of clusters
    :return: cells x 1 labels
    """
    # print 'SC3 Agglomorative hierarchical clustering.'
    # condensed distance matrix
    cdm = dist.pdist(consensus)
    # hierarchical clustering (SC3: complete agglomeration + cutree)
    hclust = spc.complete(cdm)
    cutree = spc.cut_tree(hclust, n_clusters=n_components)
    labels = cutree.reshape(consensus.shape[0])
    # Below is the hclust code for the older version, fyi
    # hclust = spc.linkage(cdm)
    # labels = spc.fcluster(hclust, n_components, criterion='maxclust')
    return labels
	

now1 = datetime.datetime.now()
print("Current date and time:")
print(now1.strftime("%Y-%m-%d %H:%M"))

# Data location
fname_data_target = '/home/bmieth/scRNAseq/data/Jim/jim_data_magic.tsv'
fname_gene_names_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_row.tsv'
fname_cell_names_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_col.tsv'
fname_data_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_data_magic.tsv'
fname_gene_names_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_row.tsv'
fname_cell_ids_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_col.tsv'
fname_labels_source = '/home/bmieth/scRNAseq/data/usoskin/Usoskin_labels_only.xlsx'

# Result file
fname_final = '/home/bmieth/scRNAseq/results/jims_data/magic/jimtarget_usoskinsource_magic_1000reps.npz'

# Parameters
reps = 1000

mixes = np.arange(0,0.75,0.05)   # np.arange(0,0.7,0.05) 
min_expr_genes = 0
non_zero_threshold_source = -1
non_zero_threshold_target = -1
perc_consensus_genes_source = 1
perc_consensus_genes_target = 1
num_cluster = 7

labels_level_ind = 3 # 1,2,3 level for source labels

nmf_alpha = 10.0
nmf_l1 = 0.75
nmf_max_iter = 4000
nmf_rel_err = 1e-3

# List of accuracy functions to be used
acc_funcs = list()
acc_funcs.append(partial(acc_ari, use_strat=False))
acc_funcs.append(partial(acc_kta, mode=0))
##acc_funcs.append(acc_transferability)
#acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
#acc_funcs.append(partial(acc_silhouette, metric='pearson'))
#acc_funcs.append(partial(acc_silhouette, metric='spearman'))
#acc_funcs.append(acc_classification)

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
cell_filter_fun = partial(sc.cell_filter, num_expr_genes=0, non_zero_threshold=-1)
gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=1, non_zero_threshold=-1)
data_transf_fun = sc.no_data_transformation
print("source data dimensions after preprocessing: genes x cells: ", data_source.shape)

# Read target data
data_target = pd.read_csv(fname_data_target, sep='\t', header=None).values
## reverse log2 for now
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

# Find gene subset
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

genes = len(gene_intersection)  # number of genes
n_src = data_source.shape[1]
n_trg = data_target.shape[1]

# List of methods to be applied
methods = list()
# original SC3 (SC3 on target data)
methods.append(partial(method_sc3_ours))
# combined baseline SC3 (SC3 on combined source and target data)
methods.append(partial(method_sc3_combined_ours))
# transfer via mixing (Transfer learning via mixing source and target before SC3)
# Experiment for all mixture_parameters
for m in mixes:
    methods.append(partial(method_transfer_ours, mix=m, calc_transferability=False))	

# Create results matrix
res = np.zeros((reps, len(acc_funcs), len(methods)))
res_opt_mix_ind = np.zeros((reps,1))
res_opt_mix_aris = np.zeros((reps,1))
exp_counter = 1
num_exps = len(methods)*reps
accs = np.zeros((len(acc_funcs), len(methods)))
trg_labels = np.zeros((n_trg, len(methods)))
trg_labels_reps = np.zeros((n_trg, len(methods), reps))

# Use perfect number of latent states for nmf and sc3
#src_labels = np.array(src_labels, dtype=np.int)
src_lbl_set = np.unique(src_labels)
n_trg_cluster = num_cluster
n_src_cluster = src_lbl_set.size

## Train source once per repetition
source_nmf = NmfClustering_initW(data_source, np.arange(data_source.shape[0]), num_cluster=n_src_cluster, labels=src_labels)
source_nmf.apply(k=n_src_cluster, alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)

## Calculate ARIs and KTAs
source_aris = metrics.adjusted_rand_score(src_labels[source_nmf.remain_cell_inds], source_nmf.cluster_labels)
print('SOURCE ARI = ', source_aris)

# MTL/DA mixing parameter loop
for r in range(reps):
	res_desc = list()
	for m in range(len(methods)):
		print(('Running experiment {0} of {1}: Train target data - {2} source cells, {3} genes, {4} target cells and the {5}th method, rep = {6}'.format(exp_counter, num_exps,  n_src, genes, n_trg, m+1, r)))
		source_nmf.cell_filter_list = list()
		source_nmf.gene_filter_list = list()
		# source data is already filtered and transformed ...
		source_nmf.add_cell_filter(lambda x: np.arange(x.shape[1]).tolist())
		source_nmf.add_gene_filter(lambda x: np.arange(x.shape[0]).tolist())
		source_nmf.set_data_transformation(lambda x: x)
		desc, target_nmf, data_for_SC3,trg_lbls_pred = methods[m](source_nmf, data_target.copy(), num_cluster=n_trg_cluster)
		trg_labels[:,m] = trg_lbls_pred
		res_desc.append(desc)

		print("Evaluation of target results")
		accs_desc = list()
		if m >= 2:
			mixed_data, _, _ = target_nmf.get_mixed_data(mix=mixes[m-2], calc_transferability=False)
		for f in range(len(acc_funcs)):
			if f != 1 or m <= 1:
				accs[f,m] = accs[f,m]
				accs_descr = "No labels, no ARIs."
				#accs[f, m], accs_descr = acc_funcs[f]([], data_target.copy(), p_trg_labels.copy(), trg_lbls_pred.copy())
			else:
				accs[f, m], accs_descr = acc_funcs[f]([], mixed_data, [], trg_lbls_pred.copy())
			accs_desc.append(accs_descr)
			print(('Accuracy: {0} ({1})'.format(accs[f, m], accs_descr)))
		perc_done = round(np.true_divide(exp_counter, num_exps)*100, 4)
		print(('{0}% of experiments done.'.format(perc_done)))
		exp_counter += 1
		if m >= 2:
		    opt_mix_ind = np.argmax(accs[1, 2:])
		    opt_mix_aris = accs[0, int(opt_mix_ind+2)]
	
	res[r, :, :] = accs
	res_opt_mix_ind[r] = opt_mix_ind
	res_opt_mix_aris[r] = opt_mix_aris
	
	trg_labels_reps[:,0,r]=trg_labels[:,0]
	trg_labels_reps[:,1,r]=trg_labels[:,1]
	trg_labels_reps[:,2,r]=trg_labels[:, opt_mix_ind+2] 

# building consensus matrices
consensus_mat_sc3 = build_consensus_here(trg_labels_reps[:,0,:].T)
consensus_mat_sc3_comb = build_consensus_here(trg_labels_reps[:,1,:].T)
consensus_mat_sc3_mix = build_consensus_here(trg_labels_reps[:,2,:].T)

# consensus clustering
cons_clustering_sc3 = consensus_clustering_here(consensus_mat_sc3, n_components=n_trg_cluster)
cons_clustering_sc3_comb = consensus_clustering_here(consensus_mat_sc3_comb, n_components=n_trg_cluster)
cons_clustering_sc3_mix = consensus_clustering_here(consensus_mat_sc3_mix, n_components=n_trg_cluster)
	
#cnt = 0.
#consensus = np.zeros((n_trg, n_trg))
#for cluster in self.intermediate_clustering_list:
#	for t in range(len(transf)):
#		_, deigv = transf[t]
#		labels = list()
#		for d in range_inds:
#			labels.append(cluster(deigv[:, 0:d].reshape((deigv.shape[0], d))))
#			if self.consensus_mode == 0:
#				consensus += build_consensus_here(np.array(labels[-1]))
#				cnt += 1.
#consensus /= cnt

np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc, trg_labels = trg_labels, data_target = data_target, method_desc=res_desc, source_aris=source_aris, min_expr_genes=min_expr_genes, non_zero_threshold_target=non_zero_threshold_target, non_zero_threshold_source=non_zero_threshold_source, perc_consensus_genes_source=perc_consensus_genes_source, perc_consensus_genes_target=perc_consensus_genes_target, num_cluster=num_cluster, nmf_alpha=nmf_alpha, nmf_l1=nmf_l1, nmf_max_iter=nmf_max_iter, nmf_rel_err=nmf_rel_err, genes=genes, n_src=n_src, n_trg=n_trg, mixes=mixes, res_opt_mix_ind=res_opt_mix_ind, res_opt_mix_aris=res_opt_mix_aris, labels_source = src_labels, gene_intersection=gene_intersection, cell_names_source=cell_ids_source, cell_names_target=cell_names_target, gene_names_target=gene_names_target, gene_names_source=gene_names_source, cons_clustering_sc3=cons_clustering_sc3, cons_clustering_sc3_comb=cons_clustering_sc3_comb, cons_clustering_sc3_mix=cons_clustering_sc3_mix, reps=reps, trg_labels_reps=trg_labels_reps)
#np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc, trg_labels = trg_labels, data_target = data_target, method_desc=res_desc, source_aris=source_aris, min_expr_genes=min_expr_genes, non_zero_threshold_target=non_zero_threshold_target, non_zero_threshold_source=non_zero_threshold_source, perc_consensus_genes_source=perc_consensus_genes_source, perc_consensus_genes_target=perc_consensus_genes_target, num_cluster=num_cluster, nmf_alpha=nmf_alpha, nmf_l1=nmf_l1, nmf_max_iter=nmf_max_iter, nmf_rel_err=nmf_rel_err, genes=genes, n_src=n_src, n_trg=n_trg, mixes=mixes, res_opt_mix_ind=res_opt_mix_ind, res_opt_mix_aris=res_opt_mix_aris, labels_source = src_labels, gene_intersection=gene_intersection, cell_names_source=cell_ids_source, cell_names_target=cell_names_target, gene_names_target=gene_names_target, gene_names_source=gene_names_source, cons_clustering_sc3=cons_clustering_sc3, cons_clustering_sc3_comb=cons_clustering_sc3_comb,  reps=reps, trg_labels_reps=trg_labels_reps)

now2 = datetime.datetime.now()
print("Current date and time:")
print(now2.strftime("%Y-%m-%d %H:%M"))
print("Time passed:")
print(now2-now1)
print('Done.')