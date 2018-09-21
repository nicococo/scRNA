import logging
from functools import partial
from experiments_utils import (method_sc3_filter, method_hub, method_sc3_combined_filter,
                               acc_ari, acc_kta, acc_transferability)
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
#fname_labels = 'C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\cell_labels_major_sub'
fname_labels = 'C:\Users\Bettina\PycharmProjects2\scRNA_new\scRNA\src_c16.labels.tsv'
fname_final = 'main_results_mouse_sc316_labels_10reps.npz'

# Parameters
reps = 10 # number of repetitions, 100
n_src = [1000]  # number of source data points, 1000
percs_aim = [10, 20, 40, 70, 100, 150, 200, 300, 500]  # [10, 20, 40, 70, 100, 150, 200, 300, 500], target sizes to use. (has to be greater than num_cluster!)
mixes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Mixture parameters of transfer learning SC3, 0.3, 0.6, 0.9
min_cell_cluster = 0
min_expr_genes = 2000
non_zero_threshold = 1
perc_consensus_genes = 0.98
num_cluster = 16
nmf_alpha = 1.0
nmf_l1 = 0.75
nmf_max_iter = 4000
nmf_rel_err = 1e-3

# runtime 1 rep, 100src, 10,20, 0.3,0.6 - 15min
# runtime 1 rep, 1000src, 10,20, 0.3,0.6 - 14min

# List of accuracy functions to be used
acc_funcs = list()
acc_funcs.append(partial(acc_ari, use_strat=False))
acc_funcs.append(partial(acc_kta, mode=0))
acc_funcs.append(acc_transferability)

# Read data
#labels = np.genfromtxt(fname_labels)
foo = np.loadtxt(fname_labels,  delimiter='\t')
labels = foo[0].astype(int)
cell_ids_labels = foo[1].astype(int)
label_names, label_counts = np.unique(labels, return_counts = True)
print "Labels: ", label_names
print "Counts: ", label_counts
data = pd.read_csv(fname_data, sep='\t').values
data = data[:, cell_ids_labels]

# Cluster filter
if min_cell_cluster >0:
    print "Cluster filter"
    clusters_to_del = label_counts<min_cell_cluster
    st = set(label_names[clusters_to_del])
    cells_to_keep = [i for i, e in enumerate(labels) if e not in st]
    labels = labels[cells_to_keep]
    label_names, label_counts = np.unique(labels, return_counts = True)
    print "New labels: ", label_names
    print "New counts: ", label_counts
    data = data[:,cells_to_keep]

print "Data dimensions: genes x cells", data.shape
genes = data.shape[0]  # number of genes
n_all = data.shape[1]
n_trg = n_all - n_src[0]    # overall number of target data points
percs = np.true_divide(np.concatenate(percs_aim, n_trg), n_trg)

# Cell and gene filter and transfromation
cell_filter_fun = partial(sc.cell_filter, num_expr_genes=min_expr_genes, non_zero_threshold=non_zero_threshold)
gene_filter_fun = partial(sc.gene_filter, perc_consensus_genes=perc_consensus_genes, non_zero_threshold=non_zero_threshold)
data_transf_fun = sc.data_transformation_log2

# List of methods to be applied
methods = list()
# original SC3 (SC3 on target data)
methods.append(partial(method_sc3_filter, cell_filter=cell_filter_fun, gene_filter=gene_filter_fun, transformation=data_transf_fun, mix=0.0, metric='euclidean', use_da_dists=False))
# combined baseline SC3 (SC3 on combined source and target data)
methods.append(partial(method_sc3_combined_filter, cell_filter=cell_filter_fun, gene_filter=gene_filter_fun, transformation=data_transf_fun, metric='euclidean'))
# transfer via mixing (Transfer learning via mixing source and target before SC3)
# Experiment for all mixture_parameters
for m in mixes:
    mixed_list = list()
    mixed_list.append(partial(method_sc3_filter, cell_filter=cell_filter_fun, gene_filter=gene_filter_fun, transformation=data_transf_fun, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))


# Create results matrix
res = np.zeros((len(n_src), len(acc_funcs), reps, len(percs), len(methods)))
res_opt_mix_ind = np.zeros((len(n_src), reps, len(percs)))
res_opt_mix_aris = np.zeros((len(n_src),  reps, len(percs)))
source_aris = np.zeros((len(n_src), reps))
source_ktas = np.zeros((len(n_src), reps))

# create empty job vector
jobs = []
params = []
exp_counter = 1
num_exps = len(n_src) * reps * len(percs) * len(methods)

# Run jobs
for s in range(len(n_src)):
    accs = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
    accs_desc = list()
    opt_mix_ind = np.zeros((reps, len(percs)))
    opt_mix_aris = np.zeros((reps, len(percs)))

    num_strat = np.zeros((reps, len(percs), len(methods)))
    res_desc = []
    r = 0
    while r < reps:
            # Split data in source and target randomly (mode =1) or randomly stratified (mode = 2)
            src, trg, src_labels, trg_labels = split_source_target(data, labels, mode=1,
                                                                   target_ncells=n_trg,
                                                                   source_ncells=n_src[s])

            trg_labels = np.array(trg_labels, dtype=np.int)
            src_labels = np.array(src_labels, dtype=np.int)

            # 3.a. Subsampling order for target
            inds = np.random.permutation(trg_labels.size)

            # 3.b. Use perfect number of latent states for nmf and sc3
            src_lbl_set = np.unique(src_labels)
            n_trg_cluster = np.unique(trg_labels).size
            n_src_cluster = src_lbl_set.size

            # 3.c. train source once per repetition
            print "Train source data of rep {0}".format(r+1)
            source_nmf = None
            source_nmf = NmfClustering(src, np.arange(src.shape[0]), num_cluster=num_cluster)
            source_nmf.add_cell_filter(cell_filter_fun)
            source_nmf.add_gene_filter(gene_filter_fun)
            source_nmf.set_data_transformation(data_transf_fun)
            source_nmf.apply(k=num_cluster, alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)

            # Calculate ARIs and KTAs
            print "Evaluation of source results"
            source_ktas[s, r] = unsupervised_acc_kta(source_nmf.pp_data, source_nmf.cluster_labels, kernel='linear')
            source_aris[s, r] = metrics.adjusted_rand_score(src_labels[source_nmf.remain_cell_inds], source_nmf.cluster_labels)
            print 'ITER(', r+1, '): SOURCE KTA = ', source_ktas[s,r]
            print 'ITER(', r+1, '): SOURCE ARI = ', source_aris[s,r]

            #if source_aris[s,r] < 0.94:
            #    continue
            # 3.d. Target data subsampling loop
            #plot_cnt = 1
            print "Target data subsampling loop"
            for i in range(len(percs)):
                n_trg_perc = np.int(n_trg * percs[i])
                p_trg = trg[:, inds[:n_trg_perc]].copy()
                p_trg_labels = trg_labels[inds[:n_trg_perc]].copy()
                # 4. MTL/DA mixing parameter loop
                res_desc = list()
                for m in range(len(methods)):
                    print('Running experiment {0} of {1}: Train target data of repetition {2} - {3} source cells, {4} genes, '
                           '{5} target cells and the {6}th method'.format(exp_counter, num_exps, r+1, n_src[s], genes, n_trg_perc, m+1))
                    #plt.subplot(len(percs), len(methods), plot_cnt)
                    source_nmf.cell_filter_list = list()
                    source_nmf.gene_filter_list = list()
                    # source data is already filtered and transformed ...
                    source_nmf.add_cell_filter(lambda x: np.arange(x.shape[1]).tolist())
                    source_nmf.add_gene_filter(lambda x: np.arange(x.shape[0]).tolist())
                    source_nmf.set_data_transformation(lambda x: x)
                    desc, target_nmf, trg_lbls_pred, mixed_data = methods[m](source_nmf, p_trg.copy(), p_trg_labels.copy(), n_trg_cluster=num_cluster)
                    res_desc.append(desc)

                    print "Evaluation of target results"
                    accs_desc = list()
                    #if m >=2:
                    #    mixed_data, _, _ = target_nmf.get_mixed_data(mix=mixes[m-2])
                    for f in range(len(acc_funcs)):
                        if f != 1 or m <= 1:
                            accs[f, r, i, m], accs_descr = acc_funcs[f](target_nmf, p_trg.copy(), p_trg_labels.copy(),
                                                                        trg_lbls_pred.copy())
                        else:
                            accs[f, r, i, m], accs_descr = acc_funcs[f](target_nmf, mixed_data, p_trg_labels.copy(),
                                                                        trg_lbls_pred.copy())
                        accs_desc.append(accs_descr)

                    perc_done = round(np.true_divide(exp_counter, num_exps)*100, 4)
                    print('{0}% of experiments done.'.format(perc_done))
                    exp_counter += 1
                    #plot_cnt+=1
                    #plt.ylim(0,6)
                    #if m ==0 :
                    #    plt.ylabel('{0} target points'.format(int(percs[i]*n_trg)))

                    #if i == 0:
                    #    # plt.title('method {0}'.format(m+1))
                    #    if m == 0:
                    #        plt.title('SC3')
                    #    elif m == 1:
                    #        plt.title('SC3 Comb')
                    #    else:
                    #        plt.title('SC3 Mix with mix={0}'.format(mixes[m - 2]))
                    #if i == 2:
                    #    plt.xlabel('ordered eigenvalues')
                    opt_mix_ind[r, i] = np.argmax(accs[1, r, i, 2:])
                    opt_mix_aris[r, i] = accs[0, r, i, opt_mix_ind[r, i]+2]

            #plt.show()
            r += 1
    params.append((s))
    res[s, :, :, :, :] = accs
    res_opt_mix_ind[s,:,:] = opt_mix_ind
    res_opt_mix_aris[s,:,:] = opt_mix_aris

np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc,
         method_desc=res_desc, source_aris=source_aris, min_cell_cluster=min_cell_cluster, min_expr_genes=min_expr_genes,
         non_zero_threshold=non_zero_threshold, perc_consensus_genes=perc_consensus_genes, num_cluster=num_cluster,
         nmf_alpha=nmf_alpha, nmf_l1=nmf_l1, nmf_max_iter=nmf_max_iter, nmf_rel_err=nmf_rel_err,
         percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, mixes=mixes, res_opt_mix_ind=res_opt_mix_ind, res_opt_mix_aris=res_opt_mix_aris)


now2 = datetime.datetime.now()
print "Current date and time:"
print now2.strftime("%Y-%m-%d %H:%M")
print "Time passed:"
print now2-now1
print('Done.')


# Script parameters for Nicos cmd script:
# ARI: 0.62: --fname "C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\matrix" --fgene-ids  "C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\gene_names" --flabels "C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\cell_labels_major_sub" --cluster-range 16
