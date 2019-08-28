###################################################
###												###
###   Complete Experiment on generated data     ###
###	  written by Bettina Mieth, Nico Görnitz,   ###
###   Marina Vidovic and Alex Gutteridge        ###
###                                             ###
###################################################

# Please change all directories to yours!

import sys
sys.path.append('C:/Users/Bettina/PycharmProjects2/scRNA_new/final_scripts/implementations')
import logging
logging.basicConfig()
from functools import partial
from experiments_utils import (method_sc3_ours, method_sc3_combined_ours, method_transfer_ours, acc_ari, acc_kta)
from nmf_clustering import NmfClustering_initW
from simulation import generate_toy_data, split_source_target
from utils import *
import datetime
import numpy as np

# Running times 

now1 = datetime.datetime.now()
print("Current date and time:")
print(now1.strftime("%Y-%m-%d %H:%M"))

# Directories and parameters

fname_final ='/home/bmieth/scRNAseq/results/toy_data_final/main_results_toydata_revision_realistic_counts_complete.npz' # directory to save results in
reps = 100  # number of repetitions, 100
genes = [10000]  # number of genes, 1000
n_src = [1000]  # number of source data points, 1000
n_trg = 800  # overall number of target data points, 800
percs = np.true_divide([10,25,50,100,200,400, 600, 800], n_trg)     # Percentages of complete target data to use, [10,25,50,100,200,400, 600, 800]
cluster_spec =[1, 2, 3, [4, 5], [6, [7, 8]]]    # Spectrum of clusters defining a hierarchical clustering structure with eight leaf nodes, [1, 2, 3, [4, 5], [6, [7, 8]]]
splitting_mode = 7 # mode defining how data is split in source and target set, 2 = randomly stratified splitting, 7 = splitting according to common, 7
common = [0,3,5] # different numbers of overlapping top node clusters in source and target data, [0,3,5]

# data generation parameters
gamma_rate=0.1

# NMF parameters
nmf_alpha = 10.0
nmf_l1 = 0.75
nmf_max_iter = 4000
nmf_rel_err = 1e-3

# Mixture parameters to choose from (algorithm will automatically pick one of them)
mixes = [0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]  # Mixture parameters of transfer learning, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 

# List of accuracy functions to be used
acc_funcs = list()
acc_funcs.append(partial(acc_ari, use_strat=False))
acc_funcs.append(partial(acc_kta, mode=0))

# Create list of methods to be applied
methods = list()
# original SC3 (SC3 on target data, TargetCluster)
methods.append(partial(method_sc3_ours))
# combined baseline SC3 (SC3 on combined source and target data, ConcatenateCluster)
methods.append(partial(method_sc3_combined_ours))
# transfer via mixing (Transfer learning via mixing source and target before SC3, TransferCluster)
# Run TransferCluster for all mixture_parameters
for m in mixes:
    methods.append(partial(method_transfer_ours, mix=m, calc_transferability=False))

# Create results matrices
res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
res_opt_mix_ind = np.zeros((len(n_src), len(genes), len(common), reps, len(percs)))
res_opt_mix_aris = np.zeros((len(n_src), len(genes), len(common), reps, len(percs)))
source_aris = np.zeros((len(n_src), len(genes), len(common), reps))

# Setup experiment
params = []
exp_counter = 1
num_exps = len(n_src) * len(genes) * len(common) * reps * len(percs) * len(methods)

# Run experiment
for s in range(len(n_src)):
    for g in range(len(genes)):
        for c in range(len(common)):
            flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
            n_cluster = len(flatten(cluster_spec))
            accs = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
            accs_desc = list()
            opt_mix_ind = np.zeros((reps, len(percs)))
            opt_mix_aris = np.zeros((reps, len(percs)))

            num_strat = np.zeros((reps, len(percs), len(methods)))
            res_desc = []
            r = 0
            while r < reps:
                # 1. Generate scRNA data
                data, labels = generate_toy_data(num_genes=genes[g], num_cells=10. * (n_trg + n_src[s]), cluster_spec=cluster_spec, gamma_rate=gamma_rate)
                # 2. Split source and target according to specified mode/setting
                src, trg, src_labels, trg_labels = split_source_target(data, labels, target_ncells=n_trg, source_ncells=n_src[s], mode=splitting_mode, common=common[c], source_clusters=None, noise_target=False, noise_sd=0.1, cluster_spec=cluster_spec)
                trg_labels = np.array(trg_labels, dtype=np.int)
                src_labels = np.array(src_labels, dtype=np.int)
                # 3.a. Subsampling order for target
                inds = np.random.permutation(trg_labels.size)
                # 3.b. Use perfect number of latent states for nmf and sc3
                src_lbl_set = np.unique(src_labels)
                n_trg_cluster = np.unique(trg_labels).size
                n_src_cluster = src_lbl_set.size
                # 3.c. train source once per repetition
                source_nmf = NmfClustering_initW(src, np.arange(src.shape[0]), num_cluster=n_src_cluster, labels=src_labels)
                source_nmf.apply(k=n_src_cluster,  alpha=nmf_alpha, l1=nmf_l1, max_iter=nmf_max_iter, rel_err=nmf_rel_err)
                # Evaluate source clustering
                source_aris[s,g,c,r] = metrics.adjusted_rand_score(src_labels, source_nmf.cluster_labels)
                print('ITER(', r, '): SOURCE ARI = ', source_aris[s,g,c,r])

                # 3.d. Target data subsampling loop
                for i in range(len(percs)):
                    n_trg_perc = np.int(n_trg * percs[i])
                    p_trg = trg[:, inds[:n_trg_perc]].copy()
                    p_trg_labels = trg_labels[inds[:n_trg_perc]].copy()
                    # 4. MTL/DA mixing parameter loop
                    res_desc = list()
                    for m in range(len(methods)):
                        # Run method
                        print(('Running experiment {0} of {1}: repetition {2} - {3} source cells, {4} genes, {5} common clusters, {6} target cells and the {7}th method'.format(exp_counter, num_exps, r+1, n_src[s], genes[g], common[c],n_trg_perc, m+1)))
                        desc, target_nmf, data_for_SC3,trg_lbls_pred = methods[m](source_nmf, p_trg, num_cluster=n_trg_cluster)
                        res_desc.append(desc)
                        accs_desc = list()
                        if m >=2:
                            mixed_data, _, _ = target_nmf.get_mixed_data(mix=mixes[m-2], calc_transferability=False)
                        # Evaluate results
                        for f in range(len(acc_funcs)):
                            if f != 1 or m <= 1:
                                accs[f, r, i, m], accs_descr = acc_funcs[f]([], p_trg.copy(), p_trg_labels.copy(), trg_lbls_pred.copy())
                            else:
                                accs[f, r, i, m], accs_descr = acc_funcs[f]([], mixed_data, p_trg_labels.copy(), trg_lbls_pred.copy())
                            accs_desc.append(accs_descr)

                        perc_done = round(np.true_divide(exp_counter,num_exps)*100, 4)
                        print(('{0}% of experiments done.'.format(perc_done)))
                        exp_counter += 1
                    # Identify optimal mixture parameter
                    opt_mix_ind[r, i] = np.argmax(accs[1, r, i, 2:])
                    opt_mix_aris[r, i] = accs[0, r, i, int(opt_mix_ind[r, i]+2)]
                r += 1
            params.append((s, g, c))
            res[s, g, c, :, :, :, :] = accs
            res_opt_mix_ind[s,g,c,:,:] = opt_mix_ind
            res_opt_mix_aris[s,g,c,:,:] = opt_mix_aris

# Save results            
np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc, method_desc=res_desc, source_aris=source_aris, percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common, mixes=mixes, res_opt_mix_ind=res_opt_mix_ind, res_opt_mix_aris=res_opt_mix_aris)

# Show running times
now2 = datetime.datetime.now()
print("Current date and time:")
print(now2.strftime("%Y-%m-%d %H:%M"))
print("Time passed:")
print(now2-now1)
print('Done.')
