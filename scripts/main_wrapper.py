import logging
logging.basicConfig()
from functools import partial
from experiments_utils import (method_sc3, method_hub, method_sc3_combined,
                               acc_ari, acc_kta, acc_silhouette, acc_transferability)

from nmf_clustering import NmfClustering
from simulation import generate_toy_data, split_source_target
from utils import *
import datetime


# Debugging - 1rep start 8.42, end 9.10, 1 rep = 27.47min (with 0-5 common and mix =0 und 1)
#           - 10 reps start 10.00, end 21:17, 10 reps = 11:17 (with 0-5 common and mix =0 und 1)
#           - 100 reps start Fr,11.02, end So 13.01 , 100reps = 2days + 2h    (with 0,1,3,5 common and without mix =0 und 1), reaching 1% took 30mins
#           - 50 reps with unsupervised accuracies on mixed data: 1d 12h
#           - 10 reps with KTA on mixed data - 18h38min for 1,3,5 overlap bwz. 24h for 0,1,3,5 overlap
now1 = datetime.datetime.now()
print "Current date and time:"
print now1.strftime("%Y-%m-%d %H:%M")

fname_final = 'main_results_part1_opt_mixparam_100reps.npz'
reps = 100  # number of repetitions, 100
genes = [1000]  # number of genes, 1000
n_src = [1000]  # number of source data points, 1000
n_trg = 800  # overall number of target data points, 800
percs = np.true_divide([10,20,40,70,100,150,200,300,500,800], n_trg)     # Percentages of complete target data to use, [10,20,40,70,100,150,200,300,500,800]
cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]  # hierarchical cluster structure, 1, 2, 3, [4, 5], [6, [7, 8]]
common = [0,1,3,5]  # different numbers of overlapping clusters in source and target data, 0, 1, 3, 5
mixes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Mixture parameters of transfer learning SC3, 0.3, 0.6, 0.9

# List of accuracy functions to be used
acc_funcs = list()
acc_funcs.append(partial(acc_ari, use_strat=False))
#acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
#acc_funcs.append(partial(acc_silhouette, metric='pearson'))
#acc_funcs.append(partial(acc_silhouette, metric='spearman'))
acc_funcs.append(partial(acc_kta, mode=0))
acc_funcs.append(acc_transferability)

# Create list of methods to be applied
methods = list()
# original SC3 (SC3 on target data)
methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
# combined baseline SC3 (SC3 on combined source and target data)
methods.append(partial(method_sc3_combined, metric='euclidean'))
# transfer via mixing (Transfer learning via mixing source and target before SC3)
# Experiment for all mixture_parameters
for m in mixes:
    mixed_list = list()
    mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))

# Create results matrix
res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
# res_mixed = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
res_opt_mix_ind = np.zeros((len(n_src), len(genes), len(common), reps, len(percs)))
res_opt_mix_aris = np.zeros((len(n_src), len(genes), len(common), reps, len(percs)))

source_aris = np.zeros((len(n_src), len(genes), len(common), reps))

# create empty job vector
jobs = []
params = []
exp_counter = 1
num_exps = len(n_src) * len(genes) * len(common) * reps * len(percs) * len(methods)
# Run jobs on cluster  only if the jobs havnt been done yet, i.e. out_fname dont already exist
for s in range(len(n_src)):
    for g in range(len(genes)):
        for c in range(len(common)):
            flatten = lambda l: flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
            n_cluster = len(flatten(cluster_spec))
            #print 'Number of cluster is ', n_cluster

            accs = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
            #accs_mixed = np.zeros((len(acc_funcs), reps, len(percs), len(methods)))
            accs_desc = list()
            opt_mix_ind = np.zeros((reps, len(percs)))
            opt_mix_aris = np.zeros((reps, len(percs)))

            num_strat = np.zeros((reps, len(percs), len(methods)))
            res_desc = []
            r = 0
            while r < reps:
                # 1. Generate scRNA data
                data, labels = generate_toy_data(num_genes=genes[g], num_cells=10. * (n_trg + n_src[s]), cluster_spec=cluster_spec)
                # 2. Split source and target according to specified mode/setting
                src, trg, src_labels, trg_labels = split_source_target(data, labels,
                                                                       target_ncells=n_trg,
                                                                       source_ncells=n_src[s],
                                                                       mode=7,
                                                                       source_clusters=None,
                                                                       noise_target=False,
                                                                       noise_sd=0.1,
                                                                       common=common[c],
                                                                       cluster_spec=cluster_spec)
                trg_labels = np.array(trg_labels, dtype=np.int)
                src_labels = np.array(src_labels, dtype=np.int)
                # 3.a. Subsampling order for target
                inds = np.random.permutation(trg_labels.size)
                # 3.b. Use perfect number of latent states for nmf and sc3
                src_lbl_set = np.unique(src_labels)
                n_trg_cluster = np.unique(trg_labels).size
                n_src_cluster = src_lbl_set.size
                # 3.c. train source once per repetition
                source_nmf = NmfClustering(src, np.arange(src.shape[0]), num_cluster=n_src_cluster)
                source_nmf.apply(k=n_src_cluster, max_iter=4000, rel_err=1e-3)
                source_aris[s,g,c,r] = metrics.adjusted_rand_score(src_labels, source_nmf.cluster_labels)
                print 'ITER(', r, '): SOURCE ARI = ', source_aris[s,g,c,r]
                if source_aris[s,g,c,r] < 0.94:
                    continue
                # 3.d. Target data subsampling loop
                plot_cnt = 1
                for i in range(len(percs)):
                    n_trg_perc = np.int(n_trg * percs[i])
                    p_trg = trg[:, inds[:n_trg_perc]].copy()
                    p_trg_labels = trg_labels[inds[:n_trg_perc]].copy()
                    # 4. MTL/DA mixing parameter loop
                    res_desc = list()
                    for m in range(len(methods)):
                        print('Running experiment {0} of {1}: repetition {2} - {3} source cells, {4} genes, {5} common clusters, '
                               '{6} target cells and the {7}th method'.format(exp_counter, num_exps, r+1, n_src[s], genes[g], common[c],n_trg_perc, m+1))
                        #plt.subplot(len(percs), len(methods), plot_cnt)
                        desc, target_nmf, trg_lbls_pred = methods[m](source_nmf, p_trg.copy(), p_trg_labels.copy(),
                                                                     n_trg_cluster=n_trg_cluster)
                        res_desc.append(desc)
                        accs_desc = list()
                        if m >=2:
                            mixed_data, _, _ = target_nmf.get_mixed_data(mix=mixes[m-2])
                        for f in range(len(acc_funcs)):
                            if f != 1 or m <= 1:
                                accs[f, r, i, m], accs_descr = acc_funcs[f](target_nmf, p_trg.copy(), p_trg_labels.copy(),
                                                                            trg_lbls_pred.copy())
                            else:
                                accs[f, r, i, m], accs_descr = acc_funcs[f](target_nmf, mixed_data, p_trg_labels.copy(),
                                                                            trg_lbls_pred.copy())

                            # For scores on mixed data:
                            #accs_mixed[f, r, i, m], _ = acc_funcs[f](target_nmf, mixed_data, p_trg_labels.copy(),
                            #                                            trg_lbls_pred.copy())

                            accs_desc.append(accs_descr)

                        perc_done = round(np.true_divide(exp_counter,num_exps)*100, 4)
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
            params.append((s, g, c))
            res[s, g, c, :, :, :, :] = accs
            #res_mixed[s, g, c, :, :, :, :] = accs_mixed
            res_opt_mix_ind[s,g,c,:,:] = opt_mix_ind
            res_opt_mix_aris[s,g,c,:,:] = opt_mix_aris

np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc,
         method_desc=res_desc, source_aris=source_aris,
         percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common, mixes=mixes, res_opt_mix_ind=res_opt_mix_ind, res_opt_mix_aris=res_opt_mix_aris)
#np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res,res_mixed=res_mixed, accs_desc=accs_desc,
#         method_desc=res_desc, source_aris=source_aris,
#         percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common, mixes=mixes)
now2 = datetime.datetime.now()
print "Current date and time:"
print now2.strftime("%Y-%m-%d %H:%M")
print "Time passed:"
print now2-now1
print('Done.')
