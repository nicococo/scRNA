import os.path
import sys
import logging
logging.basicConfig()
from functools import partial
from clustermap import process_jobs, Job
from experiments_utils import (method_sc3, method_hub, method_sc3_combined, acc_classification,
                               acc_ari, acc_kta, acc_silhouette, acc_transferability, experiment_loop)
from sc3_clustering import *
import pdb


def identity(x):
    return x


def experiment(fname, methods, acc_funcs, mode, reps, n_genes, n_common_cluster,
               cluster_spec, percs, n_src, n_trg):
    return experiment_loop(fname, methods, acc_funcs, mode=mode,
                    reps=reps, n_genes=n_genes, n_common_cluster=n_common_cluster, cluster_mode=False,
                    cluster_spec=cluster_spec, percs=percs, n_src=n_src, n_trg=n_trg)


def check_intermediate_results(fname, n_src, genes, common):
    params = list()
    all_files = list()
    missing_files = list()
    for s in range(len(n_src)):
        for g in range(len(genes)):
            for c in range(len(common)):
                out_fname = '{0}_{1}_{2}_{3}.npz'.format(fname,  n_src[s], genes[g], common[c])
                all_files.append(out_fname)
                params.append((s, g, c))
                if not os.path.isfile(out_fname):
                    missing_files.append(out_fname)
                    print out_fname
    return missing_files, all_files, params


def combine_intermediate_results(fname, n_src, genes, common):
    missing_files, all_files, params = check_intermediate_results(fname, n_src, genes, common)
    res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    cnt = 0
    cnt_all = 0
    for i in range(len(all_files)):
        cnt_all += 1
        if all_files[i] not in missing_files:
            foo = np.load(all_files[i])
            s, g, c = params[i]
            print foo['accs'].shape
            res[s, g, c, :, :, :, :] = foo['accs']
            cnt += 1
    print res.shape, res.size, cnt
    print cnt, '/', cnt_all
    return res


if __name__ == "__main__":
    # Final Parameters  (Figure 1-5)
    fname = 'final_toy_experiments_0817'
    fname_final = 'final_toy_experiments_0817.npz'
    reps = 20  # number of repetitions
    genes = [500]  # number of genes
    n_src = [1000]  # number of source data points
    n_trg = 2000  # overall number of target data points
    #percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # different percentages of target data points used
    percs = np.true_divide([10,20,40,70,100,150,200,300,500,700,1000],n_trg)
    cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]  # hierarchical cluster structure
    common = [0, 1, 4,5]  # different numbers of overlapping clusters in source and target data
    #common = [0,5]
    mixes = [0.0, 0.3, 0.6,  0.9,1]  # Mixture parameters of transfer learning SC3
    #mixes = [0.0,0.9]
    # List of accuracy functions to be used
    acc_funcs = list()
    # acc_funcs.append(partial(acc_ari, use_strat=True))
    acc_funcs.append(partial(acc_ari, use_strat=False))
    acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
    acc_funcs.append(partial(acc_silhouette, metric='pearson'))
    acc_funcs.append(partial(acc_silhouette, metric='spearman'))
    acc_funcs.append(partial(acc_kta, mode=0))
    #acc_funcs.append(partial(acc_kta, mode=1))
    #acc_funcs.append(partial(acc_kta, mode=2))
    #acc_funcs.append(acc_classification)
    acc_funcs.append(acc_transferability)
    # Create list of methods to be applied
    methods = list()
    # original SC3 (SC3 on target data)
    methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    # combined baseline SC3 (SC3 on combined source and target data)
    methods.append(partial(method_sc3_combined, metric='euclidean'))
    # transfer via mixing (Transfer learning via mixing source and target before SC3)
    ## Experiments keeping only min and max results from all mixture parameters
    #mixed_list = list()
    #for m in mixes:
    #    mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    #methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))
    #methods.append(partial(method_hub, method_list=mixed_list, func=np.argmin))
    # Experiment for all mixture_parameters
    for m in mixes:
        mixed_list = list()
        mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
        methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))

    ## Final Parameters  (Figure 6-8)
    ##fname = 'final_toy_experiments_part2'
    #fname_final = 'final_toy_experiments_part2.npz'
    #reps = 20  # number of repetitions
    #genes = [500, 1000, 2000]  # number of genes
    #n_src = [500, 1000, 2000]  # number of source data points
    #n_trg = 800  # overall number of target data points
    #percs = np.logspace(-1.3, -0, 12)[[0, 3, 5, 8, 11]]  # different percentages of target data points used
    #cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]  # hierarchical cluster structure
    #common = [0, 1, 2, 3, 4]  # different numbers of overlapping clusters in source and target data
    #mixes = [0.0, 0.3, 0.5, 0.7, 1.0]  # Mixture parameters of transfer learning SC3
    ## List of accuracy functions to be used
    #acc_funcs = list()
    ## acc_funcs.append(partial(acc_ari, use_strat=True))
    #acc_funcs.append(partial(acc_ari, use_strat=False))
    #acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
    #acc_funcs.append(partial(acc_silhouette, metric='pearson'))
    #acc_funcs.append(partial(acc_silhouette, metric='spearman'))
    #acc_funcs.append(partial(acc_kta, mode=0))
    ##acc_funcs.append(partial(acc_kta, mode=1))
    ##acc_funcs.append(partial(acc_kta, mode=2))
    ##acc_funcs.append(acc_classification)
    #acc_funcs.append(acc_transferability)
    ## Create list of methods to be applied
    #methods = list()
    ## original SC3 (SC3 on target data)
    #methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    ## combined baseline SC3 (SC3 on combined source and target data)
    #methods.append(partial(method_sc3_combined, metric='euclidean'))
    ## transfer via mixing (Transfer learning via mixing source and target before SC3)
    ## Experiments keeping only min and max results from all mixture parameters
    #mixed_list = list()
    #for m in mixes:
    #    mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    #methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))
    #methods.append(partial(method_hub, method_list=mixed_list, func=np.argmin))
    ## Experiment for all mixture_parameters
    ##for m in mixes:
    ##    mixed_list = list()
    ##    mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    ##    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))#

    ## For Debugging
    #percs = np.logspace(-1.3, -0, 12)[[0, 5, 11]]  # different percentages of target data points used
    #common = [0, 2, 4]  # different numbers of overlapping clusters in source and target data
    #mixes = [0.5, 1.0]
    ## Create list of methods to be applied
    #methods = list()
    ## original SC3 (SC3 on target data)
    #methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    ## combined baseline SC3 (SC3 on combined source and target data)
    #methods.append(partial(method_sc3_combined, metric='euclidean'))
    ## transfer via mixing (Transfer learning via mixing source and target before SC3)
    ## Experiments keeping only min and max results from all mixture parameters
    #mixed_list = list()
    #for m in mixes:
    #    mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    #methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))
    #methods.append(partial(method_hub, method_list=mixed_list, func=np.argmin))
    #acc_funcs = list()
    ## acc_funcs.append(partial(acc_ari, use_strat=True))
    #acc_funcs.append(partial(acc_ari, use_strat=False))
    #acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
    #acc_funcs.append(partial(acc_silhouette, metric='pearson'))
    #acc_funcs.append(partial(acc_silhouette, metric='spearman'))
    #acc_funcs.append(partial(acc_kta, mode=0))
    #acc_funcs.append(acc_transferability)
    #genes = [100, 500]  # number of genes
    #n_src = [100, 500]  # number of source data points
    #n_trg = 200  # overall number of target data points

    # Create results matrix
    res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    source_aris = np.zeros((len(n_src), len(genes), len(common), reps))

    # create empty job vector
    jobs = []
    params = []
    # Run jobs on cluster  only if the jobs havnt been done yet, i.e. out_fname dont already exist
    for s in range(len(n_src)):
        for g in range(len(genes)):
            for c in range(len(common)):
                out_fname = '{0}_{1}_{2}_{3}.npz'.format(fname, n_src[s], genes[g], common[c])
                if not os.path.isfile(out_fname):
                    print 'Added job for experiment: ', out_fname
                    job = Job(experiment, [out_fname, methods, acc_funcs, 7,
                            reps, genes[g], common[c], cluster_spec, percs, n_src[s], n_trg],
                            mem_max='16G', mem_free='8G', name='Da2-{0}-{1}-{2}'.format(n_src[s],genes[g], common[c]))
                    params.append((s, g, c))
                    jobs.append(job)
                else:
                    sys.exit("Outputfiles already exist! Change fname and fname_final or delete the existing files.")

    processedJobs = process_jobs(jobs, temp_dir='/home/bmieth/tmp/', local=False, max_processes=10)
    # results = []
    print "ret fields AFTER execution on local machine"
    for (i, result) in enumerate(processedJobs):
        print "Job #", i
        src_aris, accs, accs_desc, m_desc = result
        s, g, c = params[i]
        # res[s, g, c, :, :, :, :] = accs
        source_aris[s, g, c, :] = src_aris

    res = combine_intermediate_results(fname, n_src, genes, common)
    np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc,
             method_desc=m_desc, source_aris=source_aris,
             percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common, mixes=mixes)
    print('Done.')
