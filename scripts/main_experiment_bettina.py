import os.path
import logging
logging.basicConfig()
from functools import partial
from clustermap import process_jobs, Job
from experiments_utils import (method_sc3, method_hub, method_sc3_combined, acc_classification,
                               acc_ari, acc_kta, acc_silhouette, acc_transferability, experiment_loop)
from sc3_clustering import *
import pdb


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
                out_fname = '{0}_{1}_{2}_{3}.npz'.format(fname, s, g, c)
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

    fname = 'final_toy_experiments'
    fname_final = 'final_toy_experiments.npz'

    # Final Parameters
    reps = 20  # number of repetitions
    genes = [500]  # number of genes
    n_src = [1000]  # number of source data points
    n_trg = 800  # overall number of target data points
    percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 9, 11]]  # different percentages of target data points used
    cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]  # hierarchical cluster structure
    common = [0, 1, 2, 3, 4]  # different numbers of overlapping clusters in source and target data
    mixes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Mixture parameters of transfer learning SC3

    # For debugging
    reps = 3  # number of repetitions
    genes = [500]  # number of genes
    n_src = [1000]  # number of source data points
    n_trg = 800  # overall number of target data points
    percs = np.logspace(-1.3, -0, 12)[[0, 5, 11]]  # different percentages of target data points used
    cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]] # hierarchical cluster structure
    common = [0, 1]  # different numbers of overlapping clusters in source and target data
    mixes = [0.0,  0.5,  1.0] # Mixture parameters of transfer learning SC3

    # List of accuracy functions to be used
    acc_funcs = list()
    acc_funcs.append(partial(acc_ari, use_strat=True))
    acc_funcs.append(partial(acc_ari, use_strat=False))
    acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
    acc_funcs.append(partial(acc_silhouette, metric='pearson'))
    acc_funcs.append(partial(acc_silhouette, metric='spearman'))
    acc_funcs.append(partial(acc_kta, mode=0))
    acc_funcs.append(partial(acc_kta, mode=1))
    acc_funcs.append(partial(acc_kta, mode=2))
    acc_funcs.append(acc_classification)
    acc_funcs.append(acc_transferability)

    # Create list of methods to be applied
    methods = list()
    # original SC3 (SC3 on target data)
    methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    # combined baseline SC3 (SC3 on combined source and target data)
    methods.append(partial(method_sc3_combined, metric='euclidean'))
    # transfer via mixing (Transfer learning via mixing source and target before SC3)
    mixed_list = list()
    for m in mixes:
        mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', calc_transferability=False, use_da_dists=False))
    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))
    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmin))

    # Create results matrix
    res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    source_aris = np.zeros((len(n_src), len(genes), len(common), reps))

    # pdb.set_trace()
    # create empty job vector
    jobs = []
    params = []
    # Run jobs on cluster
    for s in range(len(n_src)):
        for g in range(len(genes)):
            for c in range(len(common)):
                out_fname = '{0}_{1}_{2}_{3}.npz'.format(fname, s, g, c)
                if not os.path.isfile(out_fname):
                    print 'Added job for experiment: ', out_fname
                    job = Job(experiment, ['{0}_{1}_{2}_{3}'.format(fname, s, g, c), methods, acc_funcs, 7,
                            reps, genes[g], common[c], cluster_spec, percs, n_src[s], n_trg],
                            mem_max='16G', mem_free='8G', name='Da2-{0}-{1}-{2}'.format(s, g, c))
                    params.append((s, g, c))
                    jobs.append(job)

    processedJobs = process_jobs(jobs, temp_dir='/home/bmieth/tmp/', local=False, max_processes=10)
    results = []
    print "ret fields AFTER execution on local machine"
    # pdb.set_trace()
    for (i, result) in enumerate(processedJobs):
        print "Job #", i
        # pdb.set_trace()
        src_aris, accs, accs_desc, m_desc = result
        s, g, c = params[i]
        res[s, g, c, :, :, :, :] = accs
        source_aris[s, g, c, :] = src_aris

    res = combine_intermediate_results(fname, n_src, genes, common)
    np.savez(fname_final, methods=methods, acc_funcs=acc_funcs, res=res, accs_desc=accs_desc,
             method_desc=m_desc, source_aris=source_aris,
             percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common)
    print('Done.')
