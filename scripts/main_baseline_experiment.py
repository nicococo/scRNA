import os.path
from functools import partial

import matplotlib.pyplot as plt
from clustermap import process_jobs, Job
from experiments_utils import (method_sc3, method_sc3_combined, method_hub,
                               acc_ari, acc_reject_ari, acc_reject_auc,
                               acc_kta, acc_silhouette, acc_transferability, experiment_loop)
from scRNA.sc3_clustering import *


def plot_results(fname):
    foo = np.load(fname)
    accs = foo['accs']
    accs_desc = foo['accs_desc']
    methods = foo['methods']
    methods_desc = foo['desc']
    percs = foo['percs']
    n_trg = foo['n_trg']
    n_src = foo['n_src']
    print accs

    num_methods = np.int(np.floor((len(methods) - 1) / 2.))
    num_measures = accs.shape[0]

    cnt = 1
    plt.figure(1)
    for i in range(num_measures):
        title, _ = accs_desc[i]
        for j in range(num_methods):
            plt.subplot(num_measures, num_methods, cnt)
            cnt += 1
            # plot baseline
            plt.plot(percs, np.mean(accs[i, :, :, 0], axis=0), '--k', linewidth=4.0)
            # plot upper and lower bound
            res_best = np.mean(accs[i, :, :, (j+1)*2-1], axis=0)
            res_worst = np.mean(accs[i, :, :, (j+1)*2], axis=0)
            plt.plot(percs, res_best, '.-r', linewidth=2.0)
            plt.plot(percs, res_worst, '.-k', linewidth=2.0)
            plt.fill_between(percs, res_worst, res_best, alpha=0.2, facecolor='gray', interpolate=True)
            plt.title(title, fontsize=16)
            plt.xlabel('% of target data (1.0={0})'.format(n_trg), fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.semilogx()
            plt.xlim([np.min(percs), np.max(percs)])
            plt.ylim([0., 1.])
            if i == 1:
                names = list([methods_desc[0]])
                names.append(methods_desc[(j+1)*2-1])
                names.append(methods_desc[(j+1)*2])
                # plt.legend(names, fontsize=8)
    plt.show()



def experiment(fname, methods, acc_funcs, mode, reps, n_genes, n_common_cluster,
               cluster_spec, percs, n_src, n_trg):
    return experiment_loop(fname, methods, acc_funcs, mode=mode,
                    reps=reps, n_genes=n_genes, n_common_cluster=n_common_cluster,
                    cluster_spec=cluster_spec, percs=percs, n_src=n_src, n_trg=n_trg)

if __name__ == "__main__":
    acc_funcs = list()
    acc_funcs.append(partial(acc_ari, use_strat=True, test_src_lbls=False))
    acc_funcs.append(partial(acc_ari, use_strat=False, test_src_lbls=False))
    acc_funcs.append(partial(acc_ari, use_strat=False, test_src_lbls=True))
    acc_funcs.append(partial(acc_silhouette, metric='euclidean'))
    acc_funcs.append(partial(acc_silhouette, metric='pearson'))
    acc_funcs.append(partial(acc_silhouette, metric='spearman'))
    acc_funcs.append(partial(acc_kta))
    acc_funcs.append(partial(acc_reject_ari, reject_name='Reconstr. Error', threshold=0.1))
    acc_funcs.append(partial(acc_reject_ari, reject_name='Reconstr. Error', threshold=0.2))
    acc_funcs.append(partial(acc_reject_ari, reject_name='Reconstr. Error', threshold=0.3))
    acc_funcs.append(partial(acc_reject_auc, reject_name='Reconstr. Error'))
    acc_funcs.append(partial(acc_reject_auc, reject_name='Entropy'))
    acc_funcs.append(partial(acc_reject_auc, reject_name='Kurtosis'))
    acc_funcs.append(acc_transferability)


    mixes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rratios = [0.1, 0.2, 0.3, 0.5, 0.7]

    reject_list = list()
    for m in mixes:
        for r in rratios:
            reject_list.append(partial(method_sc3, mix=m, reject_ratio=r, metric='euclidean', use_da_dists=False))

    dist_list = list()
    for m in mixes:
        dist_list.append(partial(method_sc3, mix=m, metric='euclidean', reject_ratio=0., use_da_dists=True))

    mixed_list = list()
    for m in mixes:
        mixed_list.append(partial(method_sc3, mix=m, metric='euclidean', reject_ratio=0., use_da_dists=False))

    comb_list = list()
    comb_list.append(partial(method_sc3_combined, metric='euclidean'))


    methods = list()
    # original
    methods.append(partial(method_sc3, mix=0.0, reject_ratio=0., metric='euclidean'))
    # transfer via distances
    methods.append(partial(method_hub, method_list=dist_list, func=np.argmax))
    methods.append(partial(method_hub, method_list=dist_list, func=np.argmin))
    # transfer via mixing
    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmax))
    methods.append(partial(method_hub, method_list=mixed_list, func=np.argmin))
    # transfer via mixing + rejection
    methods.append(partial(method_hub, method_list=reject_list, func=np.argmax))
    methods.append(partial(method_hub, method_list=reject_list, func=np.argmin))

    fname = 'intermediate.npz'

    percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 9, 11]]
    cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]
    n_trg = 800
    n_src = [400, 1000, 2000]
    reps = 20
    genes = [500, 1000, 2000]
    common = [0, 1, 2, 3, 4]

    res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))

    # create empty job vector
    jobs = []
    params = []
    for s in range(len(n_src)):
        for g in range(len(genes)):
            for c in range(len(common)):
                out_fname = '{0}_{1}_{2}_{3}'.format(fname, s, g, c)
                if not os.path.isfile(out_fname): 
                    job = Job(experiment, ['{0}_{1}_{2}_{3}'.format(fname, s, g, c), methods, acc_funcs, 7,
                                    reps, genes[g], common[c], cluster_spec,
                                    percs, n_src[s], n_trg], \
                        mem_max='16G', mem_free='8G', name='Da-{0}-{1}-{2}'.format(s, g, c))
                    params.append((s, g, c))
                    jobs.append(job)

    processedJobs = process_jobs(jobs, temp_dir='/home/nico/tmp/',
                                 local=False,
                                 max_processes=10)
    results = []
    print "ret fields AFTER execution on local machine"
    for (i, result) in enumerate(processedJobs):
        print "Job #", i
        (accs, accs_desc, m_desc) = result
        s, g, c = params[i]
        res[s, g, c, :, :, :, :] = accs

    np.savez('main_results_2', methods=methods, acc_funcs=acc_funcs, res=res,
             accs_desc=accs_desc, method_desc=m_desc,
             percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common)
    print('Done.')
