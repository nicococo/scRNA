from functools import partial

import matplotlib.pyplot as plt
from clustermap import process_jobs, Job
from experiments_utils import (method_sc3, method_sc3_combined, method_hub,
                               acc_ari, acc_reject_ari, acc_reject_auc,
                               acc_kta, acc_silhouette, acc_transferability, experiment_loop)
from scRNA.sc3_clustering import *


def plot_results(fname):
    foo = np.load(fname)
    accs = foo['res']
    accs_desc = foo['accs_desc']
    common = foo['common']
    methods = foo['methods']
    methods_desc = foo['method_desc']
    percs = foo['percs']
    n_trg = foo['n_trg']
    n_src = foo['n_src']
    print accs

    num_methods = np.int(np.floor((len(methods) - 1) / 2.))
    num_measures = accs.shape[3]

    cnt = 1
    plt.figure(1)
    for i in range(num_measures):
        title, _ = accs_desc[i]
        for j in range(num_methods):
            plt.subplot(num_measures, num_methods, cnt)
            cnt += 1
            # plot baseline
            plt.plot(common, np.mean(accs[0, 0, :, i, :, 0, 0], axis=1), '--k', linewidth=4.0)

            # # plot upper and lower bound
            res_best = np.mean(accs[0, 0, :, i, :, 0, (j+1)*2-1], axis=1)
            res_worst = np.mean(accs[0, 0, :, i, :, 0, (j+1)*2], axis=1)
            plt.plot(common, res_best, '.-r', linewidth=2.0)
            plt.plot(common, res_worst, '.-k', linewidth=2.0)
            plt.fill_between(common, res_worst, res_best, alpha=0.2, facecolor='gray', interpolate=True)
            plt.title(title, fontsize=16)
            plt.xlabel('Common cluster', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.xlim([0, np.max(common)])
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
    acc_funcs.append(acc_transferability)

    mixes = [0.1, 0.2, 0.3, 0.5]
    dist_list = list()
    for m in mixes:
        dist_list.append(partial(method_sc3, mix=m, metric='euclidean', reject_ratio=0., use_da_dists=True))

    methods = list()
    # original
    methods.append(partial(method_sc3, mix=0.0, reject_ratio=0., metric='euclidean'))
    # transfer via distances
    methods.append(partial(method_hub, method_list=dist_list, func=np.argmax))
    methods.append(partial(method_hub, method_list=dist_list, func=np.argmin))

    fname = 'transf_v1'

    percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 9, 11]]
    percs = [1.0]
    cluster_spec = [1, 2, 3, [4, 5], [6, [7, 8]]]
    n_trg = 300
    n_src = [300]
    reps = 2
    genes = [600]
    common = [0, 1, 2, 3, 4]

    res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))

    # # create empty job vector
    # jobs = []
    # params = []
    # for s in range(len(n_src)):
    #     for g in range(len(genes)):
    #         for c in range(len(common)):
    #             job = Job(experiment, ['{0}_{1}_{2}_{3}'.format(fname, s, g, c), methods, acc_funcs, 7,
    #                             reps, genes[g], common[c], cluster_spec,
    #                             percs, n_src[s], n_trg], \
    #                 mem_max='16G', mem_free='8G', name='Da-{0}-{1}-{2}'.format(s, g, c))
    #             params.append((s, g, c))
    #             jobs.append(job)
    #
    # processedJobs = process_jobs(jobs, temp_dir='/home/nico/tmp/',
    #                              local=True,
    #                              max_processes=5)
    # results = []
    # print "ret fields AFTER execution on local machine"
    # for (i, result) in enumerate(processedJobs):
    #     print "Job #", i
    #     (accs, accs_desc, m_desc) = result
    #     s, g, c = params[i]
    #     res[s, g, c, :, :, :, :] = accs
    #
    # np.savez(fname, methods=methods, acc_funcs=acc_funcs, res=res,
    #          accs_desc=accs_desc, method_desc=m_desc,
    #          percs=percs, reps=reps, genes=genes, n_src=n_src, n_trg=n_trg, common=common)
    plot_results('{0}.npz'.format(fname))
    print('Done.')
