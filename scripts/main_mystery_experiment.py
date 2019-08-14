from functools import partial

import matplotlib.pyplot as plt

from scripts.experiments_utils import method_sc3, method_da_nmf, acc_ari, experiment_loop
from simulation import *


def plot_results(fname):
    foo = np.load(fname)
    accs = foo['accs']
    accs_desc = foo['accs_desc']
    methods = foo['methods']
    methods_desc = foo['desc']
    percs = foo['percs']
    n_trg = foo['n_trg']
    n_src = foo['n_src']
    print(accs)

    for i in range(accs.shape[0]):
        title, _ = accs_desc[i]
        plot_single(i, title, accs[i, :, :, :], percs, n_src, n_trg, methods_desc)
    plt.show()


def plot_single(fig_num, title, aris, percs, n_src, n_trg, desc):
    plt.figure(fig_num)
    plt.subplot(1, 2, 1)
    np.random.seed(8)
    cols = np.random.rand(3, len(desc))
    cols[:, 0] = cols[:, 0] / np.max(cols[:, 0]) * 0.3
    for i in range(len(desc) - 1):
        cols[:, i + 1] = cols[:, i + 1] / np.max(cols[:, i + 1]) * np.max([(0.2 + np.float(i) * 0.1), 1.0])

    legend = []
    aucs = np.zeros(len(desc))
    for m in range(len(desc)):
        res = np.mean(aris[:, :, m], axis=0)
        res_stds = np.std(aris[:, :, m], axis=0)

        if m > 0:
            plt.plot(percs, res, '-', color=cols[:, m], linewidth=4)
            # plt.errorbar(percs, res, res_stds, fmt='-', color=cols[:, m], linewidth=4, elinewidth=1)
        else:
            plt.plot(percs, res, '--', color=cols[:, m], linewidth=4)
            # plt.errorbar(percs, res, res_stds, fmt='--', color=cols[:, m], linewidth=4, elinewidth=1)
        aucs[m] = np.trapz(res, percs)

    plt.title('{2}\n#Src={0}, #Reps={1}'.format(n_src, aris.shape[0], title))
    plt.xlabel('Fraction of target samples ('
               '1.0={0} samples)'.format(n_trg), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim([4e-2, 1.3])
    plt.ylim([0., 1.])

    plt.legend(desc, loc=4)
    # plt.legend({'100% Transfer SC3 w/ Adaptive Range','100% Transfer SC3 w/o Adaptive Range'}, loc=4)
    plt.semilogx()

    plt.subplot(1, 2, 2)
    plt.title('Overall performance')
    plt.bar(np.arange(aucs.size), aucs, color=cols.T)
    plt.xticks(np.arange(len(desc))-0.0, desc, rotation=45)
    plt.ylabel('Area under curve', fontsize=14)


if __name__ == "__main__":
    percs = np.logspace(-1.3, -0, 12)[[0, 1, 2, 3, 4, 5, 6, 9, 11]]
    # percs = [0.05, 0.075, 0.1, 0.3, 0.5, 1.0]

    acc_funcs = list()
    acc_funcs.append(partial(acc_ari, use_strat=False))
    # acc_funcs.append(partial(acc_ari, use_strat=True))
    # acc_funcs.append(partial(acc_silhouette, use_strat=False))
    # acc_funcs.append(partial(acc_silhouette, use_strat=True))
    # acc_funcs.append(partial(acc_reject, reject_name='KTA kurt1'))
    # acc_funcs.append(partial(acc_reject, reject_name='KTA kurt2'))
    # acc_funcs.append(partial(acc_reject, reject_name='KTA kurt3'))
    # acc_funcs.append(partial(acc_reject, reject_name='Kurtosis'))
    # acc_funcs.append(partial(acc_reject, reject_name='Entropy'))
    # acc_funcs.append(partial(acc_reject, reject_name='Diffs'))

    # acc_funcs.append(partial(acc_reject_ari, reject_name='Entropy', threshold=0.0))
    # acc_funcs.append(partial(acc_reject_ari, reject_name='Entropy', threshold=0.1))
    # acc_funcs.append(partial(acc_reject_ari, reject_name='Entropy', threshold=0.2))
    # acc_funcs.append(partial(acc_reject_ari, reject_name='Entropy', threshold=0.5))

    methods = list()
    # methods.append(partial(method_sc3, mix=0.0, metric='euclidean'))
    # methods.append(partial(method_sc3, mix=0.5, metric='euclidean'))
    methods.append(partial(method_sc3, mix=1.0, metric='euclidean'))
    methods.append(partial(method_sc3, mix=0.9999999999999999, metric='euclidean'))
    methods.append(partial(method_sc3, mix=0.99999999999999, metric='euclidean'))
    methods.append(partial(method_sc3, mix=0.999999999999, metric='euclidean'))
    methods.append(partial(method_sc3, mix=0.9, metric='euclidean'))
    # methods.append(partial(method_nmf))
    # methods.append(partial(method_da_nmf, mix=0.0))
    # methods.append(partial(method_da_nmf, mix=0.1))
    # methods.append(partial(method_da_nmf, mix=0.25))
    # methods.append(partial(method_da_nmf, mix=0.5))
    # methods.append(partial(method_da_nmf, mix=0.75))
    # methods.append(partial(method_da_nmf, mix=1.0))

    fname = 'res_mystery_r10.npz'
    # experiment_loop(fname, methods, acc_funcs, mode=4, reps=10,
    #                 cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]], percs=percs)
    plot_results(fname)
