from functools import partial

import matplotlib.pyplot as plt

from scripts.experiments_utils import *
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

    m_accs = list()
    m_names = list()
    for i in range(accs.shape[0]):
        title, _ = accs_desc[i]
        m_names.append(title)
        m_accs.append(accs[i, :, :, 0])

    if len(m_names) > 0:
        print(accs.shape)
        racc = np.array(m_accs)
        print(racc.shape)
        racc = np.swapaxes(racc, 0, 1)
        print(racc.shape)
        racc = np.swapaxes(racc, 2, 1)
        print(racc.shape)
        plot_single(i+1, 'Accuracy Measures', racc, percs, n_src, n_trg, m_names)

    plt.show()


def plot_single(fig_num, title, aris, percs, n_src, n_trg, desc):
    import scipy.stats as stats
    plt.figure(fig_num)
    np.random.seed(8)
    cols = np.random.rand(3, len(desc))
    cols[:, 0] = cols[:, 0] / np.max(cols[:, 0]) * 0.3
    for i in range(len(desc) - 1):
        cols[:, i + 1] = cols[:, i + 1] / np.max(cols[:, i + 1]) * np.max([(0.2 + np.float(i) * 0.1), 1.0])

    fmt = ['-','--','.-']

    legend = []
    corr = []
    comp = np.mean(aris[:, :, 0], axis=0)
    comp = (comp - np.mean(comp)) / (np.std(comp) * float(comp.size))
    for m in range(len(desc)):
        res = np.mean(aris[:, :, m], axis=0)
        res_stds = np.std(aris[:, :, m], axis=0)
        myFmt = fmt[0]
        if m > 0:
            myFmt = fmt[2]
        if 'KTA' in desc[m]:
            myFmt = fmt[1]

        comp2 = res
        comp2 = (comp2 - np.mean(comp2)) / (np.std(comp2))
        corr.append(np.correlate(comp, comp2)[0])
        desc[m] += ' (corr={0:1.2f})'.format(corr[-1])
        plt.plot(percs, res, myFmt, color=cols[:, m], linewidth=4)

    plt.title('{2}\n#Trg={2}, #Src={0}, #Reps={1}'.format(n_src, aris.shape[0], title, n_trg))
    plt.xlabel('#cluster'.format(n_trg), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    #plt.xlim([4e-2, 1.3])
    plt.ylim([-1., 1.])
    plt.legend(desc, loc=4)
    print(corr)


if __name__ == "__main__":
    percs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    acc_funcs = list()
    acc_funcs.append(partial(acc_ari, use_strat=False))
    # acc_funcs.append(partial(acc_ari, use_strat=True))
    acc_funcs.append(partial(acc_silhouette, use_strat=False))
    acc_funcs.append(partial(acc_silhouette, use_strat=False, metric='correlation'))
    acc_funcs.append(partial(acc_silhouette, use_strat=False, metric='jaccard'))
    acc_funcs.append(partial(acc_silhouette, use_strat=False, metric='pearson'))
    acc_funcs.append(partial(acc_silhouette, use_strat=False, metric='spearman'))
    acc_funcs.append(partial(acc_kta, kernel='linear'))
    acc_funcs.append(partial(acc_kta, kernel='rbf', param=10.0))
    # acc_funcs.append(partial(acc_kta, kernel='rbf', param=0.1))
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

    # methods.append(partial(method_da_nmf, mix=0.0))
    # methods.append(partial(method_da_nmf, mix=0.1))
    methods.append(partial(method_da_nmf, mix=0.25))

    fname = 'res_acc_nmf_r10.npz'
    experiment_loop(fname, methods, acc_funcs, mode=2, reps=10, cluster_mode=True,
                    n_trg=100, n_src=600, n_genes=1000,
                    cluster_spec=[1, 2, 3, [4, 5], [6, [7, 8]]], percs=percs)
    plot_results(fname)
