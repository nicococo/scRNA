import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy import stats
from random import randint


def plot_main(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)

    ind_common = common[-1]

    # ari overall
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0)
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
    # ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 1], axis=0)
    ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0)

    # Standard errors
    ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0, ddof=0)
    ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0, ddof=0)
    ste_ari_1_min = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0, ddof=0)

    # Plot with errorbars
    plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    plt.errorbar(percs, ari_1_min, fmt='-b', yerr=ste_ari_1_min, linewidth=2.0)
    # plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    # plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)
    # plt.plot(percs, ari_1_min, '-b', linewidth=2.0)

    plt.title('Overlap: {0}'.format(common[ind_common]), fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    #plt.semilogx()
    plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))

    plt.ylim([0., 1.])
    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_mixture_min_max(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)

    ind_common = common[-1]

    # ari overall
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0)
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
    ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 1], axis=0)
    ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0)

    # Standard errors
    # ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0, ddof=0)
    # ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0, ddof=0)

    # Plot with errorbars
    # plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    # plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor='blue', interpolate=True)

    plt.title('Overlap: {0}'.format(common[ind_common]), fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))

    plt.ylim([0., 1.])
    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_mixture_all(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common, mixes):
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 3], axis=0)

    # Standard errors
    # ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 3], axis=0, ddof=0)
    # ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0, ddof=0)

    # Plot with errorbars
    # plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    # plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    # plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor='blue', interpolate=True)
    indices = range(4,len(m_desc))
    cmap = plt.cm.get_cmap('hsv', len(indices)+1)

    for ind in indices:
        ari = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, ind], axis=0)
        # ste = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, ind], axis=0, ddof=0)
        # plt.errorbar(percs, ari, color=cmap(ind-3), yerr=ste, linewidth=2.0)
        plt.plot(percs, ari, color=cmap(ind-3), linewidth=2.0)

    plt.title('Overlap: {0}'.format(common[ind_common]), fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    #plt.semilogx()
    plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))

    plt.ylim([0., 1.])
    mixes_legend = list(map(str, mixes))
    legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    plt.legend(legend, fontsize=12, loc=3)
    plt.show()


def plot_percs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    color = ['blue']
    plt.figure(fig_num)
    fcnt = 1
    for ind_common in common:
        for i in range(1):
            # ari overall
            ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0)
            ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
            ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 1], axis=0)
            ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0)

            plt.subplot(1, len(common), fcnt)
            if i == 0:
                plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
                plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)
            plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if ind_common == 0:
                plt.title('Overlap: {0}'.format(common[ind_common]), fontsize=16)
                plt.xlabel('Target datapts', fontsize=16)
                plt.ylabel('ARI', fontsize=16)
            else:
                plt.title('{0}'.format(common[ind_common]), fontsize=16)

            plt.xlim([np.min(percs), np.max(percs)])
            # plt.semilogx()
            plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))

            plt.ylim([0., 1.])

        fcnt += 1
    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_overlapping_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    #ind_percs = len(percs)-1

    color = ['blue']
    plt.figure(fig_num)
    ind_percs = np.arange(len(percs))
    ind_methods = [1]
    for p in range(len(ind_percs)):
        for i in range(len(ind_methods)):

            # ari stratified
            ari_0_baseline = np.mean(res[ind_src, ind_genes, 1:, 0, :, ind_percs[p], 0], axis=1)
            ari_0_max      = np.mean(res[ind_src, ind_genes, 1:, 0, :, ind_percs[p], ind_methods[i]], axis=1)
            ari_0_min      = np.mean(res[ind_src, ind_genes, 1:, 0, :, ind_percs[p], ind_methods[i]+1], axis=1)
            # ari overall
            ari_1_baseline = np.mean(res[ind_src, ind_genes,  :, 1, :, ind_percs[p], 0], axis=1)
            ari_1_max      = np.mean(res[ind_src, ind_genes,  :, 1, :, ind_percs[p], ind_methods[i]], axis=1)
            ari_1_min      = np.mean(res[ind_src, ind_genes,  :, 1, :, ind_percs[p], ind_methods[i]+1], axis=1)

            plt.subplot(2, len(ind_percs), p+1)
            if i == 0:
                plt.plot(common[1:], ari_0_baseline, '--k', linewidth=2.0)
            plt.fill_between(common[1:], ari_0_max, ari_0_min, alpha=0.2, facecolor=color[i], interpolate=True)
            # plt.errorbar(common[1:], ari_0_max, ari_0_max_std)
            plt.vlines(1, 0., 1., colors='gray')
            plt.title('{0}'.format(np.int(percs[ind_percs[p]]*n_trg)), fontsize=14)
            if p == 0:
                plt.title('#Target datapts: {0}'.format(np.int(percs[ind_percs[p]]*n_trg)), fontsize=14)
                # plt.xlabel('Common cluster', fontsize=16)
                plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(common)])
            plt.xticks(common)
            plt.ylim([0., 1.])

            plt.subplot(2, len(ind_percs), p+len(ind_percs)+1)
            if i == 0:
                plt.plot(common, ari_1_baseline, '--k', linewidth=2.0)
            plt.fill_between(common, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if p == 0:
                plt.title('All cluster', fontsize=16)
                plt.xlabel('Overlap', fontsize=16)
                plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(common)])
            plt.xticks(common)
            plt.ylim([0., 1.])

    plt.legend(['SC3',
                'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_unsupervised_measures(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    fcnt = 1
    # res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    for c in range(len(accs_desc)):
        aris = np.mean(res[0, 0, :, c, :, -1, :], axis=1)
        # print np.mean(res[0, 0, :, c, :, -1, :], axis=1)
        plt.subplot(1, len(accs_desc), fcnt)
        plt.pcolor(aris, cmap=plt.get_cmap('Greens'))
        # plt.pcolor(aris, cmap=plt.get_cmap('Greens'), vmin=0., vmax=1.)
        print accs_desc[c]
        plt.title('{0}'.format(accs_desc[c]), fontsize=16)
        plt.xticks(np.arange(aris.shape[1]) + 0.5, ['SC3', 'Dist max', 'Dist min', 'Mix max', 'Mix min', 'Comb'], rotation=60)
        plt.yticks(np.arange(len(common)) + 0.5, common)
        if c == 0:
            plt.ylabel('#Common cluster', fontsize=16)
        fcnt += 1
        # plt.colorbar(ticks=[-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01])
        plt.colorbar()
    plt.show()


def plot_cluster_acc_measures(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    inds = [-2, 0, 1, 2, 3]
    fcnt = 1
    names = ['ARI', 'Euclidean','Pearson','Spearman','KTA']
    for c in range(len(inds)):
        aris = np.mean(res[0, 0, :, 3+inds[c], :, :, 1], axis=1)
        print np.mean(res[0, 0, :, 3+inds[c], :, :, 1], axis=1)

        plt.subplot(1, len(inds), fcnt)
        plt.pcolor(aris, cmap=plt.get_cmap('Purples'), vmin=0., vmax=1.)
        plt.title('{0}'.format(names[c]), fontsize=16)
        plt.xticks(np.arange(aris.shape[1])+0.5, percs, rotation=60)
        plt.yticks(np.arange(len(common))+0.5, common)
        if c == 0:
            plt.ylabel('#Common cluster', fontsize=16)
            plt.xlabel('Target k', fontsize=16)
        fcnt += 1
        plt.colorbar(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    plt.show()


def plot_transferability(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)

    print res.shape
    aris = np.mean(res[0, :, :, -1, :, -1, 0], axis=2).T
    print aris, aris.shape
    plt.subplot(1, 3, 1)
    plt.pcolor(aris, cmap=plt.get_cmap('Blues'), vmin=0., vmax=1.)
    plt.title('Transferability', fontsize=16)
    plt.xticks(np.arange(len(genes))+0.5, genes)
    plt.yticks(np.arange(len(common))+0.5, common)
    plt.xlabel('#Genes', fontsize=16)
    plt.ylabel('Overlap', fontsize=16)
    plt.colorbar(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

    plt.subplot(1, 3, 2)
    names = []
    for i in range(res.shape[2]):
        aris = np.mean(res[0, 0, i, -1, :, :, 0], axis=0)
        plt.plot(percs, aris, linewidth=2., alpha=0.7)
        names.append('Overlap: {0}'.format(i))
    # plt.title('Overlap = 2', fontsize=16)
    plt.semilogx()
    plt.xlim([np.min(percs),np.max(percs)])
    plt.xticks([np.min(percs), np.max(percs)/4, np.max(percs)/2, np.max(percs)],
                       np.array([np.min(percs), np.max(percs)/4, np.max(percs)/2,  np.max(percs)]*n_trg, dtype=np.int))
    plt.xlabel('#Target datapoints', fontsize=16)
    plt.ylabel('Transferability', fontsize=16)
    plt.ylim([0.,1.])
    plt.legend(names, loc=4, fontsize=14)

    plt.subplot(1, 3, 3)
    # n_src x genes x common x acc_funcs x reps x percs x methods
    cols = ['b', 'g', 'r', 'c', 'm']
    for i in range(res.shape[2]):
        aris   = np.mean(res[0, 0, i,  1, :, :, 0], axis=0)
        transf = np.mean(res[0, 0, i, -1, :, :, 0], axis=0)
        plt.scatter(aris, transf, 20, cols[i], alpha=0.7)

    plt.legend(names, loc=4)
    plt.plot([0, 1],[0, 1],'--k')
    plt.grid('on')
    plt.xlabel('ARI', fontsize=16)
    plt.ylabel('Transferability', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.show()


def plot_unsupervised_measures_percs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    ind_common = 0
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common, ' ind/# = ', ind_common, '/', common[ind_common]

    plt.figure(fig_num)
    accs = [0, 1, 5, 6, 7, 8]
    fcnt = 1
    # res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    for i in accs:
        for ind_common in common:
            mix_max = np.mean(res[ind_src, ind_genes, ind_common, i, :, :, 1], axis=0)
            mix_min = np.mean(res[ind_src, ind_genes, ind_common, i, :, :, 2], axis=0)

            plt.subplot(len(accs), len(common), fcnt)
            plt.fill_between(percs, mix_max, mix_min, alpha=0.2, facecolor='green', interpolate=True)
            if ind_common == 0:
                plt.title('Overlap: {0}'.format(ind_common), fontsize=16)
                plt.xlabel('Target datapts', fontsize=16)
                plt.ylabel(accs_desc[i], fontsize=16)
            else:
                plt.title('{0}'.format(ind_common), fontsize=16)
            plt.xlim([0, np.max(percs)])
            # plt.semilogx()
            # plt.xticks([np.min(percs), np.mean(percs), np.max(percs)],
            #            np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
            fcnt += 1
    plt.legend(['SC3-Dist', 'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


# Those need other experiments

def plot_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    # ind_common = 2

    color = ['blue']
    plt.figure(fig_num)
    fcnt = 1
    for c in common:
        cnt = 1
        for i in range(1):
            ind_common = c
            # ari overall
            ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
            ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt], axis=0)
            ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt+1], axis=0)
            cnt += 2

            plt.subplot(1, len(common), fcnt)
            if i == 0:
                plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
            plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if c == 0:
                plt.title('Overlap: {0}'.format(common[c]), fontsize=16)
                plt.xlabel('Target cluster', fontsize=16)
                plt.ylabel('ARI', fontsize=16)
            else:
                plt.title('{0}'.format(common[c]), fontsize=16)
            plt.xlim([2, np.max(percs)])
            plt.xticks(percs)
            plt.ylim([0., 1.])
        fcnt += 1
    plt.legend(['SC3',
                'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_src_accs(fig_num, res, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    fcnt = 1
    for c in common:
        ind_common = c
        aris = np.mean(res[:, :, ind_common, :], axis=2)
        print np.mean(res[:, :, ind_common, :], axis=2)

        plt.subplot(1, len(common), fcnt)
        plt.pcolor(aris, cmap=plt.get_cmap('Reds'), vmin=0., vmax=1.)
        plt.title('{0}'.format(common[c]), fontsize=16)
        plt.xticks(np.arange(len(genes))+0.5, genes)
        plt.yticks(np.arange(len(n_src))+0.5, n_src)
        if c == 0:
            plt.xlabel('#Genes', fontsize=16)
            plt.ylabel('#Source datapts', fontsize=16)
            plt.title('#Common cluster: {0}'.format(common[c]), fontsize=16)
        fcnt += 1
    plt.colorbar(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    plt.show()


if __name__ == "__main__":

    foo = np.load('final_toy_experiments.npz')
    #foo = np.load('mixture_parameters_experiment.npz')

    # methods = foo['methods']
    # acc_funcs = foo['acc_funcs']
    res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    source_aris = foo['source_aris'] # n_src x genes x common x reps
    accs_desc = foo['accs_desc']
    # print accs_desc
    method_desc = foo['method_desc']
    percs = foo['percs']
    # reps = foo['reps']
    genes = foo['genes']
    n_src = foo['n_src']
    n_trg = foo['n_trg']
    common = foo['common']
    mixes = foo['mixes']
    print 'n_src x genes x common x acc_funcs x reps x percs x methods'
    print 'Result dimensionality: ', res.shape
    #  Running
    # Main Plot 1 with all-fixed parameters
    plot_main(1, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Investigation of mixture parameter
    # Plot 2 with min and max
    plot_mixture_min_max(2, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Plot 3 with various mixture parameters
    plot_mixture_all(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)

    # Number of overlapping clusters
    plot_percs(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Plot 4 ARI only on overlapping clusters
    plot_overlapping_cluster(5, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # Probably need other experiments?!
    # plot_cluster(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_src_accs(7, source_aris, genes, n_src, n_trg, common)

    # I don't know whats going on...
    # plot_unsupervised_measures(8, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_cluster_acc_measures(9, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # plot_transferability(10, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_unsupervised_measures_percs(11, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)



    print('Done')