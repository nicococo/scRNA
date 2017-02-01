import matplotlib.pyplot as plt
import numpy as np


def plot_percs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    ind_common = 2
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common, ' ind/# = ', ind_common, '/', common[ind_common]

    color = ['blue', 'green', 'red']
    plt.figure(fig_num)
    inds = [0, 1, 2, 3, 4]
    fcnt = 1
    for c in inds:
        cnt = 1
        for i in range(2):
            ind_common = c
            # ari overall
            ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
            ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt], axis=0)
            ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt+1], axis=0)
            cnt += 2

            plt.subplot(1, len(inds), fcnt)
            if i==0:
                plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
            plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if c==0:
                plt.title('#Common cluster: {0}'.format(common[c]), fontsize=16)
            else:
                plt.title('{0}'.format(common[c]), fontsize=16)
            plt.xlabel('Target datapts', fontsize=16)
            plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(percs)])
            plt.semilogx()
            plt.xticks([np.min(percs), np.mean(percs), np.max(percs)],
                       np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
        fcnt += 1
    plt.legend(['SC3',
                'SC3-Dist   (upper and lower bound)',
                'SC3-Mix    (upper and lower bound)'], fontsize=12, loc=3)
    plt.show()


def plot_overlapping_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 1
    ind_src = 0
    ind_percs = 7
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Percs ', percs, ' ind/# = ', ind_percs, '/', percs[ind_percs]

    color = ['blue', 'green', 'red']
    plt.figure(fig_num)
    ind_percs = [0, 3, 5, 7]
    ind_methods = [1, 3]
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
            plt.title('Accuracy (common cluster only)', fontsize=16)
            plt.xlabel('Common cluster', fontsize=16)
            plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(common)])
            plt.xticks(common)
            plt.ylim([0., 1.])

            plt.subplot(2, len(ind_percs), p+len(ind_percs)+1)
            if i==0:
                plt.plot(common, ari_1_baseline, '--k', linewidth=2.0)
            plt.fill_between(common, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            plt.title('Accuracy (all cluster)', fontsize=16)
            plt.xlabel('Common cluster', fontsize=16)
            plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(common)])
            plt.xticks(common)
            plt.ylim([0., 1.])

    plt.legend(['SC3',
                'SC3-Dist   (upper and lower bound)',
                'SC3-Mix    (upper and lower bound)',
                'SC3-Reject (upper and lower bound)'], fontsize=12, loc=3)
    plt.show()


def plot_reject_mix(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 1
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common

    color = ['green', 'red']
    plt.figure(fig_num)
    inds = [0, 1, 2, 3, 4]
    fcnt = 1
    for c in inds:
        cnt = 3
        for i in range(2):
            ind_common = c
            # ari overall
            ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
            ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt], axis=0)
            ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt+1], axis=0)
            cnt += 2

            plt.subplot(1, len(inds), fcnt)
            if i==0:
                plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
            plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if c==0:
                plt.title('#Common cluster: {0}'.format(common[c]), fontsize=16)
            else:
                plt.title('{0}'.format(common[c]), fontsize=16)
            plt.xlabel('Target datapts', fontsize=16)
            plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(percs)])
            plt.semilogx()
            plt.xticks([np.min(percs), np.max(percs)/4, np.max(percs)/2, np.max(percs)],
                       np.array([np.min(percs), np.max(percs)/4, np.max(percs)/2,  np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
        fcnt += 1
    plt.legend(['SC3',
                'SC3-Mix    (upper and lower bound)',
                'SC3-Reject (upper and lower bound)'], fontsize=12, loc=3)
    plt.show()


def plot_rejection_percentage(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common

    color = ['green', 'red', 'blue']
    plt.figure(fig_num)
    inds = [1, 2, 3]
    fcnt = 1
    for c in inds:
        cnt = 1
        for i in range(3):
            ind_common = c
            # ari overall
            ari_baseline = np.mean(res[ind_src, ind_genes, ind_common, 10+i, :, :, 0], axis=0)
            ari_max = np.mean(res[ind_src, ind_genes, ind_common, 10+i, :, :, cnt], axis=0)
            ari_min = np.mean(res[ind_src, ind_genes, ind_common, 10+i, :, :, cnt+1], axis=0)

            plt.subplot(1, len(inds), fcnt)
            if i == 0:
                plt.plot(percs, ari_baseline, '--k', linewidth=2.0)

            plt.fill_between(percs, ari_max, ari_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if c == 1:
                plt.title('#Common cluster: {0}'.format(common[c]), fontsize=16)
                plt.legend(['SC3', 'Reject 10%', 'Reject 20%', 'Reject 30%'], fontsize=12, loc=3)
            else:
                plt.title('{0}'.format(common[c]), fontsize=16)
            plt.xlabel('Target datapts', fontsize=16)
            plt.ylabel('ARI', fontsize=16)
            plt.xlim([0, np.max(percs)])
            plt.semilogx()
            plt.xticks([np.min(percs), np.max(percs)/4, np.max(percs)/2, np.max(percs)],
                       np.array([np.min(percs), np.max(percs)/4, np.max(percs)/2,  np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
        fcnt += 1
    plt.show()


def plot_rejection_aucs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common

    color = ['green', 'red', 'blue']
    plt.figure(fig_num)
    inds = [0, 1, 2, 3, 4]
    fcnt = 1
    for c in inds:
        cnt = 1
        for i in range(3):
            ind_common = c
            auc_max = np.mean(res[ind_src, ind_genes, ind_common, 7+i, :, :, cnt], axis=0)

            plt.subplot(1, len(inds), fcnt)
            plt.plot(percs, auc_max, alpha=0.8, color=color[i], linewidth=2.)
            plt.xlabel('Target datapts', fontsize=16)
            plt.ylabel('AUC', fontsize=16)
            plt.xlim([0, np.max(percs)])
            plt.semilogx()
            plt.title('{0}'.format(common[c]), fontsize=16)
            if c == 0:
                plt.title('#Common cluster: {0}'.format(common[c]), fontsize=16)
                plt.legend(['Reconstr. Error', 'Entropy', 'Kurtosis'], fontsize=12, loc=3)
            plt.xticks([np.min(percs), np.max(percs)/4, np.max(percs)/2, np.max(percs)],
                       np.array([np.min(percs), np.max(percs)/4, np.max(percs)/2,  np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
        fcnt += 1

    plt.show()


def plot_src_accs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 2
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common

    color = ['green', 'red', 'blue']
    plt.figure(fig_num)
    inds = [0, 1, 2, 3, 4]
    fcnt = 1
    for c in inds:
        cnt = 1
        ind_common = c
        aris = np.mean(res[:, :, ind_common, 2, :, 0, 0], axis=2)

        plt.subplot(1, len(inds), fcnt)
        plt.pcolor(aris, cmap=plt.get_cmap('Reds'))
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
    # foo = np.load('intermediate.npz')
    foo = np.load('main_v2.npz')
    # foo = np.load('test_v4.npz')

    # methods = foo['methods']
    # acc_funcs = foo['acc_funcs']
    res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    accs_desc = foo['accs_desc']
    method_desc = foo['method_desc']
    percs = foo['percs']
    # reps = foo['reps']
    genes = foo['genes']
    n_src = foo['n_src']
    n_trg = foo['n_trg']
    common = foo['common']
    print 'n_src x genes x common x acc_funcs x reps x percs x methods'
    print 'Result dimensionality: ', res.shape

    # plot_percs(1, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_overlapping_cluster(2, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # plot_reject_mix(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_rejection_percentage(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_rejection_aucs(5, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    plot_src_accs(6, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    print('Done')