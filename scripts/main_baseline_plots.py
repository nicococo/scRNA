import matplotlib.pyplot as plt
import numpy as np


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


def plot_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    ind_common = 2
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common, ' ind/# = ', ind_common, '/', common[ind_common]
    print percs

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
                'SC3-Dist',
                'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_percs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    ind_common = 0
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common, ' ind/# = ', ind_common, '/', common[ind_common]

    color = ['blue', 'green', 'red']
    plt.figure(fig_num)
    inds = [0, 1, 2, 3, 4]
    # inds = [0]
    fcnt = 1
    # res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    for c in inds:
        cnt = 1
        for i in range(2):
            ind_common = c
            # ari overall
            ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0)
            ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
            ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt], axis=0)
            ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, cnt+1], axis=0)
            cnt += 2

            plt.subplot(1, len(inds), fcnt)
            if i == 0:
                plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
                plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)
            plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if c == 0:
                plt.title('Overlap: {0}'.format(common[c]), fontsize=16)
                plt.xlabel('Target datapts', fontsize=16)
                plt.ylabel('ARI', fontsize=16)
            else:
                plt.title('{0}'.format(common[c]), fontsize=16)
            plt.xlim([0, np.max(percs)])
            plt.semilogx()
            plt.xticks([np.min(percs), np.mean(percs), np.max(percs)],
                       np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
        fcnt += 1
    plt.legend(['SC3', 'SC3-Comb',
                'SC3-Dist',
                'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_overlapping_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 2
    ind_src = 0
    ind_percs = 8
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Percs ', percs, ' ind/# = ', ind_percs, '/', percs[ind_percs]

    color = ['blue', 'green', 'red']
    plt.figure(fig_num)
    # ind_percs = [0, 3, 5, 6, 7, 8]
    ind_percs = np.arange(len(percs))
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
                'SC3-Dist',
                'SC3-Mix',
                'SC3-Rejectpfizer'], fontsize=12, loc=3)
    plt.show()


def plot_src_accs(fig_num, res, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    inds = [0, 1, 2, 3, 4]
    print res[0,0,2,:]
    fcnt = 1
    for c in inds:
        ind_common = c
        aris = np.mean(res[:, :, ind_common, :], axis=2)
        print np.mean(res[:, :, ind_common, :], axis=2)

        plt.subplot(1, len(inds), fcnt)
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


def plot_unsupervised_measures(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    inds = [0, 1, 5, 6, 7, 8]
    fcnt = 1
    # res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    for c in inds:
        aris = np.mean(res[0, 0, :, c, :, -1, :], axis=1)
        print np.mean(res[0, 0, :, c, :, -1, :], axis=1)

        plt.subplot(1, len(inds), fcnt)
        plt.pcolor(aris, cmap=plt.get_cmap('Greens'))
        # plt.pcolor(aris, cmap=plt.get_cmap('Greens'), vmin=0., vmax=1.)
        print accs_desc[c]
        plt.title('{0}'.format(accs_desc[c]), fontsize=16)
        plt.xticks(np.arange(aris.shape[1])+0.5, ['SC3', 'Dist max', 'Dist min', 'Mix max', 'Mix min'], rotation=60)
        plt.yticks(np.arange(len(common))+0.5, common)
        if c == 0:
            plt.ylabel('#Common cluster', fontsize=16)
        fcnt += 1
        plt.colorbar(ticks=[-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01])
    plt.show()


if __name__ == "__main__":
    foo = np.load('results/main_short_v1.npz')
    # foo = np.load('results/main_result_v1.npz')

    # methods = foo['methods']
    # acc_funcs = foo['acc_funcs']
    res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    source_aris = foo['source_aris'] # n_src x genes x common x reps
    accs_desc = foo['accs_desc']
    print accs_desc
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
    plot_overlapping_cluster(2, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # plot_src_accs(6, source_aris, genes, n_src, n_trg, common)

    # plot_transferability(7, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # plot_unsupervised_measures(8, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # plot_cluster(9, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_cluster_acc_measures(10, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)


    print('Done')