import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy import stats
# from random import randint


def plot_main_opt_mix(fig_num, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes):
    # Indices of the mixing parameters to plot:
    # indices = range(3, len(m_desc)-1)
    ind_common = -1

    # Other indices
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    # ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0)
    # print ari_1_baseline, ari_2_baseline
    # print accs_desc

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0, ddof=0)

    # Plot with errorbars
    markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    # plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    # plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)


    ari = np.mean(res_opt_mix_aris[ind_src, ind_genes, ind_common, :, :], axis=0)
    ste = stats.sem(res_opt_mix_aris[ind_src, ind_genes, ind_common, :, :], axis=0, ddof=0)
    markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)

    plt.title('ARI for {0} src datapts, {1} genes, {2} overlapping top nodes, KTA mixed optimal parameter'.format(n_src[ind_src], genes[ind_genes], common[ind_common]),
              fontsize=16)
    # plt.title('ARI for 1000 src datapts, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    plt.xticks(percs, np.array(percs * n_trg, dtype=np.int))
    plt.ylim([0.0, 1.0])

    plt.legend(['SC3', 'SC3-Comb', 'SC3-Transfer'], fontsize=12, loc=4)
    plt.show()


def plot_mixture_all(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common, mixes):
    # Indices of the mixing parameters to plot:
    #indices = range(3, len(m_desc)-1)
    indices = [2,3,5,7,9,11]
    ind_common  = -1

    # Other indices
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    #ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0)
    # print ari_1_baseline, ari_2_baseline
    # print accs_desc

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0, ddof=0)

    # Plot with errorbars
    markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    #plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    #plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    cmap = plt.cm.get_cmap('hsv', len(indices)+1)

    count = 0
    for ind in indices:
        ari = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, ind], axis=0)
        ste = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, ind], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(percs, ari, color=cmap(count), yerr=ste, linewidth=2.0)
        [bar.set_alpha(0.3) for bar in bars]
        [cap.set_alpha(0.3) for cap in caps]
        #plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
        count += 1

    plt.title('ARI for {0} src datapts, {1} genes, {2} overlapping top nodes, various constant mixture parameters'.format(n_src[ind_src], genes[ind_genes], common[ind_common]), fontsize=16)
    #plt.title('ARI for 1000 src datapts, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    plt.xticks(percs, np.array(percs*n_trg, dtype=np.int))

    plt.ylim([0.0, 1.0])
    indices_now = [x - 2 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Transfer with mix=" + mixes_legend[i]
    # for i, mixes_legend in enumerate(mixes_legend):
    #    mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)
    plt.show()


def plot_percs_optmix(fig_num,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes):
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    fcnt = 1
    # common = [0,1,2,3]
    for ind_common in range(len(common)):
        # common_now = common[ind_common]
        # print ind_common
        # ari overall
        ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0)
        ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0)
        plt.subplot(1, len(common), fcnt)
        # plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
        # plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

        # Standard errors
        ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0, ddof=0)
        ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0, ddof=0)

        # Plot with errorbars
        markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]
        markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]

        # Plot our method

        ari = np.mean(res_opt_mix_aris[ind_src, ind_genes, ind_common, :, :], axis=0)
        ste = stats.sem(res_opt_mix_aris[ind_src, ind_genes, ind_common, :, :], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]
        # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
        plt.title('{0} common top nodes,  \n {1} excl. top nodes in trg, \n {2} excl. top nodes in src'.format(common[fcnt - 1], np.int(
            np.floor(np.true_divide(5 - common[fcnt - 1], 2))), np.int(np.ceil(np.true_divide(5 - common[fcnt - 1], 2)))), fontsize=12)
        if ind_common == 0:
            plt.ylabel('ARI', fontsize=16)

        plt.xlabel('Target datapts', fontsize=16)
        plt.xlim([np.min(percs), np.max(percs)])
        # plt.semilogx()
        # plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))
        plt.xticks(percs[::2], np.array(percs[::2] * n_trg, dtype=np.int))

        plt.ylim([0., 1.])

        fcnt += 1

    plt.legend(['SC3', 'SC3-Comb', 'SC3-Transfer'], fontsize=12, loc=4)
    plt.show()


def plot_percs_new(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    indices =[3,5,7,9,11]
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    fcnt = 1
    #common = [0,1,2,3]
    for ind_common in range(len(common)):
        #common_now = common[ind_common]
        #print ind_common
        # ari overall
        ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0)
        ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0)
        plt.subplot(1, len(common), fcnt)
        #plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
        #plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

        # Standard errors
        ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0, ddof=0)
        ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0, ddof=0)

        # Plot with errorbars
        markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]
        markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]

        # Plot our method
        cmap = plt.cm.get_cmap('hsv', len(indices) + 1)
        count = 0
        for ind in indices:
            ari = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, ind], axis=0)
            ste = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, ind], axis=0, ddof=0)
            markers, caps, bars = plt.errorbar(percs, ari, color=cmap(count), yerr=ste, linewidth=2.0)
            [bar.set_alpha(0.1) for bar in bars]
            [cap.set_alpha(0.1) for cap in caps]
            # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
            count += 1
        plt.title('{0} common top nodes,  \n {1} excl. top nodes in trg, \n {2} excl. top nodes in src'.format(common[fcnt - 1],
                  np.int(np.floor(np.true_divide(5 - common[fcnt - 1], 2))), np.int(np.ceil(np.true_divide(5 - common[fcnt - 1], 2)))), fontsize=12)

        if ind_common == 0:
            plt.ylabel('ARI', fontsize=16)


        plt.xlabel('Target datapts', fontsize=16)
        plt.xlim([np.min(percs), np.max(percs)])
        #plt.semilogx()
        #plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))
        plt.xticks(percs[::2], np.array(percs[::2] * n_trg, dtype=np.int))

        plt.ylim([0., 1.])

        fcnt += 1

    indices_now = [x - 2 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Transfer with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)
    plt.show()


def plot_transferability_new(fig_num, res, res_opt_mix_ind,res_opt_mix_aris,accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)

    #print res.shape
    # aris = np.mean(res[0, :, :, -1, :, -1, 0], axis=2).T

    plt.subplot(1, 2, 1)
    names = []

    common_indices = [0,1,2,3]

    for i in common_indices:
        transf = np.mean(res[0, 0, i, -1, :, :, 0], axis=0)
        ste = stats.sem(res[0, 0, i, -1, :, :, 0], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(percs, transf, yerr=ste, linewidth=2.0, alpha=0.7)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        # plt.plot(percs, aris, linewidth=2., alpha=0.7)
        names.append('{0} common top nodes'.format(common[i]))
    # plt.title('Overlap = 2', fontsize=16)
    #plt.semilogx()
    plt.xlim([np.min(percs), np.max(percs)])
    plt.xticks([np.min(percs), np.max(percs) / 4, np.max(percs) / 2, np.max(percs)],
               np.array(np.true_divide([np.min(percs), np.max(percs) / 4, np.max(percs) / 2, np.max(percs)] * n_trg,n_src)))
    plt.xlabel('Target proportion of source dataset', fontsize=16)
    plt.ylabel('Transferability', fontsize=16)
    plt.ylim([0., 1.])
    plt.legend(names, loc=4, fontsize=14)

    plt.subplot(1,2, 2)
    # n_src x genes x common x acc_funcs x reps x percs x methods
    cols = ['b', 'g', 'y', 'r', 'm', 'c', 'k', 'w']
    markers = ['o', '^', '<', 's', 'v', 'D', 'X', '*']
    for i in common_indices:
        aris = np.mean(res_opt_mix_aris[0, 0, i, :, :], axis=0)
        transf = np.mean(res[0, 0, i, -1, :, :, 0], axis=0)
        plt.scatter(transf, aris, 20, cols[i],marker=markers[i], alpha=0.7)

    plt.legend(names, loc=4)
    plt.plot([0, 1], [0, 1], '--k')
    plt.grid('on')
    plt.xlabel('Transferability', fontsize=16)
    plt.ylabel('ARI', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.text(0.0, -0.1, '*Each point represents one size of targetdata, KTA mixed optimal mixture parameter*', fontsize=12)

    plt.show()


def plot_mixtures_vs_rates(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common, mixes):
    plt.figure(fig_num)
    ind_common = [0,1, 2,3]
    fcnt = 1
    accs_indices = [0,1]
    perc_ind = 4
    for c in range(len(ind_common)):
        plt.subplot(1, len(ind_common), fcnt)
        cmap = plt.cm.get_cmap('hsv', len(accs_indices) + 1)
        count = 0

        for a in range(len(accs_indices)):
            aris = np.mean(res[0, 0, ind_common[c], accs_indices[a], :, perc_ind, 2:], axis=0)
            ste = stats.sem(res[0, 0, ind_common[c], accs_indices[a], :, perc_ind, 2:], axis=0, ddof=0)
            markers, caps, bars = plt.errorbar(mixes, aris, color=cmap(count), yerr=ste, linewidth=2.0)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            plt.plot(mixes[aris.tolist().index(max(aris))], max(aris), 'o', color=cmap(count), label='_nolegend_')
            count += 1

        plt.title('{0}'.format(ind_common[c]), fontsize=16)
        #plt.xticks(np.arange(aris.shape[1]) + 0.5, legend, rotation=80, fontsize=10)
        #plt.yticks(np.arange(len(common)) + 0.5, common)

        if c == 0:
            plt.title('# of overlapping top nodes: {0}'.format(common[ind_common[c]]), fontsize=16)
            plt.xlabel('Mixture parameter', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.text(1, 6.5, 'True cluster accuracy rates (ARI) vs. unsupervised accuracy measures (KTA score)', fontsize=20)
            plt.text(0.3, -0.1, '*{0} source and {1} target datapoints, {2} genes*'.format(n_src[0], percs[perc_ind]*n_trg, genes[0]), fontsize=12)
            legend = accs_desc[accs_indices]
            plt.legend(legend, fontsize=12, loc=2)
        else:
            plt.title('{0}'.format(common[ind_common[fcnt-1]]), fontsize=16)
        # if c == len(ind_common)-1:

        fcnt += 1
        plt.ylim([0., 1.])
        plt.xlim([0.3,0.9])
        plt.xticks(mixes, mixes, fontsize=10)
    plt.show()


# Not used anymore...

def plot_mixtures_vs_rates_mixed_and_original(fig_num, res, res_mixed, accs_desc, m_desc, percs, genes, n_src, n_trg, common, mixes):
    plt.figure(fig_num)
    ind_common = [0,1,  2]
    fcnt = 1
    for c in range(len(ind_common)):
        plt.subplot(1, len(ind_common), fcnt)
        cmap = plt.cm.get_cmap('hsv', len(accs_desc) + 1)
        count = 0

        for a in range(len(accs_desc)):
            aris = np.mean(res[0, 0, ind_common[c], a, :, -1, :], axis=0)
            ste = stats.sem(res[0, 0, ind_common[c], a, :, -1, :], axis=0, ddof=0)
            markers, caps, bars = plt.errorbar(mixes, aris, color=cmap(count), yerr=ste, linewidth=2.0)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            plt.plot(mixes[aris.tolist().index(max(aris))], max(aris), 'o', color=cmap(count),label='_nolegend_')
            if a > 0:
                aris_mixed = np.mean(res_mixed[0, 0, ind_common[c], a, :, -1, :], axis=0)
                ste_mixed = stats.sem(res_mixed[0, 0, ind_common[c], a, :, -1, :], axis=0, ddof=0)
                markers, caps, bars = plt.errorbar(mixes, aris_mixed, color=cmap(count), yerr=ste_mixed, linewidth=2.0, fmt='--')
                [bar.set_alpha(0.5) for bar in bars]
                [cap.set_alpha(0.5) for cap in caps]
                plt.plot(mixes[aris_mixed.tolist().index(max(aris_mixed))], max(aris_mixed), 'o', color=cmap(count),label='_nolegend_')
            count += 1

        plt.title('{0}'.format(ind_common[c]), fontsize=16)
        # plt.xticks(np.arange(aris.shape[1]) + 0.5, legend, rotation=80, fontsize=10)
        # plt.yticks(np.arange(len(common)) + 0.5, common)

        if c == 0:
            plt.title('# of overlapping Clusters: {0}'.format(common[ind_common[c]]), fontsize=16)
            plt.xlabel('Mixture parameter', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.text(1, 6.5, 'True cluster accuracy rates (ARI) vs. unsupervised accuracy measures (KTA and Silhouette coefficients)', fontsize=20)
            plt.text(0.3, -0.1, '*{0} source and {1} target datapoints, {2} genes*'.format(n_src[0], n_trg, genes[0]), fontsize=12)
            legend = ['ARI', 'Silhouette euclidean', 'Silhouette euclidean mixed', 'Silhouette pearson', 'Silhouette pearson mixed', 'Silhouette spearman', 'Silhouette spearman mixed', 'KTA linear', 'KTA linear mixed']
            plt.legend(legend, fontsize=10, loc=1)
        else:
            plt.title('{0}'.format(common[ind_common[fcnt - 1]]), fontsize=16)
        # if c == len(ind_common)-1:

        fcnt += 1
        plt.ylim([0., 1.])
        plt.xlim([0.3, 0.9])
        plt.xticks(mixes, mixes, fontsize=10)
    plt.show()


def plot_ari_vs_unsupervised(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common, mixes):

    common_inds = [0,1,2]
    indices = [0,1,2,3,4,5,6,7,8,9,10] # method indices
    indices_unsup = [1,2,3,4]

    plt.figure(fig_num)
    fcnt = 1
    # n_src x genes x common x acc_funcs x reps x percs x methods
    cols = plt.cm.get_cmap('hsv', len(indices))
    #cols = ['b', 'r', 'g', 'b', 'm', 'c', 'k', 'w']#cols = ['b', 'g', 'y', 'r', 'm', 'c', 'k', 'w']
    markers = ['o', '^', '<', 's', 'v', 'D', '>', '*', '1','2', '3']
    for ind_unsup in range(len(indices_unsup)):
        cnt2 = 0
        for ind_common in range(len(common_inds)):
            plt.subplot(len(indices_unsup), len(common_inds), fcnt)
            aris = np.zeros(len(indices))
            unsuperv = np.zeros(len(indices))
            for i in indices:
                aris[i] = np.mean(res[0, 0, common_inds[ind_common], 0, :, -1, i], axis=0)
                unsuperv[i] = np.mean(res[0, 0, common_inds[ind_common], indices_unsup[ind_unsup], :, -1, i], axis=0)
                plt.scatter(aris[i], unsuperv[i], 20, cols(i), marker=markers[i], alpha=0.7)
            #plt.plot(aris, unsuperv, linewidth=2.0)

            #plt.plot([0, 1], [0, 1], '--k')
            #plt.grid('on')
            if ind_unsup == len(indices_unsup)-1:
                plt.xlabel('ARI', fontsize=16)
                #plt.text(0.2, -0.2, '*Each point represents one size of targetdata*', fontsize=12)

            if ind_common == 0:
                plt.ylabel(accs_desc[indices_unsup[ind_unsup]], fontsize=12)
            #plt.xlim([0, 1])
            #plt.ylim([0, 1])

            if ind_unsup==0:
                plt.title('{0} common clusters,  \n {1} excl. clusters in trg, \n {2} excl. clusters in src'.format(common[cnt2], np.int(np.floor(np.true_divide(5 - common[cnt2], 2))),np.int(np.ceil(np.true_divide(5 - common[cnt2], 2))), fontsize=12))
            fcnt += 1
            cnt2 += 1
    ##indices_now = [x-2 for x in indices]
    indices_now = indices
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    #legend = np.concatenate((['SC3', 'SC3-Comb'], mixes_legend))
    legend = mixes_legend
    plt.legend(legend, fontsize=8, loc=(1.04,0))
    plt.show()


def plot_unsupervised_measures(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    fcnt = 1
    indices = [0,1,2, 3, 4,5,6,7, 8,9,10]     # our method indices to plot
    indices_all =  [0,1,2, 3, 4,5,6,7, 8,9,10]     # baseline and our method indices to plot
    common_indices = [0,1, 2]

    #indices_now = [x - 2 for x in indices]
    indices_now = indices
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Mix " + mixes_legend[i]
    #legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    legend=mixes_legend
    # res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    for c in range(len(accs_desc)):
        aris = np.mean(res[0, 0, :, c, :, -1,:], axis=1)
        plt.subplot(1, len(accs_desc), fcnt)
        aris_now = aris[common_indices, :]
        plt.pcolor(aris_now[:,indices_all], cmap=plt.get_cmap('Greens'))
        # plt.pcolor(aris, cmap=plt.get_cmap('Greens'), vmin=0., vmax=1.)
        #print accs_desc[c]
        plt.title('{0}'.format(accs_desc[c]), fontsize=16)
        plt.xticks(np.arange(len(indices_all)) + 0.5, legend, rotation=80, fontsize = 10)
        plt.yticks(np.arange(len(common_indices)) + 0.5, common[common_indices])
        plt.xlim(0,len(indices_all))
        if c == 0:
            plt.ylabel('#Common cluster', fontsize=16)
            plt.text(1,6.5,'True cluster accuracy rates (ARI) vs. unsupervised accuracy measures (KTA and Silhouette coefficients)', fontsize= 20)
            plt.text(-4,2, '*1000 source and 800 target datapoints, 1000 genes*', fontsize=12, rotation=90)

        fcnt += 1
        #plt.colorbar(ticks=[-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01])
        plt.colorbar()
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
            ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
            ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 1], axis=0)
            ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0)
            ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 3], axis=0)

            plt.subplot(1, len(common), fcnt)
            if i == 0:
                plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
                plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)
            plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor=color[i], interpolate=True)
            if ind_common == 0:
                plt.title('# of overlapping Clusters: {0}'.format(common[ind_common]), fontsize=16)
                plt.xlabel('Target datapts', fontsize=16)
                plt.ylabel('ARI', fontsize=16)
            else:
                plt.title('{0}'.format(common[ind_common]), fontsize=16)

            plt.xlim([np.min(percs), np.max(percs)])
            plt.semilogx()
            plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))

            plt.ylim([0., 1.])

        fcnt += 1
    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=4)
    plt.show()


def plot_overlapping_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    #ind_percs = len(percs)-1

    color = ['blue']
    plt.figure(fig_num)
    # ind_percs = np.arange(len(percs))
    ind_percs = [0, 6, 11]
    ind_methods = [1]
    for p in range(len(ind_percs)):
        for i in range(len(ind_methods)):

            # ari stratified
            ari_0_baseline = np.mean(res[ind_src, ind_genes, 1:, 0, :, ind_percs[p], 0], axis=1)
            ari_0_max      = np.mean(res[ind_src, ind_genes, 1:, 0, :, ind_percs[p], ind_methods[i]+1], axis=1)
            ari_0_min      = np.mean(res[ind_src, ind_genes, 1:, 0, :, ind_percs[p], ind_methods[i]+2], axis=1)
            # ari overall
            ari_1_baseline = np.mean(res[ind_src, ind_genes,  :, 1, :, ind_percs[p], 0], axis=1)
            ari_1_max      = np.mean(res[ind_src, ind_genes,  :, 1, :, ind_percs[p], ind_methods[i]+1], axis=1)
            ari_1_min      = np.mean(res[ind_src, ind_genes,  :, 1, :, ind_percs[p], ind_methods[i]+2], axis=1)

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
                plt.ylabel('stratified ARI', fontsize=16)
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
                'SC3-Mix'], fontsize=12, loc=4)
    plt.show()


def plot_overlapping_cluster_new(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    indices = [4, 7, 10, 13, 14]
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    fcnt = 1
    common = 1
    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, common, 1, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, common, 1, :, :, 1], axis=0)
    plt.subplot(2, 1, fcnt)
    plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    cmap = plt.cm.get_cmap('hsv', len(indices) + 1)
    count = 0
    for ind in indices:
        ari = np.mean(res[ind_src, ind_genes, common, 1, :, :, ind], axis=0)
        plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
        count += 1

    plt.title('# of overlapping Clusters: {0}'.format(common), fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    plt.semilogx()
    plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))

    plt.ylim([0., 1.])

    fcnt += 1
    indices_now = [x - 4 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'], mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)




    plt.subplot(2, 1, fcnt)

    # ari stratified
    ari_1_baseline = np.mean(res[ind_src, ind_genes, common, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, common, 0, :, :, 1], axis=0)
    plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    cmap = plt.cm.get_cmap('hsv', len(indices) + 1)
    count = 0
    for ind in indices:
        ari = np.mean(res[ind_src, ind_genes, common, 0, :, :, ind], axis=0)
        plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
        count += 1

    plt.title('# of overlapping Clusters: {0}'.format(common), fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('stratfied ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    plt.semilogx()
    plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))

    plt.ylim([0., 1.])

    fcnt += 1
    indices_now = [x - 4 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'], mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)

    plt.show()


def plot_src_accs(fig_num, res, genes, n_src, n_trg, common):
    # res are source_aris with shape: n_src x genes x common x reps
    plt.figure(fig_num)
    fcnt = 1
    for c in range(len(common)):
        ind_common = c
        aris = np.mean(res[:, :, ind_common, :], axis=2)
        print aris
        plt.subplot(1, len(common), fcnt)
        plt.pcolor(aris, cmap=plt.get_cmap('Reds'), vmin=0.95, vmax=1.)
        plt.title('{0}'.format(common[c]), fontsize=16)
        plt.xticks(np.arange(len(genes))+0.5, genes)
        plt.yticks(np.arange(len(n_src)) + 0.5, n_src)
        if c == 0:
            plt.xlabel('#Genes', fontsize=16)
            plt.ylabel('#Source datapts', fontsize=16)
            plt.title('#Common cluster: {0}'.format(common[c]), fontsize=16)
            plt.text(1,3.2,'Cluster accuracy rates (ARI) of source data', fontsize= 20)
        fcnt += 1
    plt.colorbar(ticks=[0.95, 0.96,0.97,0.98, 0.99, 1.0])

    plt.show()


def plot_transferability(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)

    print res.shape
    aris = np.mean(res[1, :, :, -1, :, -1, 0], axis=2).T
    print aris, aris.shape

    plt.subplot(1, 3, 1)
    plt.pcolor(aris, cmap=plt.get_cmap('Blues'), vmin=0., vmax=1.)
    plt.title('Transferability', fontsize=16)
    plt.xticks(np.arange(len(genes))+0.5, genes)
    plt.yticks(np.arange(len(common))+0.5, common)
    plt.xlabel('#Genes', fontsize=16)
    plt.ylabel('Overlap', fontsize=16)
    plt.colorbar(ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    plt.text(4, 5.3, 'Estimation of transferability', fontsize=20)
    plt.text(-1, 3, '*1000 source datapoints*', fontsize=12, rotation=90)

    plt.subplot(1, 3, 2)
    names = []
    for i in range(res.shape[2]):
        aris = np.mean(res[1, 0, i, -1, :, :, 0], axis=0)

        ste = stats.sem(res[1, 0, i, -1, :, :, 0], axis=0, ddof=0)
        plt.errorbar(percs, aris, yerr=ste, linewidth=2.0, alpha=0.7)
        #plt.plot(percs, aris, linewidth=2., alpha=0.7)
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
        aris   = np.mean(res[1, 0, i,  0, :, :, 2], axis=0)
        transf = np.mean(res[1, 0, i, -1, :, :, 0], axis=0)
        plt.scatter(aris, transf, 20, cols[i], alpha=0.7)

    plt.legend(names, loc=4)
    plt.plot([0, 1],[0, 1],'--k')
    plt.grid('on')
    plt.xlabel('ARI', fontsize=16)
    plt.ylabel('Transferability', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.show()


def plot_cluster_acc_measures(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    plt.figure(fig_num)
    inds = [0,1, 2, 3, 4]
    fcnt = 1
    names = ['ARI', 'Euclidean','Pearson','Spearman','KTA']
    for c in range(len(inds)):
        aris = np.mean(res[0, 0, :, inds[c], :, :, 1], axis=1)
        print np.mean(res[0, 0, :, inds[c], :, :, 1], axis=1)

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


def plot_unsupervised_measures_percs(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 1
    ind_common = 0
    print '#Src   ', n_src, ' ind/# = ', ind_src, '/', n_src[ind_src]
    print '#Genes ', genes, ' ind/# = ', ind_genes, '/', genes[ind_genes]
    print '#Common ', common, ' ind/# = ', ind_common, '/', common[ind_common]

    plt.figure(fig_num)
    accs = [0, 1, 2, 3, 4, 5]
    fcnt = 1
    # res = np.zeros((len(n_src), len(genes), len(common), len(acc_funcs), reps, len(percs), len(methods)))
    for i in accs:
        for ind_common in range(len(common)):
            mix_max = np.mean(res[ind_src, ind_genes, common[ind_common], i, :, :, 2], axis=0)
            mix_min = np.mean(res[ind_src, ind_genes, common[ind_common], i, :, :, 3], axis=0)

            plt.subplot(len(accs), len(common), fcnt)
            plt.fill_between(percs, mix_max, mix_min, alpha=0.2, facecolor='green', interpolate=True)
            if ind_common == 0:
                plt.title('Overlap: {0}'.format(common[ind_common]), fontsize=16)
                plt.xlabel('Target datapts', fontsize=16)
                plt.ylabel(accs_desc[i], fontsize=16)
            else:
                plt.title('{0}'.format(common[ind_common]), fontsize=16)
            plt.xlim([0, np.max(percs)])
            # plt.semilogx()
            # plt.xticks([np.min(percs), np.mean(percs), np.max(percs)],
            #            np.array([np.min(percs), np.mean(percs), np.max(percs)]*n_trg, dtype=np.int))
            plt.ylim([0., 1.])
            fcnt += 1
    plt.legend(['SC3-Dist', 'SC3-Mix'], fontsize=12, loc=3)
    plt.show()


def plot_main(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)

    ind_common = -1

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0)
    # ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 2], axis=0)
    ari_1_last = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, -1], axis=0)


    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0, ddof=0)
    # ste_ari_1_max = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0, ddof=0)
    ste_ari_1_last = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, -1], axis=0, ddof=0)


    # Plot with errorbars
    plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    # plt.errorbar(percs, ari_1_max, fmt='-b', yerr=ste_ari_1_max, linewidth=2.0)
    plt.errorbar(percs, ari_1_last, fmt='-b', yerr=ste_ari_1_last, linewidth=2.0)

    # plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    # plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)
    # plt.plot(percs, ari_1_min, '-b', linewidth=2.0)

    plt.title('ARI for 1000 src datapts, 500 genes, {0} overlapping clusters'.format(common[ind_common]), fontsize=16)
    #plt.title('ARI for 1000 src datapts, 500 genes, 100% overlapping clusters', fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    plt.semilogx()
    plt.xticks(percs, np.array(percs*n_trg, dtype=np.int))

    plt.ylim([0., 1.])
    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix with fix mixture parameter'], fontsize=12, loc=4)
    plt.show()


def plot_mixture_min_max(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)

    ind_common = -1

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 1], axis=0)
    ari_1_max = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 2], axis=0)
    ari_1_min = np.mean(res[ind_src, ind_genes, ind_common, 1, :, :, 3], axis=0)

    # Standard errors
    # ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, -1], axis=0, ddof=0)
    # ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 1, :, :, 0], axis=0, ddof=0)

    # Plot with errorbars
    # plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    # plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    plt.fill_between(percs, ari_1_max, ari_1_min, alpha=0.2, facecolor='blue', interpolate=True)

    plt.title('ARI for 1000 src datapts, 500 genes, {0} overlapping clusters, optimal and worst case mixture parameter'.format(common[ind_common]), fontsize=16)

    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    plt.semilogx()
    plt.xticks(percs, np.array(percs*n_trg, dtype=np.int))

    plt.ylim([0., 1.])
    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=4)
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


def plot_sc3_only(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):

    # Other indices
    ind_genes = 0
    ind_src = 0
    plt.figure(fig_num)
    #ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, ind_genes, -1, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, ind_genes, -1, 0, :, :, 1], axis=0)
    print ari_1_baseline, ari_2_baseline

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, -1, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, -1, 0, :, :, 1], axis=0, ddof=0)

    # Plot with errorbars
    plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    #plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    #plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)

    plt.title('ARI for 1000 src datapts, 500 genes, {0} overlapping clusters'.format(common[-1]), fontsize=16)
    #plt.title('ARI for 1000 src datapts, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    plt.semilogx()
    plt.xticks(percs, np.array(percs*n_trg, dtype=np.int))

    plt.ylim([-0.1, 1.1])

    legend =['SC3', 'SC3-Comb']
    plt.legend(legend, fontsize=12, loc=4)
    plt.show()


if __name__ == "__main__":

    # For Part 1, Figures 1-3
    foo = np.load('C:\Users\Bettina\PycharmProjects2\scRNA_new\scripts\main_results_part1_opt_mixparam_100reps.npz')
    # For Part 2, Figures 4
    #foo = np.load('C:\Users\Bettina\PycharmProjects2\scRNA_new\\results\main_results\main_results_part2_100reps_100trg.npz')

    # For Figures 6-...
    # foo = np.load('final_toy_experiments_part2.npz')
    # For debugging data
    # foo = np.load('toy_experiments_does_it_still_work.npz')

    methods = foo['methods']
    acc_funcs = foo['acc_funcs']
    res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    #res_mixed = foo['res_mixed']
    res_opt_mix_ind = foo['res_opt_mix_ind']
    res_opt_mix_aris = foo['res_opt_mix_aris']
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
    mixes = foo['mixes']
    print 'n_src x genes x common x acc_funcs x reps x percs x methods'
    print 'Result dimensionality: ', res.shape
    print 'n_src x genes x common x reps x percs'
    print 'Result optimal mixture parameter', res_opt_mix_ind.shape
    #  Running

    plot_main_opt_mix(1, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    plot_percs_optmix(2, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    plot_mixture_all(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    plot_percs_new(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    plot_transferability_new(5, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    plot_mixtures_vs_rates(6, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)


    # plot_mixtures_vs_rates(5, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    # plot_mixtures_vs_rates_mixed_and_original(7, res, res_mixed,accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    #plot_ari_vs_unsupervised(8, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    #plot_ari_vs_unsupervised(9, res_mixed, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    # For the plot of ordered eigenvalues, run debugging with one repetition and un-comment the lines 133-149 in debugging and 186-190 in sc3_clustering_impl.py, and maybe some more lines...

    # Main Plot 1 with all-fixed parameters
    # plot_main(1, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Investigation of mixture parameter
    # Plot 2 with min and max
    # plot_mixture_min_max(2, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Plot 3 with various mixture parameters
    # plot_mixture_all(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    # plot_unsupervised_measures(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_unsupervised_measures(4, res_mixed, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Number of overlapping clusters
    # plot_percs(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_percs_new(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # Plot 4 ARI only on overlapping clusters
    # plot_overlapping_cluster(5, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_overlapping_cluster_new(5, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # Second experiment
    # plot_src_accs(6, source_aris, genes, n_src, n_trg, common)
    # plot_unsupervised_measures(7, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_transferability(8, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # I don't know whats going on...
    # plot_cluster_acc_measures(9, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_cluster(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
    # plot_unsupervised_measures_percs(11, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)

    # plot_sc3_only(1, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)
print('Done')