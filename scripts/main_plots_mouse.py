import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy import stats
import pandas as pd

# from random import randint


def plot_main_opt_mix(fig_num, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes):
    # Indices of the mixing parameters to plot:
    # indices = range(3, len(m_desc)-1)

    # Other indices
    ind_src = 0
    plt.figure(fig_num)
    # ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, 0, :, :, 1], axis=0)
    # print ari_1_baseline, ari_2_baseline
    # print accs_desc

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, 0, :, :, 1], axis=0, ddof=0)

    # Plot with errorbars
    markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    # plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    # plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)


    ari = np.mean(res_opt_mix_aris[ind_src, :, :], axis=0)
    ste = stats.sem(res_opt_mix_aris[ind_src, :, :], axis=0, ddof=0)
    markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
    plt.title('ARI for {0} src datapts, {1} genes, KTA mixed optimal parameter'.format(n_src[ind_src], genes),
              fontsize=16)
    # plt.title('ARI for 1000 src datapts, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target datapts', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    plt.xticks(percs, np.array(percs * n_trg, dtype=np.int))
    plt.ylim([0.0, 1.0])

    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=4)
    plt.show()


def plot_mixture_all(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, mixes):
    # Indices of the mixing parameters to plot:
    #indices = range(3, len(m_desc)-1)
    indices = [2,3]

    # Other indices
    ind_src = 0
    plt.figure(fig_num)
    #ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, 0, :, :, 1], axis=0)
    # print ari_1_baseline, ari_2_baseline
    # print accs_desc

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, 0, :, :, 1], axis=0, ddof=0)

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
        ari = np.mean(res[ind_src, 0, :, :, ind], axis=0)
        ste = stats.sem(res[ind_src, 0, :, :, ind], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(percs, ari, color=cmap(count), yerr=ste, linewidth=2.0)
        [bar.set_alpha(0.3) for bar in bars]
        [cap.set_alpha(0.3) for cap in caps]
        #plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
        count += 1

    plt.title('ARI for {0} src datapts, {1} genes, various constant mixture parameters'.format(n_src[ind_src], genes), fontsize=16)
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
        mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    # for i, mixes_legend in enumerate(mixes_legend):
    #    mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)
    plt.show()


def plot_percs_optmix(fig_num,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes):
    ind_src = 0
    plt.figure(fig_num)
    # common_now = common[ind_common]
    # print ind_common
    # ari overall
    ari_1_baseline = np.mean(res[ind_src, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, 0, :, :, 1], axis=0)

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, 0, :, :, 1], axis=0, ddof=0)

    # Plot with errorbars
    markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='--k', yerr=ste_ari_1_baseline, linewidth=2.0)
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='-.g', yerr=ste_ari_2_baseline, linewidth=2.0)
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]

    # Plot our method

    ari = np.mean(res_opt_mix_aris[ind_src, :, :], axis=0)
    ste = stats.sem(res_opt_mix_aris[ind_src, :, :], axis=0, ddof=0)
    markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]
    # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
    plt.ylabel('ARI', fontsize=16)
    plt.xlabel('Target datapts', fontsize=16)
    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    # plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))
    plt.xticks(percs[::2], np.array(percs[::2] * n_trg, dtype=np.int))

    plt.ylim([0., 1.])

    plt.legend(['SC3', 'SC3-Comb', 'SC3-Mix'], fontsize=12, loc=4)
    plt.show()


def plot_percs_new(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, mixes):
    indices = [2,3]
    ind_src = 0
    plt.figure(fig_num)
    # ari overall
    ari_1_baseline = np.mean(res[ind_src, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, 0, :, :, 1], axis=0)

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, 0, :, :, 1], axis=0, ddof=0)

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
        ari = np.mean(res[ind_src, 0, :, :, ind], axis=0)
        ste = stats.sem(res[ind_src, 0, :, :, ind], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(percs, ari, color=cmap(count), yerr=ste, linewidth=2.0)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]
        # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
        count += 1
        plt.ylabel('ARI', fontsize=16)

    plt.xlabel('Target datapts', fontsize=16)
    plt.xlim([np.min(percs), np.max(percs)])
    #plt.semilogx()
    #plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))
    plt.xticks(percs[::2], np.array(percs[::2] * n_trg, dtype=np.int))

    plt.ylim([0., 1.])


    indices_now = [x - 2 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3 Mix with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)
    plt.show()


def plot_transferability_new(fig_num, res, res_opt_mix_ind,res_opt_mix_aris,accs_desc, m_desc, percs, genes, n_src, n_trg):
    plt.figure(fig_num)

    #print res.shape
    # aris = np.mean(res[0, :, :, -1, :, -1, 0], axis=2).T

    plt.subplot(1, 2, 1)
    names = []
    transf = np.mean(res[0, -1, :, :, 0], axis=0)
    ste = stats.sem(res[0, -1, :, :, 0], axis=0, ddof=0)
    markers, caps, bars = plt.errorbar(percs, transf, yerr=ste, linewidth=2.0, alpha=0.7)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    # plt.plot(percs, aris, linewidth=2., alpha=0.7)

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
    aris = np.mean(res_opt_mix_aris[0,  :, :], axis=0)
    transf = np.mean(res[0, -1, :, :, 0], axis=0)
    plt.scatter(transf, aris, 20, alpha=0.7)

    plt.legend(names, loc=4)
    plt.plot([0, 1], [0, 1], '--k')
    plt.grid('on')
    plt.xlabel('Transferability', fontsize=16)
    plt.ylabel('ARI', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.text(0.0, -0.1, '*Each point represents one size of targetdata, KTA mixed optimal mixture parameter*', fontsize=12)

    plt.show()


def plot_mixtures_vs_rates(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, mixes):
    plt.figure(fig_num)
    fcnt = 1
    accs_indices = [0,1]
    perc_ind = 1
    cmap = plt.cm.get_cmap('hsv', len(accs_indices) + 1)
    count = 0

    for a in range(len(accs_indices)):
        aris = np.mean(res[0, accs_indices[a], :, perc_ind, 2:], axis=0)
        ste = stats.sem(res[0, accs_indices[a], :, perc_ind, 2:], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(mixes, aris, color=cmap(count), yerr=ste, linewidth=2.0)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        plt.plot(mixes[aris.tolist().index(max(aris))], max(aris), 'o', color=cmap(count), label='_nolegend_')
        count += 1

    plt.xlabel('Mixture parameter', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.text(1, 6.5, 'True cluster accuracy rates (ARI) vs. unsupervised accuracy measures (KTA score)', fontsize=20)
    plt.text(0.3, -0.1, '*{0} source and {1} target datapoints, {2} genes*'.format(n_src[0], percs[perc_ind]*n_trg, genes), fontsize=12)
    legend = accs_desc[accs_indices]
    plt.legend(legend, fontsize=12, loc=2)

    plt.ylim([0., 1.])
    plt.xlim([0.3,0.9])
    plt.xticks(mixes, mixes, fontsize=10)
    plt.show()


def plot_expression_histogram(data_matrix):
    pdb.set_trace()
    plt.hist(data_matrix, bins=100)
    plt.show()


if __name__ == "__main__":

    # foo = np.load('C:\Users\Bettina\PycharmProjects2\scRNA_new\scripts\main_results_mouse.npz')
    # methods = foo['methods']
    # acc_funcs = foo['acc_funcs']
    # res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    # #res_mixed = foo['res_mixed']
    # res_opt_mix_ind = foo['res_opt_mix_ind']
    # res_opt_mix_aris = foo['res_opt_mix_aris']
    # source_aris = foo['source_aris'] # n_src x genes x common x reps
    # accs_desc = foo['accs_desc']
    # print accs_desc
    # method_desc = foo['method_desc']
    # percs = foo['percs']
    # # reps = foo['reps']
    # genes = foo['genes']
    # n_src = foo['n_src']
    # n_trg = foo['n_trg']
    # mixes = foo['mixes']
    # print 'n_src x genes x common x acc_funcs x reps x percs x methods'
    # print 'Result dimensionality: ', res.shape
    # print 'n_src x genes x common x reps x percs'
    # print 'Result optimal mixture parameter', res_opt_mix_ind.shape
    # #  Running
    #
    # plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    # plot_mixture_all(2, res, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    # plot_percs_optmix(3,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    # plot_percs_new(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    # plot_transferability_new(5, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg)
    # plot_mixtures_vs_rates(6, res, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)

    fname_data = 'C:\Users\Bettina\PycharmProjects2\scRNA_new\data\mouse\mouse_vis_cortex\matrix'
    # Careful, this takes too long..
    data = pd.read_csv(fname_data, sep='\t').values
    plot_expression_histogram(data)

print('Done')
