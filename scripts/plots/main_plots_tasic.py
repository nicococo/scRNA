import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from random import randint
from scipy import stats


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
    plt.title('Real data, 18 clusters, complete overlap between source (1000 cells) and target data', fontsize=16)
    #plt.title('ARI for {0} src datapoints, {1} genes, KTA mixed optimal parameter'.format(n_src[ind_src], genes),fontsize=16)
    # plt.title('ARI for 1000 src datapoints, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target datapoints', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    plt.xticks(percs, np.array(percs * n_trg, dtype=np.int))
    plt.ylim([0.0, 1.0])

    plt.legend(['SC3', 'SC3-Comb', 'SC3-Transfer'], fontsize=12, loc=4)
    plt.show()


def plot_mixture_all(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, mixes):
    # Indices of the mixing parameters to plot:
    #indices = range(3, len(m_desc)-1)
    #indices = [2,3,4,5,6,7,8,9,10,11,12]
    indices = list(range(2, len(m_desc)))

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

    plt.title('Transfer with fix mixture parameters - Real data, 18 clusters, complete overlap', fontsize=16)
    #plt.title('ARI for {0} src datapoints, {1} genes, various constant mixture parameters'.format(n_src[ind_src], genes), fontsize=16)
    #plt.title('ARI for 1000 src datapoints, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target datapoints', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    plt.xticks(percs, np.array(percs*n_trg, dtype=np.int))

    plt.ylim([0.0, 1.0])
    indices_now = [x - 2 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3-Transfer with mix=" + mixes_legend[i]
    # for i, mixes_legend in enumerate(mixes_legend):
    #    mixes_legend[i] = "SC3-Transfer with mix=" + mixes_legend[i]
    legend = np.concatenate((['SC3', 'SC3-Comb'],mixes_legend))
    plt.legend(legend, fontsize=12, loc=4)
    plt.show()

	
def plot_mixtures_vs_rates(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, mixes):
    plt.figure(fig_num)
    accs_indices = [0,1]
    perc_inds = list(range(len(percs)))
    cmap = plt.cm.get_cmap('hsv', len(accs_indices) + 1)

    for perc_ind in perc_inds:
        plt.subplot(2, 3, perc_ind+1)
        count = 0
        for a in range(len(accs_indices)):
            aris = np.mean(res[0, accs_indices[a], :, perc_ind, 2:], axis=0)
            ste = stats.sem(res[0, accs_indices[a], :, perc_ind, 2:], axis=0, ddof=0)
            markers, caps, bars = plt.errorbar(mixes, aris, color=cmap(count), yerr=ste, linewidth=2.0)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            plt.plot(mixes[aris.tolist().index(max(aris))], max(aris), 'o', color=cmap(count), label='_nolegend_')
            count += 1
        plt.title('{0} target datapoints'.format(int(percs[perc_ind] * n_trg)), fontsize=16)
        plt.xlabel('Mixture parameter', fontsize=12)
        plt.ylabel('ARI', fontsize=12)
        #plt.text(1, 6.5, 'True cluster accuracy rates (ARI) vs. unsupervised accuracy measures (KTA score)', fontsize=20)
        plt.ylim([0.0, 1.0])
        plt.xlim([np.min(mixes),np.max(mixes)])
        plt.xticks(mixes, mixes, fontsize=10)
        legend = accs_desc[accs_indices]
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
    plt.xlabel('Target datapoints', fontsize=16)
    plt.xlim([np.min(percs), np.max(percs)])
    # plt.semilogx()
    # plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))
    plt.xticks(percs[::2], np.array(percs[::2] * n_trg, dtype=np.int))

    plt.ylim([0., 1.])

    plt.legend(['SC3', 'SC3-Comb', 'SC3-Transfer'], fontsize=12, loc=4)
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

    plt.xlabel('Target datapoints', fontsize=16)
    plt.xlim([np.min(percs), np.max(percs)])
    #plt.semilogx()
    #plt.xticks([np.min(percs), np.mean(percs), np.max(percs)], np.array([np.min(percs), np.mean(percs), np.max(percs)] * n_trg, dtype=np.int))
    plt.xticks(percs[::2], np.array(percs[::2] * n_trg, dtype=np.int))

    plt.ylim([0., 1.])


    indices_now = [x - 2 for x in indices]
    mixes_legend = list(map(str, mixes[indices_now]))
    for i in range(len(mixes_legend)):
        mixes_legend[i] = "SC3-Transfer with mix=" + mixes_legend[i]
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
    #plt.legend(names, loc=4, fontsize=14)

    plt.subplot(1,2, 2)
    # n_src x genes x common x acc_funcs x reps x percs x methods
    cols = ['b', 'g', 'y', 'r', 'm', 'c', 'k', 'w']
    markers = ['o', '^', '<', 's', 'v', 'D', 'X', '*']
    aris = np.mean(res_opt_mix_aris[0,  :, :], axis=0)
    transf = np.mean(res[0, -1, :, :, 0], axis=0)
    plt.scatter(transf, aris, 20, alpha=0.7)

    #plt.legend(names, loc=4)
    plt.plot([0, 1], [0, 1], '--k')
    plt.grid('on')
    plt.xlabel('Transferability', fontsize=16)
    plt.ylabel('ARI', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.text(0.0, -0.1, '*Each point represents one size of targetdata, KTA mixed optimal mixture parameter*', fontsize=12)

    plt.show()


def plot_expression_histogram(data_matrix, num_bins=1000, x_range=(0,800), y_lim=1000000):
    a_size,b_size = data_matrix.shape
    X = np.log10(data_matrix.reshape(a_size * b_size, 1)+1.)
    #X = data_matrix.reshape(a_size * b_size, 1)
    plt.hist(X, range=x_range,bins=num_bins)
    plt.ylim(0,y_lim)
    #plt.xscale('log')
    #plt.title("Histogram of expression values of all {0} genes in all {1} cells (total of {2} values), \n{3} entries equal zero, x- and y-achis are cropped.".format(a_size, b_size, a_size*b_size, np.sum(np.sum(data==0))))
    plt.xlabel("$log_{10}(x_{exp} + 1)$", fontsize=16)
    plt.ylabel("Frequency in thousands", fontsize=16)
    plt.xlim(0,4)
    plt.xticks(fontsize=13)
    plt.yticks([100000, 200000,300000,400000,500000,600000], ['100','200','300','400','500','600'],fontsize=13)
    #plt.show()


if __name__ == "__main__":

    fname_plot ='/home/bmieth/scRNAseq/results/mouse_data_final/mouse_histogram'
    #foo = np.load('/home/bmieth/scRNAseq/results/mouse_data_tryparams/mouse_completeoverlap_loosefilter_high_butabithigher_regularization.npz')

    #methods = foo['methods']
    #acc_funcs = foo['acc_funcs']
    #res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    ##res_mixed = foo['res_mixed']
    #res_opt_mix_ind = foo['res_opt_mix_ind']
    #res_opt_mix_aris = foo['res_opt_mix_aris']
    #source_aris = foo['source_aris'] # n_src x genes x common x reps
    #print 'Source ARIs from NMF clustering: ', source_aris
    #accs_desc = foo['accs_desc']
    #print accs_desc
    #method_desc = foo['method_desc']
    #percs = foo['percs']
    #reps = foo['reps']
    #genes = foo['genes']
    #n_src = foo['n_src']
    #n_trg = foo['n_trg']
    ##print n_trg
    #mixes = foo['mixes']
    #print 'n_src x genes x common x acc_funcs x reps x percs x methods'
    #print 'Result dimensionality: ', res.shape
    #print 'n_src x genes x common x reps x percs'
    #print 'Result optimal mixture parameter', res_opt_mix_ind.shape
    #  Running
    #fig = plt.figure(figsize=(16,12))
    #plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    #plt.savefig(fname_plot+'1'+'.jpg')
    #fig = plt.figure(figsize=(16,12))
    #plot_mixture_all(2, res, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    #plt.savefig(fname_plot+'2'+'.jpg')
    #fig = plt.figure(figsize=(16,12))
    #plot_mixtures_vs_rates(3, res, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    #plt.savefig(fname_plot+'3'+'.jpg')
	
    #plot_percs_optmix(3,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    #plot_percs_new(4, res, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes)
    #plot_transferability_new(3, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg)

    fname_data = '/home/bmieth/scRNAseq/data/matrix'
    data = pd.read_csv(fname_data, sep='\t').values
    fig = plt.figure(figsize=(8,6))
    plot_expression_histogram(data, num_bins=200, x_range=(0,5), y_lim=600000)
    plt.savefig(fname_plot+'.jpg')


print('Done')
