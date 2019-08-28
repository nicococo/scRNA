###################################################
###						###
### Plot script for experiment on generated data ###
### written by Bettina Mieth, Nico Görnitz,     ###
###   Marina Vidovic and Alex Gutteridge        ###
###                                             ###
###################################################

# Please change all directories to yours!

import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_percs_optmix(fig_num,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes):
    ind_genes = 0
    ind_src = 0
    fcnt = 1
    print(common)
    for ind_common in reversed(list(range(len(common)))):
        # Baseline methods (TargetCluster and ConcatenateCluster)
        ari_1_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0)
        ari_2_baseline = np.mean(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0)
        ste_ari_1_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 0], axis=0, ddof=0)
        ste_ari_2_baseline = stats.sem(res[ind_src, ind_genes, ind_common, 0, :, :, 1], axis=0, ddof=0)

        # Plot with errorbars
        plt.subplot(2, 2, fcnt+1)
        markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='c', yerr=ste_ari_1_baseline, linewidth=2.0)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='y', yerr=ste_ari_2_baseline, linewidth=2.0)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]

        # Plot our method (TransferCluster)
        ari = np.mean(res_opt_mix_aris[ind_src, ind_genes, ind_common, :, :], axis=0)
        ste = stats.sem(res_opt_mix_aris[ind_src, ind_genes, ind_common, :, :], axis=0, ddof=0)
        markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        if common[ind_common] == 0:
            plt.title('No overlap', fontsize = 22, x=0.5, y=0.9)
        elif common[ind_common] == 3:
            plt.title('Incomplete overlap', fontsize = 22, x=0.5, y=0.9)
        elif common[ind_common] == 5:
            plt.title('Complete overlap', fontsize = 22, x=0.5, y=0.9)
            plt.ylabel('ARI', fontsize=16)

        plt.xlabel('Target cells', fontsize=16)
        plt.xlim([np.min(percs), np.max(percs)])
        percs_now = np.delete(percs, 1)
        plt.xticks(percs_now, np.array(percs_now * n_trg, dtype=np.int), fontsize=13)
        plt.ylim([0., 1.])
        plt.yticks(fontsize=13)
        fcnt += 1

        plt.legend(['TargetCluster', 'ConcatenateCluster', 'TransferCluster'], fontsize=13, loc=4)


if __name__ == "__main__":

    # Figure direction to save to
    fname_plot ='/home/bmieth/scRNAseq/results/toy_data_final/main_results_toydata_figure_'
    # Data location - please change directory to yours
    foo = np.load('/home/bmieth/scRNAseq/results/toy_data_final/main_results_toydata.npz')

    # Load data
    methods = foo['methods']
    acc_funcs = foo['acc_funcs']
    res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    res_opt_mix_ind = foo['res_opt_mix_ind']
    res_opt_mix_aris = foo['res_opt_mix_aris']
    source_aris = foo['source_aris'] # n_src x genes x common x reps
    accs_desc = foo['accs_desc']
    print(accs_desc)
    method_desc = foo['method_desc']
    percs = foo['percs']
    # reps = foo['reps']
    genes = foo['genes']
    n_src = foo['n_src']
    n_trg = foo['n_trg']
    common = foo['common']
    mixes = foo['mixes']
    print('n_src x genes x common x acc_funcs x reps x percs x methods')
    print('Result dimensionality: ', res.shape)
    print('n_src x genes x common x reps x percs')
    print('Result optimal mixture parameter', res_opt_mix_ind.shape)
    
    #  Plot figure
    fig = plt.figure(figsize=(16,12))
    plot_percs_optmix(1, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, common, mixes)
    plt.savefig(fname_plot+'1'+'.jpg')
print('Done')
