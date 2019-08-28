###################################################
###						###
### Plot script for experiments on Tasic data   ###
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
import pdb
from scipy import stats
import pandas as pd
from scipy import stats


def plot_main_opt_mix(fig_num, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting):
    # Setting up plot
    ind_src = 0
    plt.figure(fig_num)

    # Baseline methods (TargetCluster and ConcatenateCluster)
    ari_1_baseline = np.mean(res[ind_src, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, 0, :, :, 1], axis=0)
    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, 0, :, :, 1], axis=0, ddof=0)

    # Plot with errorbars
    markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='c', yerr=ste_ari_1_baseline, linewidth=2.0)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='y', yerr=ste_ari_2_baseline, linewidth=2.0)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]

    # Plot our method (TransferCluster)
    ari = np.mean(res_opt_mix_aris[ind_src, :, :], axis=0)
    ste = stats.sem(res_opt_mix_aris[ind_src, :, :], axis=0, ddof=0)
    if overlap_setting==0:
        markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)
    else:
        markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)

    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    if overlap_setting == 0:
        plt.title('Complete overlap', fontsize=22, x=0.5, y=0.93)
    else:
        plt.title('Incomplete overlap', fontsize=22, x=0.5, y=0.93)	
    plt.text( x=0.15, y=0.88, s='Ground truth labels from NMF clustering', fontsize= 14)
    plt.xlabel('Target cells', fontsize=16)
    plt.ylabel('ARI', fontsize=16)
    plt.xlim([np.min(percs), np.max(percs)])
    plt.xticks(percs, np.array(percs * n_trg, dtype=np.int), fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim([0.0, 1.0])
    plt.legend(['TargetCluster', 'ConcatenateCluster', 'TransferCluster'], fontsize=13, loc=4)

if __name__ == "__main__":
    # Figure direction to save to
    fname_plot ='/home/bmieth/scRNAseq/results/mouse_data_final/main_results_mouse'
    # Location of experimental results - change to yours
    foo_com = np.load('/home/bmieth/scRNAseq/results/mouse_data_final/main_results_mouse.npz')
    foo_incom = np.load('/home/bmieth/scRNAseq/results/mouse_data_final/main_results_mouse.npz')

    # Load data
    res = foo_com['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    res_opt_mix_ind = foo_com['res_opt_mix_ind']
    res_opt_mix_aris = foo_com['res_opt_mix_aris']
    accs_desc = foo_com['accs_desc']
    method_desc = foo_com['method_desc']
    percs = foo_com['percs']
    genes = foo_com['genes']
    n_src = foo_com['n_src']
    n_trg = foo_com['n_trg']
    mixes = foo_incom['mixes']
    
    #  Plot figure of complete overlap
    fig = plt.figure(figsize=(14,8))
    plt.subplot(1,2,1)
    plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting = 0)
	
    res = foo_incom['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    res_opt_mix_ind = foo_incom['res_opt_mix_ind']
    res_opt_mix_aris = foo_incom['res_opt_mix_aris']
    accs_desc = foo_incom['accs_desc']
    method_desc = foo_incom['method_desc']
    percs = foo_incom['percs']
    genes = foo_incom['genes']
    n_src = foo_incom['n_src']
    n_trg = foo_incom['n_trg']
    mixes = foo_incom['mixes']

    #  Plot figure of incomplete overlap
    plt.subplot(1,2,2)
    plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting = 1)
    plt.savefig(fname_plot+'.jpg')

print('Done')

