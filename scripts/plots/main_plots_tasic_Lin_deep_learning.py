import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# from random import randint
from scipy import stats


def plot_main_opt_mix(fig_num, res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting):
    # Indices of the mixing parameters to plot:
    # indices = range(3, len(m_desc)-1)

    # Other indices
    ind_src = 0
    plt.figure(fig_num)
    # ind_common = common[-1]

    # ari overall
    ari_1_baseline = np.mean(res[ind_src, 0, :, :, 0], axis=0)
    ari_2_baseline = np.mean(res[ind_src, 0, :, :, 1], axis=0)
    ari_3_baseline = np.mean(res[ind_src, 0, :, :, 2], axis=0)
    # print ari_1_baseline, ari_2_baseline
    # print accs_desc

    # Standard errors
    ste_ari_1_baseline = stats.sem(res[ind_src, 0, :, :, 0], axis=0, ddof=0)
    ste_ari_2_baseline = stats.sem(res[ind_src, 0, :, :, 1], axis=0, ddof=0)
    ste_ari_3_baseline = stats.sem(res[ind_src, 0, :, :, 2], axis=0, ddof=0)

    # Plot with errorbars
    markers, caps, bars = plt.errorbar(percs, ari_1_baseline, fmt='c', yerr=ste_ari_1_baseline, linewidth=2.0)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_2_baseline, fmt='y', yerr=ste_ari_2_baseline, linewidth=2.0)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    markers, caps, bars = plt.errorbar(percs, ari_3_baseline, fmt='r', yerr=ste_ari_3_baseline, linewidth=2.0)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    # plt.plot(percs, ari_1_baseline, '--k', linewidth=2.0)
    # plt.plot(percs, ari_2_baseline, '-.k', linewidth=2.0)


    ari = np.mean(res_opt_mix_aris[ind_src, :, :], axis=0)
    ste = stats.sem(res_opt_mix_aris[ind_src, :, :], axis=0, ddof=0)
    markers, caps, bars = plt.errorbar(percs, ari, fmt='-b', yerr=ste, linewidth=2.0)

    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    # plt.plot(percs, ari, color=cmap(count), linewidth=2.0)
    if overlap_setting == 0:
        plt.title('Complete overlap', fontsize=22, x=0.5, y=0.93)
    elif overlap_setting == 1:
        plt.title('Incomplete overlap', fontsize=22, x=0.5, y=0.93)	
    else:
        plt.title('Subsampled Tasic data (no overlap)', fontsize=22, x=0.5, y=0.93)	

    plt.text( x=0.3, y=0.88, s='Ground truth labels from NMF', fontsize= 12)
    #plt.title('ARI for {0} src datapoints, {1} genes, KTA mixed optimal parameter'.format(n_src[ind_src], genes),fontsize=16)
    # plt.title('ARI for 1000 src datapoints, 500 genes, 100% overlapping clusters', fontsize=16)

    plt.xlabel('Target cells', fontsize=16)
    plt.ylabel('ARI', fontsize=16)

    plt.xlim([np.min(percs), np.max(percs)])
    #plt.semilogx()
    plt.xticks(percs, np.array(percs * n_trg, dtype=np.int), fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylim([0.0, 1.0])

    plt.legend(['TargetCluster', 'ConcatenateCluster', 'Deep learning Lin', 'TransferCluster'], fontsize=13, loc=4)
    #plt.show()


if __name__ == "__main__":

    fname_plot ='/home/bmieth/scRNAseq/results/mouse_data_NN/main_results_mouse_nooverlap_NMF_complete'
    #foo_com = np.load('/home/bmieth/scRNAseq/results/mouse_data_NN/mouse_completeoverlap_NN_rate001_100dims.npz')
    #foo_incom = np.load('/home/bmieth/scRNAseq/results/mouse_data_NN/mouse_incompleteoverlap_NN_rate001.npz')
    foo_no = np.load('/home/bmieth/scRNAseq/results/mouse_data_NN/mouse_nooverlap_NMF_complete.npz')

    #res = foo_com['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    #res_opt_mix_ind = foo_com['res_opt_mix_ind']
    #res_opt_mix_aris = foo_com['res_opt_mix_aris']
    #accs_desc = foo_com['accs_desc']
    #method_desc = foo_com['method_desc']
    #percs = foo_com['percs']
    #genes = foo_com['genes']
    #n_src = foo_com['n_src']
    #n_trg = foo_com['n_trg']
    #mixes = foo_com['mixes']
    
    #  Running
    fig = plt.figure(figsize=(10,8))
    #plt.subplot(1,3,1)
    #plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting = 0)
	
    #res = foo_incom['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    #res_opt_mix_ind = foo_incom['res_opt_mix_ind']
    #res_opt_mix_aris = foo_incom['res_opt_mix_aris']
    #accs_desc = foo_incom['accs_desc']
    #method_desc = foo_incom['method_desc']
    #percs = foo_incom['percs']
    #genes = foo_incom['genes']
    #n_src = foo_incom['n_src']
    #n_trg = foo_incom['n_trg']
    #mixes = foo_incom['mixes']

    #plt.subplot(1,3,2)
    #plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting = 1)
	
    res = foo_no['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    res_opt_mix_ind = foo_no['res_opt_mix_ind']
    res_opt_mix_aris = foo_no['res_opt_mix_aris']
    accs_desc = foo_no['accs_desc']
    method_desc = foo_no['method_desc']
    percs = foo_no['percs']
    genes = foo_no['genes']
    n_src = foo_no['n_src']
    n_trg = foo_no['n_trg']
    mixes = foo_no['mixes']
    print(foo_no['splitting_mode'])
    print('Training Accuracy: ', np.mean(foo_no['training_acc']))
    print('Training Loss: ', np.mean(foo_no['training_loss']))
    print('Validation Accuracy: ', np.mean(foo_no['validation_acc']))
    print('Validation Loss: ', np.mean(foo_no['validation_loss']))

    #plt.subplot(1,3,3)
    plot_main_opt_mix(1,res, res_opt_mix_ind,res_opt_mix_aris, accs_desc, method_desc, percs, genes, n_src, n_trg, mixes, overlap_setting = 2)
	
	
    plt.savefig(fname_plot+'.jpg')



print('Done')
