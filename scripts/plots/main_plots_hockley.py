###################################################
###						###
### Plot script for experiment on Hockley data  ###
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

	
def plot_tsne(data_matrix, labels, method_string):
    #generate a list of markers and another of colors 
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","." , "," , "o" , "v" , "^" , "<", ">"]
    colors = ['r','g','b','c','m', 'y', 'k', 'k', 'y','m', 'b','c','g','r']
    num_cluster = len(np.unique(labels))
    clusters = np.unique(labels)
    
    model = TSNE(n_components=2, random_state=0, init='pca', metric='euclidean', perplexity=30, method='exact')
    ret = model.fit_transform(data_matrix.T)
    for i in range(num_cluster):
        plt.scatter(ret[labels==clusters[i], 0], ret[labels==clusters[i], 1], 20, marker=markers[i], color = colors[i], label=i)
    plt.title(method_string)	
	

if __name__ == "__main__":
    # Figure direction to save to
    fname_plot ='/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_figure_'
    # Experimental results location - please change directory to yours
    foo_NMF = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_NMFlabels.npz')
    foo_l1 = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_level1labels.npz')
    foo_l2 = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_level2labels.npz')
    foo_l3 = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_level3labels.npz')

    # Data location - please change directory to yours	    
    fname_data_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_mat.tsv'
    fname_data_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_mat.tsv'
    
    ## TSNE plots of results
    # TargetCluster 
    trg_labels = foo_NMF['trg_labels']
    data_target_preprocessed = foo_NMF['data_target']
    fig = plt.figure(figsize=(16,12))
    plt.subplot(2,3,1)
    plot_tsne(data_target_preprocessed, trg_labels[:, 0], method_string = 'TargetCluster')
    
    ## SC3 comb results, ConcatenateCluster
    trg_labels = foo_NMF['trg_labels']
    data_target_preprocessed = foo_NMF['data_target']
    plt.subplot(2,3,2)
    plot_tsne(data_target_preprocessed, trg_labels[:, 1], method_string = 'ConcatenateCluster')

    # SC3 Mix with NMF labels (TransferCluster with NMF labels)
    trg_labels = foo_NMF['trg_labels']
    data_target_preprocessed = foo_NMF['data_target']
    res_opt_mix_ind = foo_NMF['res_opt_mix_ind']
    plt.subplot(2,3,3)
    plot_tsne(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'TransferCluster with NMF labels')
 
    ## SC3 Mix with level 1 labels (TransferCluster with level 1 labels)
    trg_labels = foo_l1['trg_labels']
    data_target_preprocessed = foo_l1['data_target']
    res_opt_mix_ind = foo_l1['res_opt_mix_ind']
    plt.subplot(2,3,4)
    plot_tsne(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'TransferCluster with level 1 labels')	

    ## SC3 Mix with level 2 labels (TransferCluster with level 2 labels)
    trg_labels = foo_l2['trg_labels']
    data_target_preprocessed = foo_l2['data_target']
    res_opt_mix_ind = foo_l2['res_opt_mix_ind']
    plt.subplot(2,3,5)
    plot_tsne(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'TransferCluster with level 2 labels')		

    ## SC3 Mix with level 3 labels (TransferCluster with level 3 labels)
    trg_labels = foo_l3['trg_labels']
    data_target_preprocessed = foo_l3['data_target']
    res_opt_mix_ind = foo_l3['res_opt_mix_ind']
    plt.subplot(2,3,6)
    plot_tsne(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'TransferCluster with level 3 labels')	
	
    plt.savefig(fname_plot+'S9.jpg')

print('Done')
