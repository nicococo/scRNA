###################################################
###						###
### Plot script for robustness experiment	### 
### on Hockley data  				###
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
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","." , "," , "o" , "v" , "^" , "<", ">", "." ]
    colors = ['r','g','b','c','m', 'y', 'k', 'k', 'y','m', 'b','c','g','r']
    num_cluster = len(np.unique(labels))
    clusters = np.unique(labels)
    
    model = TSNE(n_components=2, random_state=0, init='pca', metric='euclidean', perplexity=30, method='exact')
    ret = model.fit_transform(data_matrix.T)
    for i in range(num_cluster):
        plt.scatter(ret[labels==clusters[i], 0], ret[labels==clusters[i], 1], 20, marker=markers[i], color = colors[i], label=i)
    plt.title(method_string +', {0} cluster'.format(num_cluster))
    plt.legend()

if __name__ == "__main__":
    # Figure direction to save to
    fname_plot ='/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_k7_1000reps_figure_'
    # Experimental results location - please change directory to yours
    foo_NMF = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_NMFlabels_k7_1000reps.npz')
    foo_l1 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level1labels_k7_1000reps.npz')
    foo_l2 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level2labels_k7_1000reps.npz')
    foo_l3 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level3labels_k7_1000reps.npz')
    
    # Data location - please change directory to yours	        
    fname_data_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_mat.tsv'
    fname_data_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_mat.tsv'
   
    # TSNE plots of results
    # SC3 results, Consensus results for TargetCluster
    trg_labels = foo_l3['cons_clustering_sc3']
    data_target_preprocessed = foo_NMF['data_target']
    fig = plt.figure(figsize=(16,12))
    plt.subplot(2,3,1)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3')
    
    # SC3 comb results, Consensus results for ConcatenateCluster
    trg_labels = foo_l3['cons_clustering_sc3_comb']
    data_target_preprocessed = foo_NMF['data_target']
    plt.subplot(2,3,2)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Comb')

    # SC3 Mix with NMF labels, Consensus results for TransferCluster with NMF labels
    trg_labels = foo_NMF['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_NMF['data_target']
    plt.subplot(2,3,3)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with NMF labels')
 
    ## SC3 Mix with level 1 labels, Consensus results for TransferCluster with level 1 labels 
    trg_labels = foo_l1['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_l1['data_target']
    plt.subplot(2,3,4)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with level 1 labels')
			
    ## SC3 Mix with level 2 labels, Consensus results for TransferCluster with level 2 labels
    trg_labels = foo_l2['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_l2['data_target']
    plt.subplot(2,3,5)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with level 2 labels')
	
    # SC3 Mix with level 3 labels, Consensus results for TransferCluster with level 3 labels
    trg_labels = foo_l3['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_l3['data_target']
    plt.subplot(2,3,6)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with level 3 labels')	
	
    plt.savefig(fname_plot+'tsne_plots.jpg')

print('Done')
