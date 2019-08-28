###################################################
###						###
### Plot script for experiment on Hockley data  ###
### using MAGIC pre-processed data		###
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
    #plt.title(method_string +', {0} cluster'.format(num_cluster))
    plt.title(method_string)
    #plt.legend()


if __name__ == "__main__":
    # Figure direction to save to
    fname_plot ='/home/bmieth/scRNAseq/results/jims_data/magic/jimtarget_usoskinsource_magic'
    # Experimental results location - please change directory to yours
    foo_l3 = np.load('/home/bmieth/scRNAseq/results/jims_data/magic/jimtarget_usoskinsource_magic_without_filter.npz')
    # Data location - please change directory to yours	      
    foo_data = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_NMFlabels.npz')
    data_target = foo_data['data_target']
    
    ## TSNE plots of results
    # SC3 results (TargetCluster)
    trg_labels = foo_l3['trg_labels']
    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,3,1)
    plot_tsne(data_target, trg_labels[:, 0], method_string = 'TargetCluster')
    
    ## SC3 comb results (ConcatenateCluster)
    plt.subplot(1,3,2)
    plot_tsne(data_target, trg_labels[:, 1], method_string = 'ConcatenateCluster')

    ## TransferCluster with level 3 labels
    res_opt_mix_ind = foo_l3['res_opt_mix_ind']
    plt.subplot(1,3,3)
    plot_tsne(data_target, trg_labels[:, res_opt_mix_ind+2], method_string = 'TransferCluster with level 3 labels')	
	
    plt.savefig(fname_plot+'.jpg')

print('Done')
