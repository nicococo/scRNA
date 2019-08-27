import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_mixtures_vs_rates(fig_num, res, accs_desc, m_desc, genes, n_src, n_trg, mixes):
    plt.figure(fig_num)
    accs_indices = [0,1]
    cmap = plt.cm.get_cmap('hsv', len(accs_indices) + 1)
    count=0
    for a in range(len(accs_indices)):
        aris = res[accs_indices[a], 2:]
        markers, caps, bars = plt.errorbar(mixes, aris, color=cmap(count), linewidth=2.0)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        plt.plot(mixes[aris.tolist().index(max(aris))], max(aris), 'o', color=cmap(count), label='_nolegend_')
        count += 1
    plt.xlabel('Mixture parameter', fontsize=12)
    plt.ylabel('ARI', fontsize=12)
    #plt.text(1, 6.5, 'True cluster accuracy rates (ARI) vs. unsupervised accuracy measures (KTA score)', fontsize=20)
    plt.ylim([0.0, 1.0])
    plt.xlim([np.min(mixes),np.max(mixes)])
    plt.xticks(mixes, mixes, fontsize=10)
    legend = accs_desc[accs_indices]
    plt.legend(legend, fontsize=12, loc=4)

	
def plot_expression_histogram(data_matrix, num_bins=200, x_range=(0,5), y_lim=100000):
    a_size,b_size = data_matrix.shape
    X = np.log10(data_matrix.reshape(a_size * b_size, 1)+1.)
    #X = data_matrix.reshape(a_size * b_size, 1)
    plt.hist(X, range=x_range,bins=num_bins)
    plt.ylim(0,y_lim)
    #plt.xscale('log')
    plt.title("Histogram of expression values of all {0} genes in all {1} cells (total of {2} values), \n{3} entries equal zero, x- and y-achis are cropped.".format(a_size, b_size, a_size*b_size, np.sum(np.sum(data_matrix==0))))
    plt.xlabel("log10(x_exp +1)")


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
	
	
def plot_pca(data_matrix, labels, method_string):
    #generate a list of markers and another of colors 
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","." , "," , "o" , "v" , "^" , "<", ">"]
    colors = ['r','g','b','c','m', 'y', 'k', 'k', 'y','m', 'c','b','g','r']
    num_cluster = len(np.unique(labels))
    clusters = np.unique(labels)
    
    model = PCA(n_components=2)
    ret = model.fit_transform(data_matrix.T)
    for i in range(num_cluster):
        plt.scatter(ret[labels==clusters[i], 0], ret[labels==clusters[i], 1], 20, marker=markers[i], color = colors[i], label = i)
    plt.title(method_string +', {0} cluster'.format(num_cluster))
    plt.legend()


if __name__ == "__main__":

    fname_plot ='/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_k7_1000reps_figure_'
    foo_NMF = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_NMFlabels_k7_1000reps.npz')
    foo_l1 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level1labels_k7_1000reps.npz')
    foo_l2 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level2labels_k7_1000reps.npz')
    foo_l3 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level3labels_k7_1000reps.npz')
    
    fname_data_target = '/home/bmieth/scRNAseq/data/Jim/Visceraltpm_m_fltd_mat.tsv'
    fname_data_source = '/home/bmieth/scRNAseq/data/usoskin/usoskin_m_fltd_mat.tsv'
    
    ## Figure 1 Expression histograms Target data
    #data_target = pd.read_csv(fname_data_target, sep='\t', header=None).values
    ### reverse log2
    #data_target = np.power(2,data_target)-1
    #fig = plt.figure(figsize=(16,12))
    #plot_expression_histogram(data_target, num_bins=200, x_range=(0,5), y_lim=100000)
    #plt.savefig(fname_plot+'target_expression_histogram'+'.jpg')


    ## Figure 2 Expression histograms Source data
    #data_source = pd.read_csv(fname_data_source, sep='\t', header=None).values
    #fig = plt.figure(figsize=(16,12))
    #plot_expression_histogram(data_source, num_bins=200, x_range=(0,5), y_lim=100000)
    #plt.savefig(fname_plot+'source_expression_histogram'+'.jpg')	
	
    #res = foo_l3['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    #res_opt_mix_ind = foo['res_opt_mix_ind']
    #source_aris = foo['source_aris'] # n_src x genes x common x reps
    #print 'Source ARIs from NMF clustering: ', source_aris
    #accs_desc = foo['accs_desc']
    #print accs_desc
    #method_desc = foo['method_desc']
    #genes = foo['genes']
    #n_src = foo['n_src']
    #n_trg = foo['n_trg']

    #print n_trg
    #mixes = foo['mixes']
    #print 'Mixture parameters: ', mixes
    #print 'KTA optimal mixture parameter: ', mixes[res_opt_mix_ind]
    #print 'acc_funcs x methods'
    #print 'Result dimensionality: ', res.shape
    #print '1'
    #print 'Result optimal mixture parameter dimensionality', res_opt_mix_ind.shape
    
    # Figure 3 TSNE plots of results
	# SC3 results
    trg_labels = foo_l3['cons_clustering_sc3']
    data_target_preprocessed = foo_NMF['data_target']
    fig = plt.figure(figsize=(16,12))
    plt.subplot(2,3,1)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3')
    
    # SC3 comb results
    trg_labels = foo_l3['cons_clustering_sc3_comb']
    data_target_preprocessed = foo_NMF['data_target']
    plt.subplot(2,3,2)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Comb')

    # SC3 Mix with NMF labels
    trg_labels = foo_NMF['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_NMF['data_target']
    plt.subplot(2,3,3)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with NMF labels')
 
    ## SC3 Mix with level 1 labels
    trg_labels = foo_l1['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_l1['data_target']
    plt.subplot(2,3,4)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with level 1 labels')
			
    ## SC3 Mix with level 2 labels
    trg_labels = foo_l2['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_l2['data_target']
    plt.subplot(2,3,5)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with level 2 labels')
	
	## SC3 results
    #trg_labels = foo_l3['cons_clustering_sc3']
    #data_target_preprocessed = foo_l3['data_target']
    #plt.subplot(2,4,6)
    #plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3')
    
    ## SC3 comb results
    #trg_labels = foo_l3['cons_clustering_sc3_comb']
    #data_target_preprocessed = foo_l3['data_target']
    #plt.subplot(2,4,7)
    #plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Comb')

	
    # SC3 Mix with level 3 labels
    trg_labels = foo_l3['cons_clustering_sc3_mix']
    data_target_preprocessed = foo_l3['data_target']
    plt.subplot(2,3,6)
    plot_tsne(data_target_preprocessed, trg_labels, method_string = 'SC3 Mix with level 3 labels')	
	
    plt.savefig(fname_plot+'tsne_plots.jpg')
	
    ## Figure 4 PCA plots of results
	## SC3 results
    #trg_labels = foo_NMF['trg_labels']
    #data_target_preprocessed = foo_NMF['data_target']
    #fig = plt.figure(figsize=(16,12))
    #plt.subplot(2,3,1)
    #plot_pca(data_target_preprocessed, trg_labels[:, 0], method_string = 'SC3')
    
    ## SC3 comb results
    #trg_labels = foo_NMF['trg_labels']
    #data_target_preprocessed = foo_NMF['data_target']
    #plt.subplot(2,3,2)
    #plot_pca(data_target_preprocessed, trg_labels[:, 1], method_string = 'SC3 Comb')

    ## SC3 Mix with NMF labels
    #trg_labels = foo_NMF['trg_labels']
    #data_target_preprocessed = foo_NMF['data_target']
    #res_opt_mix_ind = foo_NMF['res_opt_mix_ind']
    #plt.subplot(2,3,3)
    #plot_pca(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'SC3 Mix with NMF labels')
 
    ## SC3 Mix with level 1 labels
    #trg_labels = foo_l1['trg_labels']
    #data_target_preprocessed = foo_l1['data_target']
    #res_opt_mix_ind = foo_l1['res_opt_mix_ind']
    #plt.subplot(2,3,4)
    #plot_pca(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'SC3 Mix with level 1 labels')	

    ## SC3 Mix with level 2 labels
    #trg_labels = foo_l2['trg_labels']
    #data_target_preprocessed = foo_l2['data_target']
    #res_opt_mix_ind = foo_l2['res_opt_mix_ind']
    #plt.subplot(2,3,5)
    #plot_pca(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'SC3 Mix with level 2 labels')		

    ## SC3 Mix with level 3 labels
    #trg_labels = foo_l3['trg_labels']
    #data_target_preprocessed = foo_l3['data_target']
    #res_opt_mix_ind = foo_l3['res_opt_mix_ind']
    #plt.subplot(2,3,6)
    #plot_pca(data_target_preprocessed, trg_labels[:, res_opt_mix_ind+2], method_string = 'SC3 Mix with level 3 labels')	
	
    #plt.savefig(fname_plot+'pca_plots.jpg')
    
    ## Figure 6 mixture parameters vs. KTA scores
    #fig = plt.figure(figsize=(16,12))
    #plot_mixtures_vs_rates(6, res, accs_desc, method_desc, genes, n_src, n_trg, mixes)
    #plt.savefig(fname_plot+'mix_vs_kta.jpg')

print('Done')
