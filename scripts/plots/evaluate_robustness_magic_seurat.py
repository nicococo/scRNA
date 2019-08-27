import sys
sys.path.append('/home/bmieth/scRNAseq/implementations')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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

    #foo_NMF = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_NMFlabels_k7_1000reps.npz')
    #foo_l1 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level1labels_k7_1000reps.npz')
    #foo_l2 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level2labels_k7_1000reps.npz')
    #foo_l3 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level3labels_k7_1000reps.npz')
	
    foo = np.load('/home/bmieth/scRNAseq/results/jims_data/magic/jimtarget_usoskinsource_magic_1000reps.npz')
	
    fname_plot ='/home/bmieth/scRNAseq/results/jims_data/magic/jimtarget_usoskinsource_magic_1000reps_'

	
    foo_for_clusterident = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_level3labels.npz')
	
    #fname_plot = '/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_NMFlabels_k9_9reps_SC3Comb.jpg'
	
    num_exps = foo['reps']
	
	# mNP and mNFa clusters
    print('Counting the numbers for mNP and mNFa clusters!')
	# Identify the two clusters
    trg_labels_all = foo_for_clusterident['trg_labels']
    res_opt_mix_ind = foo_for_clusterident['res_opt_mix_ind']
    trg_labels = trg_labels_all[:, res_opt_mix_ind+2]
    #trg_labels = foo['cons_clustering_sc3_mix']
    cl1 = (trg_labels == 6)
    cell_names_target = foo_for_clusterident['cell_names_target']
    cluster_1 = cell_names_target[cl1].flatten()
    #print cluster_1
    cl2 = (trg_labels == 4) 
    cluster_2 = cell_names_target[cl2].flatten()
    #print cluster_2

	## Count how many times those two clusters are seperated correctly
	## NMF results
    #trg_labels_NMF = foo_NMF['trg_labels_reps']	
    #trg_labels_NMF = trg_labels_NMF[:,2,:]
    #counter_NMF = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_NMF[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_NMF[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)

    #    if cl1_most_common != cl2_most_common:
    #        counter_NMF=counter_NMF+1
    #    
    ### SC3 Mix with level 1 labels
    #trg_labels_l1 = foo_l1['trg_labels_reps']
    #trg_labels_l1 = trg_labels_l1[:,2,:]
    #counter_l1 = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_l1[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_l1[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
    #    if cl1_most_common != cl2_most_common:
    #        counter_l1=counter_l1+1
	
    ## SC3 Mix with level 2 labels
    #trg_labels_l2 = foo_l2['trg_labels_reps']
    #trg_labels_l2 = trg_labels_l2[:,2,:]
    #counter_l2 = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_l2[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_l2[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
    #    if cl1_most_common != cl2_most_common:
    #        counter_l2=counter_l2+1
	
	
    # mNP/mNFA cluster
    # SC3 Mix with level 3 labels
    trg_labels_l3 = foo['trg_labels_reps']
    data_target_preprocessed = foo['data_target']
    trg_labels_SC3 = trg_labels_l3[:,0,:]
    counter_SC3 = 0
	
    successful_flag = np.ones(num_exps, dtype=bool)

	# SC3 alone
    for i in np.arange(num_exps):
        #if i == 14:
        #    pdb.set_trace()
        cl1_labels = trg_labels_SC3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3=counter_SC3+1
            successful_flag[i] = False
        #plt.subplot(3,3,i+1-10)
        #plot_tsne(data_target_preprocessed, trg_labels_SC3[:,i], 'SC3, rep: {0}'.format(i))
    #plt.savefig(fname_plot)

    # SC3 comb
    #fig = plt.figure(figsize=(16,12))
    trg_labels_SC3_COMB = trg_labels_l3[:,1,:]
    counter_SC3_COMB = 0
    for i in np.arange(num_exps):
        #pdb.set_trace()
        cl1_labels = trg_labels_SC3_COMB[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3_COMB[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3_COMB=counter_SC3_COMB+1
        else:
            successful_flag[i] = False
        #plt.subplot(3,3,i+1)
        #plot_tsne(data_target_preprocessed, trg_labels_SC3_COMB[:,i], 'SC3 Comb, rep: {0}'.format(i))
    #plt.savefig(fname_plot)	
	
    trg_labels_l3 = trg_labels_l3[:,2,:]
    counter_l3 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l3=counter_l3+1	
        else:
            successful_flag[i] = False
			


    print('Counter SC3: ', counter_SC3)
    print('Counter SC3 Comb: ', counter_SC3_COMB)
    #print 'Cunter SC3 NMF: ', counter_NMF
    #print 'Counter SC3 L1: ', counter_l1
    #print 'Counter SC3 L2: ', counter_l2
    print('Counter SC3 L3: ', counter_l3)
	
	
	
	
	
	# pNF clusters
    print('Counting the numbers for pNF clusters!')
	# Identify the two clusters
    trg_labels_all = foo_for_clusterident['trg_labels']
    res_opt_mix_ind = foo_for_clusterident['res_opt_mix_ind']
    trg_labels = trg_labels_all[:, res_opt_mix_ind+2]
    #trg_labels = foo['cons_clustering_sc3_mix']
    cl1 = (trg_labels == 0)
    cell_names_target = foo_for_clusterident['cell_names_target']
    cluster_1 = cell_names_target[cl1].flatten()
    #print cluster_1
    cl2 = (trg_labels == 3) 
    cluster_2 = cell_names_target[cl2].flatten()
    #print cluster_2

	## Count how many times those two clusters are seperated correctly
	## NMF results
    #trg_labels_NMF = foo_NMF['trg_labels_reps']	
    #trg_labels_NMF = trg_labels_NMF[:,2,:]
    #counter_NMF = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_NMF[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_NMF[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)#

    #    if cl1_most_common != cl2_most_common:
    #        counter_NMF=counter_NMF+1
        
    ## SC3 Mix with level 1 labels
    #trg_labels_l1 = foo_l1['trg_labels_reps']
    #trg_labels_l1 = trg_labels_l1[:,2,:]
    #counter_l1 = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_l1[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_l1[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
    #    if cl1_most_common != cl2_most_common:
    #        counter_l1=counter_l1+1
	
    ## SC3 Mix with level 2 labels
    #trg_labels_l2 = foo_l2['trg_labels_reps']
    #trg_labels_l2 = trg_labels_l2[:,2,:]
    #counter_l2 = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_l2[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_l2[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
    #    if cl1_most_common != cl2_most_common:
    #        counter_l2=counter_l2+1
	
    # SC3 Mix with level 3 labels
    trg_labels_l3 = foo['trg_labels_reps']
    data_target_preprocessed = foo['data_target']
    trg_labels_SC3 = trg_labels_l3[:,0,:]
    counter_SC3 = 0

	# SC3 alone
    for i in np.arange(num_exps):
        #if i == 14:
        #    pdb.set_trace()
        cl1_labels = trg_labels_SC3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3=counter_SC3+1
        else:
            successful_flag[i] = False
        #plt.subplot(3,3,i+1-10)
        #plot_tsne(data_target_preprocessed, trg_labels_SC3[:,i], 'SC3, rep: {0}'.format(i))
    #plt.savefig(fname_plot)

	# SC3 Comb
    #fig = plt.figure(figsize=(16,12))
    trg_labels_SC3_COMB = trg_labels_l3[:,1,:]
    counter_SC3_COMB = 0
    for i in np.arange(num_exps):
        #pdb.set_trace()
        cl1_labels = trg_labels_SC3_COMB[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3_COMB[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3_COMB=counter_SC3_COMB+1
        else:
            successful_flag[i] = False
        #plt.subplot(3,3,i+1)
        #plot_tsne(data_target_preprocessed, trg_labels_SC3_COMB[:,i], 'SC3 Comb, rep: {0}'.format(i))
    #plt.savefig(fname_plot)	
	
    trg_labels_l3 = trg_labels_l3[:,2,:]
    counter_l3 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l3=counter_l3+1	
        else:
            successful_flag[i] = False


    print('Counter SC3: ', counter_SC3)
    print('Counter SC3 Comb: ', counter_SC3_COMB)
    #print 'Cunter SC3 NMF: ', counter_NMF
    #print 'Counter SC3 L1: ', counter_l1
    #print 'Counter SC3 L2: ', counter_l2
    print('Counter SC3 L3: ', counter_l3)
	
	
	
	# pPEP clusters
    print('Counting the numbers for pPEP clusters!')
	# Identify the two clusters
    trg_labels_all = foo_for_clusterident['trg_labels']
    res_opt_mix_ind = foo_for_clusterident['res_opt_mix_ind']
    trg_labels = trg_labels_all[:, res_opt_mix_ind+2]
    #trg_labels = foo['cons_clustering_sc3_mix']
    cl1 = (trg_labels == 5)
    cell_names_target = foo_for_clusterident['cell_names_target']
    cluster_1 = cell_names_target[cl1].flatten()
    #print cluster_1
    cl2 = (trg_labels == 2) 
    cluster_2 = cell_names_target[cl2].flatten()
    #print cluster_2

	## Count how many times those two clusters are seperated correctly
	## NMF results
    #trg_labels_NMF = foo_NMF['trg_labels_reps']	
    #trg_labels_NMF = trg_labels_NMF[:,2,:]
    #counter_NMF = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_NMF[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_NMF[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)##

    #    if cl1_most_common != cl2_most_common:
    #        counter_NMF=counter_NMF+1
        
    ## SC3 Mix with level 1 labels
    #trg_labels_l1 = foo_l1['trg_labels_reps']
    #trg_labels_l1 = trg_labels_l1[:,2,:]
    #counter_l1 = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_l1[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_l1[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
    #    if cl1_most_common != cl2_most_common:
    #        counter_l1=counter_l1+1
	
    ## SC3 Mix with level 2 labels
    #trg_labels_l2 = foo_l2['trg_labels_reps']
    #trg_labels_l2 = trg_labels_l2[:,2,:]
    #counter_l2 = 0
    #for i in np.arange(num_exps):
    #    cl1_labels = trg_labels_l2[cl1,i] .tolist()
    #    cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
    #    cl2_labels = trg_labels_l2[cl2,i] .tolist()
    #    cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
    #    if cl1_most_common != cl2_most_common:
    #        counter_l2=counter_l2+1
	
    # SC3 Mix with level 3 labels
    trg_labels_l3 = foo['trg_labels_reps']
    data_target_preprocessed = foo['data_target']
    trg_labels_SC3 = trg_labels_l3[:,0,:]
    counter_SC3 = 0

	# SC3 alone
    for i in np.arange(num_exps):
        #if i == 14:
        #    pdb.set_trace()
        cl1_labels = trg_labels_SC3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3=counter_SC3+1
        else:
            successful_flag[i] = False
        #plt.subplot(3,3,i+1-10)
        #plot_tsne(data_target_preprocessed, trg_labels_SC3[:,i], 'SC3, rep: {0}'.format(i))
    #plt.savefig(fname_plot)

	# SC3 Comb
    #fig = plt.figure(figsize=(16,12))
    trg_labels_SC3_COMB = trg_labels_l3[:,1,:]
    counter_SC3_COMB = 0
    for i in np.arange(num_exps):
        #pdb.set_trace()
        cl1_labels = trg_labels_SC3_COMB[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3_COMB[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3_COMB=counter_SC3_COMB+1
            successful_flag[i] = False
        #plt.subplot(3,3,i+1)
        #plot_tsne(data_target_preprocessed, trg_labels_SC3_COMB[:,i], 'SC3 Comb, rep: {0}'.format(i))
    #plt.savefig(fname_plot)	
	
    trg_labels_l3 = trg_labels_l3[:,2,:]
    counter_l3 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l3=counter_l3+1	
        else:
            successful_flag[i] = False


    print('Counter SC3: ', counter_SC3)
    print('Counter SC3 Comb: ', counter_SC3_COMB)
    #print 'Cunter SC3 NMF: ', counter_NMF
    #print 'Counter SC3 L1: ', counter_l1
    #print 'Counter SC3 L2: ', counter_l2
    print('Counter SC3 L3: ', counter_l3)
    print(np.where(successful_flag)[0])
	
	## TSNE Plot results of one or all successful replication
    ##index_success = successful_flag.tolist().index(True)
    #foo_data = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_NMFlabels.npz')
    #data_target = foo_data['data_target']
    #index_success = 13
	
	
    #for index_success in np.where(~successful_flag)[0]: 
    #    print index_success
    #    # SC3 results
    #    fig = plt.figure(figsize=(16,6))
    #    plt.subplot(1,3,1)
    #    plot_tsne(data_target, trg_labels_SC3[:, index_success], method_string = 'TargetCluster')
    # 
    #     ## SC3 comb results
    #     plt.subplot(1,3,2)
    #     plot_tsne(data_target, trg_labels_SC3_COMB[:,index_success], method_string = 'ConcatenateCluster')

    #    # SC3 Mix with level 3 labels
    #    plt.subplot(1,3,3)
    #    #plot_tsne(data_target, trg_labels[:, 2], method_string = 'TransferCluster with level 3 labels')	
    #    plot_tsne(data_target, trg_labels_l3[:, index_success], method_string = 'TransferCluster with level 3 labels')	
    #    plt.savefig(fname_plot+ str(index_success)+'_'+'Seurat.jpg')
	
	

print('Done')
