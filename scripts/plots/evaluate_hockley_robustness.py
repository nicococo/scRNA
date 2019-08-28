###################################################
###						###
###   Evaluation of Robustness experiment using ###
###  written by Bettina Mieth, Nico Görnitz,    ###
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


if __name__ == "__main__":
    # Loading data - Please change directories to yours
    foo_NMF = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_NMFlabels_k7_1000reps.npz')
    foo_l1 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level1labels_k7_1000reps.npz')
    foo_l2 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level2labels_k7_1000reps.npz')
    foo_l3 = np.load('/home/bmieth/scRNAseq/results/jims_data/multiple_reps/jimtarget_usoskinsource_level3labels_k7_1000reps.npz')
    foo_for_clusterident = np.load('/home/bmieth/scRNAseq/results/jims_data/final_for_pub_k7/jimtarget_usoskinsource_level3labels.npz')
		
    num_exps = 1000
	
    # mNP and mNFa clusters
    print('Counting the numbers for mNP and mNFa clusters!')
    # Identify the two clusters
    trg_labels_all = foo_for_clusterident['trg_labels']
    res_opt_mix_ind = foo_for_clusterident['res_opt_mix_ind']
    trg_labels = trg_labels_all[:, res_opt_mix_ind+2]
    cl1 = (trg_labels == 6)
    cell_names_target = foo_for_clusterident['cell_names_target']
    cluster_1 = cell_names_target[cl1].flatten()
    print(cluster_1)
    cl2 = (trg_labels == 4) 
    cluster_2 = cell_names_target[cl2].flatten()
    print(cluster_2)

    # Count how many times those two clusters are seperated correctly
    # TransferCluster with NMF source labels
    trg_labels_NMF = foo_NMF['trg_labels_reps']	
    trg_labels_NMF = trg_labels_NMF[:,2,:]
    counter_NMF = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_NMF[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_NMF[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)

        if cl1_most_common != cl2_most_common:
            counter_NMF=counter_NMF+1
        
    ## SC3 Mix with level 1 labels (TransferCluster with level 1 labels)
    trg_labels_l1 = foo_l1['trg_labels_reps']
    trg_labels_l1 = trg_labels_l1[:,2,:]
    counter_l1 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l1[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l1[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l1=counter_l1+1
	
    ## SC3 Mix with level 2 labels (TransferCluster with level 2 labels)
    trg_labels_l2 = foo_l2['trg_labels_reps']
    trg_labels_l2 = trg_labels_l2[:,2,:]
    counter_l2 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l2[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l2[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l2=counter_l2+1

    # TargetCluster
    trg_labels_l3 = foo_l3['trg_labels_reps']
    data_target_preprocessed = foo_l3['data_target']
    trg_labels_SC3 = trg_labels_l3[:,0,:]
    counter_SC3 = 0

    for i in np.arange(num_exps):
        cl1_labels = trg_labels_SC3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3=counter_SC3+1

    # ConcatenateCluster
    trg_labels_SC3_COMB = trg_labels_l3[:,1,:]
    counter_SC3_COMB = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_SC3_COMB[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3_COMB[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3_COMB=counter_SC3_COMB+1

    # SC3 Mix with level 3 labels (TransferCluster with level 3 labels)
    trg_labels_l3 = trg_labels_l3[:,2,:]
    counter_l3 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l3=counter_l3+1	
			
    # Print results (i.e. counts of successful identifications of mNP / mNFa clusters)			
    print('Counter SC3: ', counter_SC3)
    print('Counter SC3 Comb: ', counter_SC3_COMB)
    print('Cunter SC3 NMF: ', counter_NMF)
    print('Counter SC3 L1: ', counter_l1)
    print('Counter SC3 L2: ', counter_l2)
    print('Counter SC3 L3: ', counter_l3)
	
    # pNF clusters
    print('Counting the numbers for pNF clusters!')
    # Identify the two clusters
    trg_labels_all = foo_for_clusterident['trg_labels']
    res_opt_mix_ind = foo_for_clusterident['res_opt_mix_ind']
    trg_labels = trg_labels_all[:, res_opt_mix_ind+2]
    cl1 = (trg_labels == 0)
    cell_names_target = foo_for_clusterident['cell_names_target']
    cluster_1 = cell_names_target[cl1].flatten()
    cl2 = (trg_labels == 3) 
    cluster_2 = cell_names_target[cl2].flatten()

    # Count how many times those two clusters are seperated correctly
    # TransferCluster with NMF source labels
    trg_labels_NMF = foo_NMF['trg_labels_reps']	
    trg_labels_NMF = trg_labels_NMF[:,2,:]
    counter_NMF = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_NMF[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_NMF[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)

        if cl1_most_common != cl2_most_common:
            counter_NMF=counter_NMF+1
        
    ## SC3 Mix with level 1 labels (TransferCluster with level 1 labels)
    trg_labels_l1 = foo_l1['trg_labels_reps']
    trg_labels_l1 = trg_labels_l1[:,2,:]
    counter_l1 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l1[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l1[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l1=counter_l1+1
	
    ## SC3 Mix with level 2 labels (TransferCluster with level 2 labels)
    trg_labels_l2 = foo_l2['trg_labels_reps']
    trg_labels_l2 = trg_labels_l2[:,2,:]
    counter_l2 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l2[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l2[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l2=counter_l2+1
	
    # TargetCluster
    trg_labels_l3 = foo_l3['trg_labels_reps']
    data_target_preprocessed = foo_l3['data_target']
    trg_labels_SC3 = trg_labels_l3[:,0,:]
    counter_SC3 = 0

    for i in np.arange(num_exps):
        cl1_labels = trg_labels_SC3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3=counter_SC3+1
    
    # ConcatenateCluster
    trg_labels_SC3_COMB = trg_labels_l3[:,1,:]
    counter_SC3_COMB = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_SC3_COMB[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3_COMB[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3_COMB=counter_SC3_COMB+1

    # SC3 Mix with level 3 labels  (TransferCluster with level 3 labels)	
    trg_labels_l3 = trg_labels_l3[:,2,:]
    counter_l3 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l3=counter_l3+1			

    # Print results (i.e. counts of successful identifications of pNF clusters)			
    print('Counter SC3: ', counter_SC3)
    print('Counter SC3 Comb: ', counter_SC3_COMB)
    print('Cunter SC3 NMF: ', counter_NMF)
    print('Counter SC3 L1: ', counter_l1)
    print('Counter SC3 L2: ', counter_l2)
    print('Counter SC3 L3: ', counter_l3)
	
    # pPEP clusters
    print('Counting the numbers for pPEP clusters!')
    # Identify the two clusters
    trg_labels_all = foo_for_clusterident['trg_labels']
    res_opt_mix_ind = foo_for_clusterident['res_opt_mix_ind']
    trg_labels = trg_labels_all[:, res_opt_mix_ind+2]
    cl1 = (trg_labels == 5)
    cell_names_target = foo_for_clusterident['cell_names_target']
    cluster_1 = cell_names_target[cl1].flatten()
    print(cluster_1)
    cl2 = (trg_labels == 2) 
    cluster_2 = cell_names_target[cl2].flatten()
    print(cluster_2)

    # Count how many times those two clusters are seperated correctly
    # TransferCluster with NMF source labels
    trg_labels_NMF = foo_NMF['trg_labels_reps']	
    trg_labels_NMF = trg_labels_NMF[:,2,:]
    counter_NMF = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_NMF[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_NMF[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)

        if cl1_most_common != cl2_most_common:
            counter_NMF=counter_NMF+1
        
    ## SC3 Mix with level 1 labels (TransferCluster with level 1 labels)
    trg_labels_l1 = foo_l1['trg_labels_reps']
    trg_labels_l1 = trg_labels_l1[:,2,:]
    counter_l1 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l1[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l1[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l1=counter_l1+1
	
    ## SC3 Mix with level 2 labels (TransferCluster with level 2 labels)
    trg_labels_l2 = foo_l2['trg_labels_reps']
    trg_labels_l2 = trg_labels_l2[:,2,:]
    counter_l2 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l2[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l2[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l2=counter_l2+1
	
    # TargetCluster
    trg_labels_l3 = foo_l3['trg_labels_reps']
    data_target_preprocessed = foo_l3['data_target']
    trg_labels_SC3 = trg_labels_l3[:,0,:]
    counter_SC3 = 0

    for i in np.arange(num_exps):
        cl1_labels = trg_labels_SC3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3=counter_SC3+1

    # ConcatenateCluster
    trg_labels_SC3_COMB = trg_labels_l3[:,1,:]
    counter_SC3_COMB = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_SC3_COMB[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_SC3_COMB[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_SC3_COMB=counter_SC3_COMB+1

    # SC3 Mix with level 3 labels (TransferCluster with level 3 labels)	
    trg_labels_l3 = trg_labels_l3[:,2,:]
    counter_l3 = 0
    for i in np.arange(num_exps):
        cl1_labels = trg_labels_l3[cl1,i] .tolist()
        cl1_most_common = max(set(cl1_labels), key=cl1_labels.count)
        cl2_labels = trg_labels_l3[cl2,i] .tolist()
        cl2_most_common = max(set(cl2_labels), key=cl2_labels.count)
        if cl1_most_common != cl2_most_common:
            counter_l3=counter_l3+1	
			

    # Print results (i.e. counts of successful identifications of pPEP clusters			
    print('Counter SC3: ', counter_SC3)
    print('Counter SC3 Comb: ', counter_SC3_COMB)
    print('Cunter SC3 NMF: ', counter_NMF)
    print('Counter SC3 L1: ', counter_l1)
    print('Counter SC3 L2: ', counter_l2)
    print('Counter SC3 L3: ', counter_l3)


print('Done')
