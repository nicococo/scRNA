import numpy as np
import pdb
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

# Code and examples of Kernel Target Alignments (Christianini et al, NIPS 2001 and JMLR 2002).
# Author: Nico Goernitz, TU Berlin, 2016
# Adjusted for scRNAseq data by Bettina Mieth

def center_kernel(K):
    # Mean free in feature space
    N = K.shape[0]
    a = np.ones((N, N)) / np.float(N)
    return K - a.dot(K) - K.dot(a) + a.dot(K.dot(a))

def normalize_kernel(K):
    # A kernel K is normalized, iff K_ii = 1 \forall i
    N = K.shape[0]
    a = np.sqrt(np.diag(K)).reshape((N, 1))
    if any(np.isnan(a)) or any(np.isinf(a)) or any(np.abs(a)<=1e-16):
        print 'Numerical instabilities.'
        C = np.eye(N)
    else:
        b = 1. / a
        C =  b.dot(b.T)
    return K * C

def center_kernel(K):
    # Mean free in feature space
    N = K.shape[0]
    a = np.ones((N, N)) / np.float(N)
    return K - a.dot(K) - K.dot(a) + a.dot(K.dot(a))

def kta_align_general(K1, K2):
    # Computes the (empirical) alignment of two kernels K1 and K2

    # Definition 1: (Empirical) Alignment
    #   a = <K1, K2>_Frob
    #   b = sqrt( <K1, K1> <K2, K2>)
    #   kta = a / b
    # with <A, B>_Frob = sum_ij A_ij B_ij = tr(AB')
    return K1.dot(K2.T).trace() / np.sqrt(K1.dot(K1.T).trace() * K2.dot(K2.T).trace())


def split_data(org_data):
    inds = np.random.permutation(len(np.transpose(org_data)))
    new_data_1 = org_data[:, inds[:(len(inds)/2)]]
    new_data_2 = org_data[:, inds[(len(inds)/2):]]
    return [new_data_1, new_data_2]


def intersect(a, b):
    return list(set(a) & set(b))


def cross_validation(data, labels):
    kf = KFold(len(labels), n_folds=10, shuffle=True)
    # pdb.set_trace()
    accs = []
    for train, test in kf:
        # print("TRAIN:", train, "TEST:", test)
        x_train, x_test, y_train, y_test = data[:, train], data[:, test], labels[train], labels[test]
        clf_now = SVC()
        clf_now.fit(np.transpose(x_train), y_train)
        accs.append(clf_now.score(np.transpose(x_test), y_test))
    return accs

if __name__ == "__main__":

    # load Uso data
    data_uso = np.load('C:\Users\Bettina\ml\scRNAseq\data\Usoskin.npz')
    cell_names_uso = data_uso['cells']
    transcript_names_uso = data_uso['transcripts']
    data_array_uso = data_uso['data']
    num_genes = len(transcript_names_uso)

    genes_to_keep = 2000

    # Split according to clusters (Clusters 1-5 in one dataset, Clusters 6-11 in the other)
    # Load Cluster Results
    cluster_labels_inf = np.loadtxt("C:\Users\Bettina\ml\scRNAseq\data\cluster_labels_uso.txt", dtype={'names': ('cluster', 'cell_name'), 'formats': ('i2', 'S8')}, skiprows=1)

    cluster_labels = cluster_labels_inf['cluster']
    cells_to_keep = cluster_labels_inf['cell_name']
    new_data_uso = np.asarray([data_array_uso[:,cell_names_uso.tolist().index(cell)] for cell in cells_to_keep])

    data_cluster1_uso = new_data_uso[cluster_labels<5,]
    data_cluster2_uso = new_data_uso[cluster_labels>4,]

    # Random split of Uso
    [data_rand1_uso, data_rand2_uso] = split_data(np.transpose(new_data_uso))

    # To avoid memory error use only little bit of data for now
    indices = np.random.permutation(num_genes)

    data_rand1_uso = data_rand1_uso[indices[:genes_to_keep],]
    data_rand2_uso = data_rand2_uso[indices[:genes_to_keep],]
    data_cluster1_uso = data_cluster1_uso[:,indices[:genes_to_keep]]
    data_cluster2_uso = data_cluster2_uso[:,indices[:genes_to_keep]]

    # setup kernels (all linear, but other kernels are applicable too)
    K_lin_uso_rand1 = np.transpose(data_rand1_uso).T.dot(np.transpose(data_rand1_uso))
    K_lin_uso_rand2 = np.transpose(data_rand2_uso).T.dot(np.transpose(data_rand2_uso))

    K_lin_uso_cluster1 = data_cluster1_uso.T.dot(data_cluster1_uso)
    K_lin_uso_cluster2 = data_cluster2_uso.T.dot(data_cluster2_uso)

    # plot the kernels

    # plt.figure(1)
    #plt.subplot(1, 2, 1)
    #plt.pcolor(K_lin_uso_rand1)
    #plt.subplot(1, 2, 2)
    #plt.pcolor(K_lin_uso_rand2)
    #plt.show()

    #plt.figure(2)
    #plt.subplot(1, 2, 1)
    #plt.pcolor(K_lin_uso_cluster1)
    #plt.subplot(1, 2, 2)
    #plt.pcolor(K_lin_uso_cluster2)
    #plt.show()

    pdb.set_trace()
    labels = np.concatenate([np.array([0] * len(np.transpose(data_rand1_uso))), np.array([1] * len(np.transpose(data_rand2_uso)))])
    data_for_svm = np.concatenate([data_rand1_uso, data_rand2_uso])
    clf = SVC()
    clf.fit(np.transpose(data_for_svm), labels)
    accuracies = cross_validation(data_for_svm, labels)

    print '--------------------------------------------------------------------------'

    print ''

    print '  -KTA for Random Split of Usoskin: '

    print 'Use kta_align_general and center both kernels before.'
    #K_lin_uso_rand1 = center_kernel(K_lin_uso_rand1)
    #K_lin_uso_rand2 = center_kernel(K_lin_uso_rand2)
    print '  -K_lin_uso_rand1. and K_lin_uso_rand2: ', kta_align_general(K_lin_uso_rand1, K_lin_uso_rand2)

    print ''

    print '-KTA for Split of Usoskin according to clusters (Clusters 1-5 in one dataset, Clusters 6-11 in the other): '

    print 'Use kta_align_general and center both kernels before.'
    K_lin_uso_cluster1 = center_kernel(K_lin_uso_cluster1)
    K_lin_uso_cluster2 = center_kernel(K_lin_uso_cluster2)
    print '  -K_lin_uso_cluster1. and K_lin_uso_cluster2: ', kta_align_general(K_lin_uso_cluster1, K_lin_uso_cluster2)

    print("Prediction random split in Usososkin - Mean accuracy: ", np.mean(accuracies))


    print ''
    print '--------------------------------------------------------------------------'


