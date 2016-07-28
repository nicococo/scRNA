import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn import preprocessing
import time

# Code for scRNAseq data analysis.
# Author: Bettina Mieth, TU Berlin, 2016
# Code and examples of Kernel Target Alignments (Christianini et al, NIPS 2001 and JMLR 2002).
# Author: Nico Goernitz, TU Berlin, 2016


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
        C = b.dot(b.T)
    return K * C


def kta_align_general(K1, K2):
    # Computes the (empirical) alignment of two kernels K1 and K2

    # Definition 1: (Empirical) Alignment
    #   a = <K1, K2>_Frob
    #   b = sqrt( <K1, K1> <K2, K2>)
    #   kta = a / b
    # with <A, B>_Frob = sum_ij A_ij B_ij = tr(AB')
    return K1.dot(K2.T).trace() / np.sqrt(K1.dot(K1.T).trace() * K2.dot(K2.T).trace())


def split_data(org_data):
    inds = np.random.permutation(len(org_data))
    new_data_1 = org_data[inds[:(len(inds)/2)],]
    new_data_2 = org_data[inds[(len(inds)/2):],]
    return [new_data_1, new_data_2]


def intersect(a, b):
    return list(set(a) & set(b))


def cross_validation(data, labels, plot_var):
    if len(labels) == np.shape(data)[0]:
        data = np.transpose(data)
    kf = StratifiedKFold(labels, n_folds=np.amin([len(labels),10, sum(labels),len(labels)-sum(labels)]))
    accs_all = []
    aucs_all = []
    gridsearch_best_acc = []
    gridsearch_best_C = []

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(kf):
        x_train_raw, x_test_raw, y_train, y_test = data[:, train], data[:, test], labels[train], labels[test]
        x_train = preprocessing.scale(np.transpose(x_train_raw))
        x_test = preprocessing.scale(np.transpose(x_test_raw))
        clf = GridSearchCV(estimator=SVC(probability=True, kernel='linear'), param_grid=dict(C=np.logspace(-10, 0, 10)), n_jobs=-1, cv=StratifiedKFold(y_train))
        clf.fit(x_train, y_train)
        gridsearch_best_acc.append(clf.best_score_)
        gridsearch_best_C.append(clf.best_estimator_.C)
        accs_all.append(clf.score(x_test, y_test))
        probas_ = clf.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs_all.append(roc_auc)

        if plot_var:
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    if plot_var:
        mean_tpr /= len(kf)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    return accs_all, aucs_all


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

if __name__ == "__main__":

    # load Pfizer + Uso data
    # Pfizer + Usoskin
    data_pfizer_uso = np.loadtxt('C:\Users\Bettina\ml\scRNAseq\data\Pfizer Uso data\data_pfizer_uso.txt')
    cell_names_pfizer_uso = np.loadtxt('C:\Users\Bettina\ml\scRNAseq\data\Pfizer Uso data\cell_names_pfizer_uso.txt', dtype='string')
    transcript_names_pfizer_uso = np.loadtxt('C:\Users\Bettina\ml\scRNAseq\data\Pfizer Uso data\\transcript_names_pfizer_uso.txt', dtype='string')
    data_array_pfizer_uso = np.transpose(np.nan_to_num(data_pfizer_uso))
    num_genes = len(transcript_names_pfizer_uso)
    genes_to_keep = 3000

    # Split in Pfizer and Usoskin
    data_array_pfizer = data_array_pfizer_uso[0:417,]
    data_array_uso = data_array_pfizer_uso[417:,]


    # Split according to clusters (Clusters 1-5 in one dataset, Clusters 6-11 in the other)
    # Load Cluster Results
    cluster_labels_inf = np.loadtxt("C:/Users/Bettina/ml/scRNAseq/Results/SC3 results/Pfizer + Uso data/cluster_4_labels_pfizer_uso_after_scaling.txt",
                                    dtype={'names': ('cluster', 'cell_name'), 'formats': ('i2', 'S8')}, skiprows=1)

    cluster_labels_raw = cluster_labels_inf['cluster']
    cells_to_keep_raw = cluster_labels_inf['cell_name']
    cells_to_keep = cells_to_keep_raw[cells_to_keep_raw!='NA']
    cluster_labels = cluster_labels_raw[cells_to_keep_raw!='NA']

    new_data_pfizer_uso = np.asarray([data_array_pfizer_uso[cell_names_pfizer_uso.tolist().index(cell),] for cell in cells_to_keep])
    num_clusters = max(cluster_labels)

    data_cluster1_uso = new_data_pfizer_uso[cluster_labels==1,]
    data_cluster2_uso = new_data_pfizer_uso[cluster_labels>1,]

    data_cluster3_uso = new_data_pfizer_uso[(cluster_labels == 2)]
    data_cluster4_uso = new_data_pfizer_uso[(cluster_labels < 2) | (cluster_labels > 2)]

    data_cluster5_uso = new_data_pfizer_uso[(cluster_labels == 3)]
    data_cluster6_uso = new_data_pfizer_uso[(cluster_labels < 3) | (cluster_labels > 3)]

    data_cluster7_uso = new_data_pfizer_uso[(cluster_labels == 4)]
    data_cluster8_uso = new_data_pfizer_uso[(cluster_labels < 4)]

    # To avoid memory error use only little bit of data for now
    indices = np.random.permutation(num_genes)

    # Random split of Uso
    [data_rand1_pfizer_uso, data_rand2_pfizer_uso] = split_data(data_array_pfizer_uso)
    data_rand1 = data_rand1_pfizer_uso[:,indices[:genes_to_keep]]
    data_rand2 = data_rand2_pfizer_uso[:,indices[:genes_to_keep]]
    data_pfizer = data_array_pfizer[:, indices[:genes_to_keep]]
    data_uso = data_array_uso[:, indices[:genes_to_keep]]
    data_cluster1_uso = data_cluster1_uso[:, indices[:genes_to_keep]]
    data_cluster2_uso = data_cluster2_uso[:, indices[:genes_to_keep]]
    data_cluster3_uso = data_cluster3_uso[:, indices[:genes_to_keep]]
    data_cluster4_uso = data_cluster4_uso[:, indices[:genes_to_keep]]
    data_cluster5_uso = data_cluster5_uso[:, indices[:genes_to_keep]]
    data_cluster6_uso = data_cluster6_uso[:, indices[:genes_to_keep]]
    data_cluster7_uso = data_cluster7_uso[:, indices[:genes_to_keep]]
    data_cluster8_uso = data_cluster8_uso[:, indices[:genes_to_keep]]

    # setup kernels (all linear, but other kernels are applicable too)
    K_lin_rand1 = np.transpose(data_rand1).T.dot(np.transpose(data_rand1))
    K_lin_rand2 = np.transpose(data_rand2).T.dot(np.transpose(data_rand2))

    K_lin_pfizer = data_pfizer.T.dot(data_pfizer)
    K_lin_uso = data_uso.T.dot(data_uso)

    K_lin_uso_cluster1 = data_cluster1_uso.T.dot(data_cluster1_uso)
    K_lin_uso_cluster2 = data_cluster2_uso.T.dot(data_cluster2_uso)

    K_lin_uso_cluster3 = data_cluster3_uso.T.dot(data_cluster3_uso)
    K_lin_uso_cluster4 = data_cluster4_uso.T.dot(data_cluster4_uso)

    K_lin_uso_cluster5 = data_cluster5_uso.T.dot(data_cluster5_uso)
    K_lin_uso_cluster6 = data_cluster6_uso.T.dot(data_cluster6_uso)

    K_lin_uso_cluster7 = data_cluster7_uso.T.dot(data_cluster7_uso)
    K_lin_uso_cluster8 = data_cluster8_uso.T.dot(data_cluster8_uso)

    labels_rand = np.concatenate([np.array([0] * len(np.transpose(data_rand1))), np.array([1] * len(np.transpose(data_rand2)))])
    data_for_svm_rand = np.concatenate([np.transpose(data_rand1), np.transpose(data_rand2)])
    accuracies_rand, aucs_rand = cross_validation(np.transpose(data_for_svm_rand), labels_rand, True)

    labels_datasets = np.concatenate([np.array([0] * len(data_pfizer)), np.array([1] * len(data_uso))])
    data_for_svm_cluster = np.concatenate([data_pfizer, data_uso])
    accuracies_datasets, aucs_datasets = cross_validation(np.transpose(data_for_svm_cluster), labels_datasets, True)

    labels_cluster = np.concatenate([np.array([0] * len(data_cluster1_uso)), np.array([1] * len(data_cluster2_uso))])
    data_for_svm_cluster = np.concatenate([data_cluster1_uso, data_cluster2_uso])
    accuracies_cluster_12, aucs_cluster_12 = cross_validation(data_for_svm_cluster, labels_cluster, False)

    labels_cluster = np.concatenate([np.array([0] * len(data_cluster3_uso)), np.array([1] * len(data_cluster4_uso))])
    data_for_svm_cluster = np.concatenate([data_cluster3_uso, data_cluster4_uso])
    accuracies_cluster_34, aucs_cluster_34 = cross_validation(data_for_svm_cluster, labels_cluster, False)

    labels_cluster = np.concatenate([np.array([0] * len(data_cluster5_uso)), np.array([1] * len(data_cluster6_uso))])
    data_for_svm_cluster = np.concatenate([data_cluster5_uso, data_cluster6_uso])
    accuracies_cluster_56, aucs_cluster_56 = cross_validation(data_for_svm_cluster, labels_cluster, False)

    labels_cluster = np.concatenate([np.array([0] * len(data_cluster7_uso)), np.array([1] * len(data_cluster8_uso))])
    data_for_svm_cluster = np.concatenate([data_cluster7_uso, data_cluster8_uso])
    accuracies_cluster_78, aucs_cluster_78 = cross_validation(data_for_svm_cluster, labels_cluster, False)

    print '--------------------------------------------------------------------------'
    print ''
    print 'KTA for random split of Pfizer + Usoskin:                                ', '{:.3f}'.format(kta_align_general(K_lin_rand1, K_lin_rand2))
    print 'KTAs for split in Pfizer and Usoskin'
    print '     Pfizer vs. Usoskin:                                                 ', '{:.3f}'.format(kta_align_general(K_lin_pfizer, K_lin_uso))
    print '     Cluster 1 in one dataset, Clusters 2-4 in the other:                ', '{:.3f}'.format(kta_align_general(K_lin_uso_cluster1, K_lin_uso_cluster2))
    print '     Cluster 2 in one dataset, Clusters 1, 3-4 in the other:             ', '{:.3f}'.format(kta_align_general(K_lin_uso_cluster3, K_lin_uso_cluster4))
    print '     Clusters 3 in one dataset, Clusters 1-2, 4 in the other:            ', '{:.3f}'.format(kta_align_general(K_lin_uso_cluster5, K_lin_uso_cluster6))
    print '     Clusters 4 in one dataset, Clusters 1-3 in the other:               ', '{:.3f}'.format(kta_align_general(K_lin_uso_cluster7, K_lin_uso_cluster8))
    print ''
    print 'centered KTA for random split of Pfizer + Usoskin:                       ', '{:.3f}'.format(
        kta_align_general(center_kernel(K_lin_rand1), center_kernel(K_lin_rand2)))
    print 'centered KTAs for split in Pfizer and Usoskin:'
    print '     Pfizer vs. Usoskin:                                                 ', '{:.3f}'.format(
        kta_align_general(center_kernel(K_lin_pfizer), center_kernel(K_lin_uso)))
    print '     Cluster 1 in one dataset, Clusters 2-4 in the other:                ', '{:.3f}'.format(
       kta_align_general(center_kernel(K_lin_uso_cluster1), center_kernel(K_lin_uso_cluster2)))
    print '     Cluster 2 in one dataset, Clusters 1, 2-4 in the other:             ', '{:.3f}'.format(
        kta_align_general(center_kernel(K_lin_uso_cluster3), center_kernel(K_lin_uso_cluster4)))
    print '     Clusters 3 in one dataset, Clusters 1-2, 4 in the other:            ', '{:.3f}'.format(
        kta_align_general(center_kernel(K_lin_uso_cluster5), center_kernel(K_lin_uso_cluster6)))
    print '     Clusters 4 in one dataset, Clusters 1-3 in the other:               ', '{:.3f}'.format(
        kta_align_general(center_kernel(K_lin_uso_cluster7), center_kernel(K_lin_uso_cluster8)))
    print ''
    print 'centered and normalized KTA for random split of Pfizer + Usoskin:        ', '{:.3f}'.format(
        kta_align_general(normalize_kernel(center_kernel(K_lin_rand1)), normalize_kernel(center_kernel(K_lin_rand2))))
    print 'centered and normalized KTAs for split in Pfizer and Usoskin:'
    print '     Cluster 1 in one dataset, Clusters 2-6 in the other:                ', '{:.3f}'.format(
        kta_align_general(normalize_kernel(center_kernel(K_lin_pfizer)), normalize_kernel(center_kernel(K_lin_uso))))
    print '     Cluster 1 in one dataset, Clusters 2-4 in the other:                ', '{:.3f}'.format(
        kta_align_general(normalize_kernel(center_kernel(K_lin_uso_cluster1)), normalize_kernel(center_kernel(K_lin_uso_cluster2))))
    print '     Cluster 2 in one dataset, Clusters 1, 3-4 in the other:             ', '{:.3f}'.format(
        kta_align_general(normalize_kernel(center_kernel(K_lin_uso_cluster3)), normalize_kernel(center_kernel(K_lin_uso_cluster4))))
    print '     Clusters 3 in one dataset, Clusters 1-2, 4 in the other:            ', '{:.3f}'.format(
        kta_align_general(normalize_kernel(center_kernel(K_lin_uso_cluster5)), normalize_kernel(center_kernel(K_lin_uso_cluster6))))
    print '     Clusters 4 in one dataset, Clusters 1-3 in the other:               ', '{:.3f}'.format(
        kta_align_general(normalize_kernel(center_kernel(K_lin_uso_cluster7)), normalize_kernel(center_kernel(K_lin_uso_cluster8))))
    print ''
    [acc_mean, acc_lci, acc_uci] = mean_confidence_interval(accuracies_rand, confidence=0.95)
    print 'Mean SVM accuracy for random split of Pfizer + Usoskin:                  ', '{:.3f}'.format(acc_mean), '- 95% CI [', '{:.3f}'.format(
        acc_lci), ',', '{:.3f}'.format(acc_uci), '].'
    [acc_mean, acc_lci, acc_uci] = mean_confidence_interval(accuracies_datasets, confidence=0.95)
    print 'Mean SVM accuracies for split Pfizer and Usoskin:'
    print '     Pfizer vs. Usoskin:                                                 ', '{:.3f}'.format(acc_mean), '- 95% CI [', '{:.3f}'.format(
        acc_lci), ',', '{:.3f}'.format(acc_uci), '].'
    [acc_mean, acc_lci, acc_uci] = mean_confidence_interval(accuracies_cluster_12, confidence=0.95)
    print '     Clusters 1 in one dataset, Clusters 2-4 in the other:               ', '{:.3f}'.format(acc_mean), '- 95% CI [', '{:.3f}'.format(
        acc_lci), ',', '{:.3f}'.format(acc_uci), '].'
    [acc_mean, acc_lci, acc_uci] = mean_confidence_interval(accuracies_cluster_34, confidence=0.95)
    print '     Clusters 2 in one dataset, Clusters 1,3-4 in the other:             ', '{:.3f}'.format(acc_mean), '- 95% CI [', '{:.3f}'.format(
        acc_lci), ',', '{:.3f}'.format(acc_uci), '].'
    [acc_mean, acc_lci, acc_uci] = mean_confidence_interval(accuracies_cluster_56, confidence=0.95)
    print '     Clusters 3 in one dataset, Clusters 1-2,4 in the other:             ', '{:.3f}'.format(acc_mean), '- 95% CI [', '{:.3f}'.format(
        acc_lci), ',', '{:.3f}'.format(acc_uci), '].'
    [acc_mean, acc_lci, acc_uci] = mean_confidence_interval(accuracies_cluster_78, confidence=0.95)
    print '     Clusters 4 in one dataset, Clusters 1-3 in the other:               ', '{:.3f}'.format(acc_mean), '- 95% CI [', '{:.3f}'.format(
        acc_lci), ',', '{:.3f}'.format(acc_uci), '].'
    print ''

    [auc_mean, auc_lci, auc_uci] = mean_confidence_interval(aucs_rand, confidence=0.95)
    print 'Mean SVM AUCs for random split of Pfizer + Usoskin:                      ', '{:.3f}'.format(auc_mean), '- 95% CI [', '{:.3f}'.format(auc_lci), ',', '{:.3f}'.format(
        auc_uci), '].'
    [auc_mean, auc_lci, auc_uci] = mean_confidence_interval(aucs_datasets, confidence=0.95)
    print 'Mean SVM AUCs for split in Pfizer and Usoskin:'
    print '     Pfizer vs. Usoskin:                                                 ', '{:.3f}'.format(auc_mean), '- 95% CI [', '{:.3f}'.format(
        auc_lci), ',', '{:.3f}'.format(auc_uci), '].'
    [auc_mean, auc_lci, auc_uci] = mean_confidence_interval(aucs_cluster_12, confidence=0.95)
    print '     Clusters 1 in one dataset, Clusters 2-4 in the other:               ', '{:.3f}'.format(auc_mean), '- 95% CI [', '{:.3f}'.format(
        auc_lci), ',', '{:.3f}'.format(auc_uci), '].'
    [auc_mean, auc_lci, auc_uci] = mean_confidence_interval(aucs_cluster_34, confidence=0.95)
    print '     Clusters 2 in one dataset, Clusters 1,3-4 in the other:             ', '{:.3f}'.format(auc_mean), '- 95% CI [', '{:.3f}'.format(
        auc_lci), ',', '{:.3f}'.format(auc_uci), '].'
    [auc_mean, auc_lci, auc_uci] = mean_confidence_interval(aucs_cluster_56, confidence=0.95)
    print '     Clusters 3 in one dataset, Clusters 1-2,4 in the other:             ', '{:.3f}'.format(auc_mean), '- 95% CI [', '{:.3f}'.format(
        auc_lci), ',', '{:.3f}'.format(auc_uci), '].'
    [auc_mean, auc_lci, auc_uci] = mean_confidence_interval(aucs_cluster_78, confidence=0.95)
    print '     Clusters 4 in one dataset, Clusters 1-3 in the other:               ', '{:.3f}'.format(auc_mean), '- 95% CI [', '{:.3f}'.format(
        auc_lci), ',', '{:.3f}'.format(auc_uci), '].'
    print ''

    pairwise_KTAs = [[None for i in range(num_clusters)] for j in range(num_clusters)]
    pairwise_KTAs_center = [[None for i in range(num_clusters)] for j in range(num_clusters)]
    pairwise_KTAs_center_norma = [[None for i in range(num_clusters)] for j in range(num_clusters)]
    pairwise_accuracies = [[None for i in range(num_clusters)] for j in range(num_clusters)]
    pairwise_aucs = [[None for i in range(num_clusters)] for j in range(num_clusters)]

    for cluster_1 in range(num_clusters):
        for cluster_2 in range(num_clusters):
            print(cluster_1+1, cluster_2+1)
            if cluster_2 >= cluster_1:
                if cluster_1 == cluster_2:
                    data_cluster_uso_pair_1_raw = new_data_pfizer_uso[cluster_labels==cluster_1+1,]
                    data_cluster_uso_pair_2_raw = new_data_pfizer_uso[cluster_labels==cluster_2+1,]
                    indices_cells = np.random.permutation(sum(cluster_labels==cluster_1+1))
                    data_cluster_uso_pair_1_now = data_cluster_uso_pair_1_raw[indices_cells[:len(indices_cells)/2],]
                    data_cluster_uso_pair_2_now = data_cluster_uso_pair_2_raw[indices_cells[len(indices_cells)/2:],]
                else:
                    data_cluster_uso_pair_1_now = new_data_pfizer_uso[cluster_labels==cluster_1+1,]
                    data_cluster_uso_pair_2_now = new_data_pfizer_uso[cluster_labels==cluster_2+1,]

                # To avoid memory error use only little bit of data for now
                data_cluster_uso_pair_1 = data_cluster_uso_pair_1_now[:,indices[:genes_to_keep]]
                data_cluster_uso_pair_2 = data_cluster_uso_pair_2_now[:,indices[:genes_to_keep]]

                K_lin_uso_cluster_pair_1 = data_cluster_uso_pair_1.T.dot(data_cluster_uso_pair_1)
                K_lin_uso_cluster_pair_2 = data_cluster_uso_pair_2.T.dot(data_cluster_uso_pair_2)
                pairwise_KTAs[cluster_1][cluster_2] = kta_align_general(K_lin_uso_cluster_pair_1, K_lin_uso_cluster_pair_2)
                K_lin_uso_cluster_pair_1_center = center_kernel(K_lin_uso_cluster_pair_1)
                K_lin_uso_cluster_pair_2_center = center_kernel(K_lin_uso_cluster_pair_2)
                pairwise_KTAs_center[cluster_1][cluster_2] = kta_align_general(K_lin_uso_cluster_pair_1_center, K_lin_uso_cluster_pair_2_center)
                pairwise_KTAs_center_norma[cluster_1][cluster_2] = kta_align_general(normalize_kernel(K_lin_uso_cluster_pair_1_center), normalize_kernel(K_lin_uso_cluster_pair_2_center))

                labels_cluster_pair = np.concatenate([np.array([0] * len(data_cluster_uso_pair_1)), np.array([1] * len(data_cluster_uso_pair_2))])
                data_for_svm_cluster_pair = np.concatenate([data_cluster_uso_pair_1, data_cluster_uso_pair_2])

                if np.amin([len(labels_cluster_pair),10, sum(labels_cluster_pair),len(labels_cluster_pair)-sum(labels_cluster_pair)])>2:
                    accs_now, aucs_now = cross_validation(np.transpose(data_for_svm_cluster_pair), labels_cluster_pair, False)
                else:
                    print('Warning: too few training samples!')
                    accs_now = 0.5
                    aucs_now = 0.5
                pairwise_accuracies[cluster_1][cluster_2] = np.mean(accs_now)
                pairwise_aucs[cluster_1][cluster_2] =np.mean(aucs_now)
            else:
                pairwise_accuracies[cluster_1][cluster_2] = pairwise_accuracies[cluster_2][cluster_1]
                pairwise_aucs[cluster_1][cluster_2] = pairwise_aucs[cluster_2][cluster_1]
                pairwise_KTAs[cluster_1][cluster_2] = pairwise_KTAs[cluster_2][cluster_1]
                pairwise_KTAs_center[cluster_1][cluster_2] = pairwise_KTAs_center[cluster_2][cluster_1]
                pairwise_KTAs_center_norma[cluster_1][cluster_2] = pairwise_KTAs_center_norma[cluster_2][cluster_1]

    print 'Pairwise comparison of clusters - KTA scores (Pfizer + Usoskin): '
    print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in pairwise_KTAs]))
    print ''

    print 'Pairwise comparison of clusters - KTA scores centered (Pfizer + Usoskin): '
    print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in pairwise_KTAs_center]))
    print ''

    print 'Pairwise comparison of clusters - KTA scores centered and normalized (Pfizer + Usoskin): '
    print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in pairwise_KTAs_center_norma]))
    print ''

    print 'Pairwise comparison of clusters - Mean SVM accuracy (Pfizer + Usososkin): '
    print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in pairwise_accuracies]))
    print ''

    print 'Pairwise comparison of clusters - Mean SVM AUCs (Pfizer + Usososkin): '
    print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in pairwise_aucs]))
    print ''
    print '--------------------------------------------------------------------------'

    image_plot = plt.imshow(pairwise_accuracies)
    plt.figure(1)
    sp1 = plt.subplot(1, 5, 1)
    sp1.set_title('KTA')
    plt.imshow(pairwise_KTAs,cmap='Greys', interpolation='nearest')
    sp2 = plt.subplot(1, 5, 2)
    sp2.set_title('KTA centered')
    plt.imshow(pairwise_KTAs_center,cmap='Greys', interpolation='nearest')
    sp3 = plt.subplot(1, 5, 3)
    sp3.set_title('KTA centered and normalized')
    plt.imshow(pairwise_KTAs_center_norma,cmap='Greys', interpolation='nearest')
    sp4 = plt.subplot(1,5, 4)
    sp4.set_title('SVM accuracies')
    plt.imshow(pairwise_accuracies,cmap='Greys', interpolation='nearest')
    sp5 = plt.subplot(1,5, 5)
    sp5.set_title('SVM AUCs')
    plt.imshow(pairwise_aucs,cmap='Greys', interpolation='nearest')
    plt.show()
