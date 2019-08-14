import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import decomposition as decomp

from scRNA.abstract_clustering import AbstractClustering
from scRNA.utils import center_kernel, normalize_kernel, kta_align_binary, \
    get_matching_gene_inds, get_transferred_data_matrix, get_transferability_score


class NmfClustering(AbstractClustering):
    num_cluster = -1
    dictionary = None
    data_matrix = None

    def __init__(self, data, gene_ids, num_cluster, labels):
        super(NmfClustering, self).__init__(data, gene_ids=gene_ids)
        self.num_cluster = num_cluster

    def apply(self, k=-1, alpha=1.0, l1=0.75, max_iter=100, rel_err=1e-3):
        if k == -1:
            k = self.num_cluster
        X = self.pre_processing()

        nmf = decomp.NMF(alpha=alpha, init='nndsvdar', l1_ratio=l1, max_iter=max_iter,
                         n_components=k, random_state=0, shuffle=True, solver='cd',
                         tol=rel_err, verbose=0)

        W = nmf.fit_transform(X)
        H = nmf.components_
        self.cluster_labels = np.argmax(nmf.components_, axis=0)

        if np.any(np.isnan(H)):
            raise Exception('H contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, k, l1, X.shape[0], X.shape[1]))
        if np.any(np.isnan(W)):
            raise Exception('W contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, k, l1, X.shape[0], X.shape[1]))

        # self.print_reconstruction_error(X, W, H)
        self.dictionary = W
        self.data_matrix = H

    def print_reconstruction_error(self, X, W, H):
        print(('  Elementwise absolute reconstruction error   : ', np.sum(np.abs(X - W.dot(H))) / np.float(X.size)))
        print(('  Fro-norm reconstruction error               : ', np.sqrt(np.sum((X - W.dot(H))*(X - W.dot(H)))) / np.float(X.size)))


class NmfClustering_initW(AbstractClustering):
    num_cluster = -1
    dictionary = None
    data_matrix = None

    def __init__(self, data, gene_ids, num_cluster, labels):
        super(NmfClustering_initW, self).__init__(data, gene_ids=gene_ids)
        self.num_cluster = num_cluster
        self.labels=labels

    def apply(self, k=-1, alpha=1.0, l1=0.75, max_iter=100, rel_err=1e-3):
        if k == -1:
            k = self.num_cluster
        X = self.pre_processing()

        fixed_W = pd.get_dummies(self.labels)
        fixed_W_t = fixed_W.T  # interpret W as H (transpose), you can only fix H, while optimizing W in the code. So we simply switch those matrices (invert their roles).
        learned_H_t, fixed_W_t_same, n_iter = decomp.non_negative_factorization(X.astype(np.float), n_components=k, init='custom', random_state=0, update_H=False, H=fixed_W_t.astype(np.float), alpha=alpha, l1_ratio=l1, max_iter=max_iter, shuffle=True, solver='cd',tol=rel_err, verbose=0)

        init_W = fixed_W_t_same.T
        init_H = learned_H_t.T

        nmf = decomp.NMF(alpha=alpha, init='custom',l1_ratio=l1, max_iter=max_iter, n_components=k, random_state=0, shuffle=True, solver='cd', tol=rel_err, verbose=0)
        W = nmf.fit_transform(X.T, W=init_W, H = init_H)
        H = nmf.components_
        self.cluster_labels = np.argmax(W, axis=1)

        if np.any(np.isnan(H)):
            raise Exception('H contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, k, l1, X.shape[0], X.shape[1]))
        if np.any(np.isnan(W)):
            raise Exception('W contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, k, l1, X.shape[0], X.shape[1]))

        # self.print_reconstruction_error(X, W, H)
        self.dictionary = H.T
        self.data_matrix = W.T

    def print_reconstruction_error(self, X, W, H):
        print(('  Elementwise absolute reconstruction error   : ', np.sum(np.abs(X - W.dot(H))) / np.float(X.size)))
        print(('  Fro-norm reconstruction error               : ', np.sqrt(np.sum((X - W.dot(H))*(X - W.dot(H)))) / np.float(X.size)))


class NmfClustering_fixW(AbstractClustering):
    num_cluster = -1
    dictionary = None
    data_matrix = None

    def __init__(self, data, gene_ids, num_cluster,labels):
        super(NmfClustering_fixW, self).__init__(data, gene_ids=gene_ids)
        self.num_cluster = num_cluster
        self.labels=labels

    def apply(self, k=-1, alpha=1.0, l1=0.75, max_iter=100, rel_err=1e-3):
        if k == -1:
            k = self.num_cluster
        X_t = self.pre_processing()
        X = X_t.T

        fixed_W = pd.get_dummies(self.labels)
        fixed_W_t = fixed_W.T  # interpret W as H (transpose), you can only fix H, while optimizing W in the code. So we simply switch those matrices (invert their roles).
        learned_H_t, fixed_W_t_same, n_iter = decomp.non_negative_factorization(X_t.astype(np.float), n_components=k, init='custom', random_state=0, update_H=False, H=fixed_W_t.astype(np.float), alpha=alpha, l1_ratio=l1, max_iter=max_iter, shuffle=True, solver='cd',tol=rel_err, verbose=0)

        assert(np.all(fixed_W_t == fixed_W_t_same))
        #self.cluster_labels = np.argmax(fixed_W_t_same.T, axis=1)

        # Now take the learned H, fix it and learn W to see how well it worked
        learned_W, learned_H_fix, n_iter = decomp.non_negative_factorization(X.astype(np.float), n_components=k, init='custom', random_state=0, update_H=False, H=learned_H_t.T, alpha=alpha, l1_ratio=l1, max_iter=max_iter, shuffle=True, solver='cd',tol=rel_err, verbose=0)

        assert(np.all(learned_H_t.T == learned_H_fix))
        self.cluster_labels = np.argmax(learned_W, axis=1)

        if np.any(np.isnan(learned_H_t)):
            raise Exception('H contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, k, l1, X.shape[0], X.shape[1]))
        if np.any(np.isnan(fixed_W_t)):
            raise Exception('W contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, k, l1, X.shape[0], X.shape[1]))

        #self.print_reconstruction_error(X, fixed_W_t, learned_H_t)
        self.dictionary = learned_H_t
        self.data_matrix = fixed_W_t


class DaNmfClustering(NmfClustering):
    reject = None
    transferability_score = 0.0
    transferability_percs = None
    transferability_rand_scores = None
    transferability_pvalue = 1.0
    src = None
    intermediate_model = None
    mixed_data = None

    def __init__(self, src, trg_data, trg_gene_ids, num_cluster):
        super(DaNmfClustering, self).__init__(trg_data, gene_ids=trg_gene_ids, num_cluster=num_cluster, labels=[])
        self.src = src

    def get_mixed_data(self, mix=0.0, reject_ratio=0., use_H2=True, calc_transferability=False, max_iter=100, rel_err=1e-3):
        trg_data = self.pre_processing()
        trg_gene_ids = self.gene_ids[self.remain_gene_inds]
        # print self.src.gene_ids.shape
        # print self.src.remain_gene_inds.shape
        src_gene_ids = self.src.gene_ids[self.src.remain_gene_inds].copy()
        inds1, inds2 = get_matching_gene_inds(src_gene_ids, trg_gene_ids)

        # print 'MTL source {0} genes -> {1} genes.'.format(src_gene_ids.size, inds2.size)
        # print 'MTL target {0} genes -> {1} genes.'.format(trg_gene_ids.size, inds1.size)

        src_gene_ids = src_gene_ids[inds2]
        self.gene_ids = trg_gene_ids[inds1]
        trg_data = trg_data[inds1, :]

        # print('Sorted, filtered gene ids for src/trg. They should coincide!')
        for i in range(inds1.size):
            #if i < 10 or src_gene_ids[i] != self.gene_ids[i]:
            #    print i, src_gene_ids[i], self.gene_ids[i]
            assert(src_gene_ids[i] == self.gene_ids[i])

        assert(self.src.dictionary is not None)  # source data should always be pre-processed
        W, H, H2, new_err = get_transferred_data_matrix(self.src.dictionary[inds2, :], trg_data, max_iter=max_iter, rel_err=rel_err)
        self.cluster_labels = np.argmax(H, axis=0)
        #self.print_reconstruction_error(trg_data, W, H2)
        self.intermediate_model = (W, H, H2)
        self.reject = self.calc_rejection(trg_data, W, H, H2)

        if calc_transferability:
            #print('Calculating transferability score...')
            self.transferability_score, self.transferability_rand_scores, self.transferability_pvalue = \
                get_transferability_score(W, H, trg_data, max_iter=max_iter)
            self.transferability_percs = np.percentile(self.transferability_rand_scores, [25, 50, 75, 100])
            self.reject.append(('Transfer_Percentiles', self.transferability_percs))
            self.reject.append(('Transferability', self.transferability_score))
            self.reject.append(('Transferability p-value', self.transferability_pvalue))

        if use_H2:
            new_trg_data = W.dot(H2)
        else:
            new_trg_data = W.dot(H)

        # reject option enabled?
        assert(reject_ratio < 1.)  # rejection of 100% (or more) does not make any sense

        if reject_ratio > 0.:
            name, neg_entropy = self.reject[2]
            # inds = np.arange(0, trg_data.shape[1], dtype=np.int)
            inds = np.argsort(-neg_entropy)  # ascending order
            keep = np.float(inds.size) * reject_ratio
            inds = inds[:keep]
            new_trg_data[:, inds] = trg_data[:, inds]

        mixed_data = mix*new_trg_data + (1.-mix)*trg_data
        if np.any(trg_data < 0.0):
            print('Error! Negative values in target data!')
        if np.any(mixed_data < 0.0):
            print('Error! Negative values in reconstructed data!')
        return mixed_data, new_trg_data, trg_data

    def apply(self, k=-1, mix=0.0, reject_ratio=0., alpha=1.0, l1=0.75, max_iter=100, rel_err=1e-3, calc_transferability=False):
        if k == -1:
            k = self.num_cluster
        mixed_data, new_trg_data, trg_data = self.get_mixed_data(mix=mix,
                                                                 reject_ratio=reject_ratio,
                                                                 max_iter=max_iter,
                                                                 rel_err=rel_err,
                                                                 calc_transferability=calc_transferability)
        nmf = decomp.NMF(alpha=alpha, init='nndsvdar', l1_ratio=l1, max_iter=max_iter,
                         n_components=k, random_state=0, shuffle=True, solver='cd', tol=1e-6, verbose=0)
        W = nmf.fit_transform(mixed_data)
        H = nmf.components_
        self.dictionary = W
        self.data_matrix = H
        self.cluster_labels = np.argmax(nmf.components_, axis=0)
        self.mixed_data = mixed_data
        # print('Labels used: {0} of {1}.'.format(np.unique(self.cluster_labels).size, k))

    def calc_rejection(self, trg_data, W, H, H2):
        diffs = np.zeros(H2.shape[1])
        for c in range(self.src.num_cluster):
            inds = np.where(self.cluster_labels == c)[0]
            if inds.size > 0:
                min_h2 = np.min(H[:, inds])
                max_h2 = np.max(H[:, inds])
                foo = H[:, inds]-min_h2 / (max_h2 - min_h2)
                foo = np.max(foo, axis=0) - np.min(foo, axis=0)
                diffs[inds] = foo

        sum_expr = np.sum(trg_data, axis=0)
        sum_expr -= np.min(sum_expr)
        sum_expr /= np.max(sum_expr)
        sum_expr = sum_expr + 1.0
        sum_expr /= np.max(sum_expr)
        weight = 1. - sum_expr

        reconstr_err = np.sum(np.abs(trg_data - W.dot(H2)), axis=0)
        reconstr_err -= np.min(reconstr_err)
        reconstr_err /= np.max(reconstr_err)

        final_values = weight * reconstr_err #* neg_entropy
        # final_values = reconstr_err #* neg_entropy

        reject = list()
        reject.append(('Reconstr. Error', -final_values))

        # kurts = stats.kurtosis(H, fisher=False, axis=0)
        # K1 = trg_data.T.dot(trg_data)
        # K2 = W.dot(H).T.dot(W.dot(H))
        # K3 = W.dot(H2).T.dot(W.dot(H2))

        neg_entropy = stats.entropy(H)
        neg_entropy -= np.min(neg_entropy)
        neg_entropy /= np.max(neg_entropy)

        reject.append(('Kurtosis', stats.kurtosis(H, fisher=False, axis=0)))
        reject.append(('Entropy', -neg_entropy))
        # reject.append(('KTA kurt1', self.reject_classifier(K1, diffs)))
        # reject.append(('KTA kurt2', self.reject_classifier(K2, kurts)))
        # reject.append(('KTA kurt3', self.reject_classifier(K3, kurts)))
        reject.append(('Diffs', diffs))
        reject.append(('Dist L2 H', -np.sum((np.abs(trg_data - W.dot(H))**2. ), axis=0)))
        reject.append(('Dist L2 H2', -np.sum((np.abs(trg_data - W.dot(H2))**2. ), axis=0)))
        reject.append(('Dist L1 H', -np.sum(np.abs(trg_data - W.dot(H)), axis=0)))
        reject.append(('Dist L1 H2', -np.sum(np.abs(trg_data - W.dot(H2)), axis=0)))
        return reject

    def reject_classifier(self, K, kurts):
        """
        :param K: numpy.array
        :param kurts: numpy.array
        :return: numpy.array
        """
        sinds = np.argsort(kurts)
        K = center_kernel(K)
        K = normalize_kernel(K)
        max_kta = -1.0
        max_kta_ind = -1
        for i in range(K.shape[1]-2):
            # 1. build binary label matrix
            labels = np.ones(kurts.size, dtype=np.int)
            labels[sinds[:i+1]] = -1
            kta = kta_align_binary(K, labels)
            if kta > max_kta:
                max_kta = kta
                max_kta_ind = i+1
        labels = np.ones(kurts.size, dtype=np.int)
        labels[sinds[:max_kta_ind]] = -1
        return labels
