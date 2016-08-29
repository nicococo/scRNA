import numpy as np
import sklearn.decomposition as decomp
import scipy.stats as stats

from abstract_clustering_pipeline import AbstractClusteringPipeline
from utils import kta_align_binary, normalize_kernel, center_kernel


class DaNmfPipeline(AbstractClusteringPipeline):
    src_gene_ids = None
    src_data = None
    src_pp_data = None

    src_cluster_labels = None
    src_remain_cell_inds = None
    src_remain_gene_inds = None

    common_ids = None
    src_common_gene_inds = None
    trg_common_gene_inds = None
    trg_reject_option = None

    def __init__(self, src_data, src_gene_ids, trg_data, trg_gene_ids):
        super(DaNmfPipeline, self).__init__(trg_data, gene_ids=trg_gene_ids)
        self.src_data = src_data
        self.src_gene_ids = src_gene_ids

    def src_pre_processing(self):
        self.src_pp_data, self.src_remain_gene_inds, self.src_remain_cell_inds = self.pre_processing_impl(self.src_data)
        return self.src_pp_data

    def apply(self, trg_k=3, src_k=3, alpha=1.0, l1=0.75, max_iter=4000, rel_err=1e-3):
        src_data = self.src_pre_processing()
        trg_data = self.pre_processing()

        src_gene_ids = self.src_gene_ids[self.src_remain_gene_inds]
        trg_gene_ids = self.gene_ids[self.remain_gene_inds]

        # target data (filtered gene ids) NMF clustering
        nmf = decomp.NMF(alpha=alpha, init='nndsvdar', l1_ratio=l1, max_iter=1000,
                         n_components=trg_k, random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
        W = nmf.fit_transform(trg_data)
        self.trg_standalone_cluster_labels = np.argmax(nmf.components_, axis=0) + src_k

        if not np.unique(src_gene_ids).size == src_gene_ids.size:
            # raise Exception('(MTL) Gene ids are supposed to be unique.')
            print('\nError! (MTL gene ids) Gene ids are supposed to be unique. '
                  'Only {0} of {1}  entries are unique.'.format(np.unique(src_gene_ids).shape[0], src_gene_ids.shape[0]))
            print('Only first occurance will be used.\n')
        if not np.unique(trg_gene_ids).size == trg_gene_ids.size:
            # raise Exception('(Target) Gene ids are supposed to be unique.')
            print('\nError! (Target gene ids) Gene ids are supposed to be unique. '
                  'Only {0} of {1}  entries are unique.'.format(np.unique(trg_gene_ids).shape[0], trg_gene_ids.shape[0]))
            print('Only first occurance will be used.\n')

        common_ids = np.intersect1d(trg_gene_ids, src_gene_ids)
        print('Both datasets have (after processing) {0} gene ids in common.'.format(common_ids.shape[0]))

        # find indices of common_ids in pgene_ids and gene_ids
        inds1 = np.zeros(common_ids.shape[0], dtype=np.int)
        inds2 = np.zeros(common_ids.shape[0], dtype=np.int)
        for i in range(common_ids.shape[0]):
            inds1[i] = np.where(common_ids[i] == trg_gene_ids)[0][0]
            inds2[i] = np.where(common_ids[i] == src_gene_ids)[0][0]

        print 'MTL source {0} genes -> {1} genes.'.format(src_gene_ids.size, inds2.size)
        print 'MTL target {0} genes -> {1} genes.'.format(trg_gene_ids.size, inds1.size)

        self.common_ids = common_ids
        self.src_common_gene_inds = inds2
        self.trg_common_gene_inds = inds1

        src_data = src_data[inds2, :]
        trg_data = trg_data[inds1, :]

        # Source data (common gene ids) NMF clustering
        nmf = decomp.NMF(alpha=alpha, init='nndsvdar', l1_ratio=l1, max_iter=1000,
                         n_components=src_k, random_state=0, shuffle=True, solver='cd',
                         tol=0.00001, verbose=0)
        W = nmf.fit_transform(src_data)
        Hsrc = nmf.components_
        self.src_cluster_labels = np.argmax(Hsrc, axis=0)

        # check solution: if regularizer is too strong this can result in 'NaN's
        if np.any(np.isnan(W)):
            raise Exception('W contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, src_k, l1, src_data.shape[0], src_data.shape[1]))
        if np.any(np.isnan(Hsrc)):
            raise Exception('Hsrc contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, src_k, l1, src_data.shape[0], src_data.shape[1]))

        H = np.random.randn(src_k, trg_data.shape[1])
        a1, a2 = np.where(H < 0.)
        H[a1, a2] *= -1.

        a1, a2 = np.where(H < 1e-10)
        H[a1, a2] = 1e-10

        n_iter = 0
        err = 1e10
        while n_iter < max_iter:
            n_iter += 1
            if np.any(W.T.dot(W.dot(H))==0.):
                raise Exception('DA target nmf: division by zero.')
            H *= W.T.dot(trg_data) / W.T.dot(W.dot(H))
            new_err = np.sum(np.abs(trg_data - W.dot(H))) / np.float(trg_data.size)  # absolute
            # new_err = np.sqrt(np.sum((Xtrg - W.dot(H))*(Xtrg - W.dot(H)))) / np.float(Xtrg.size)  # frobenius
            if np.abs((err - new_err) / err) <= rel_err and err > new_err:
                break
            err = new_err
        print '  Number of iterations for reconstruction     : ', n_iter
        print '  Elementwise absolute reconstruction error   : ', np.sum(np.abs(trg_data - W.dot(H))) / np.float(trg_data.size)
        print '  Fro-norm reconstruction error               : ', np.sqrt(np.sum((trg_data - W.dot(H))*(trg_data - W.dot(H)))) / np.float(trg_data.size)

        if np.any(np.isnan(H)):
            raise Exception('Htrg contains NaNs (alpha={0}, k={1}, l1={2}, data={3}x{4}'.format(
                alpha, src_k, l1, trg_data.shape[0], trg_data.shape[1]))

        H2 = np.zeros((src_k, trg_data.shape[1]))
        H2[(np.argmax(H, axis=0), np.arange(trg_data.shape[1]))] = 1
        # H2[ (np.argmax(H, axis=0), np.arange(Xtrg.shape[1])) ] = np.sum(H, axis=0)
        self.cluster_labels = np.argmax(H, axis=0)

        print '  H2 Elementwise absolute reconstruction error: ', np.sum(np.abs(trg_data - W.dot(H2))) / np.float(trg_data.size)
        print '  H2 Fro-norm reconstruction error            : ', np.sqrt(np.sum((trg_data - W.dot(H2))*(trg_data - W.dot(H2)))) / np.float(trg_data.size)

        diffs = np.zeros(H2.shape[1])
        for c in range(src_k):
            inds = np.where(self.cluster_labels == c)[0]
            if inds.size > 0:
                min_h2 = np.min(H[:, inds])
                max_h2 = np.max(H[:, inds])
                foo = H[:, inds]-min_h2 / (max_h2 - min_h2)
                foo = np.max(foo, axis=0) - np.min(foo, axis=0)
                diffs[inds] = foo

        kurts = stats.kurtosis(H, fisher=False, axis=0)
        K1 = trg_data.T.dot(trg_data)
        K2 = W.dot(H).T.dot(W.dot(H))
        K3 = W.dot(H2).T.dot(W.dot(H2))

        reject = list()
        reject.append(('kurtosis', stats.kurtosis(H, fisher=False, axis=0)))
        reject.append(('KTA kurt1', self.reject_classifier(K1, diffs)))
        reject.append(('KTA kurt2', self.reject_classifier(K2, kurts)))
        reject.append(('KTA kurt3', self.reject_classifier(K3, kurts)))
        reject.append(('Diffs', diffs))
        reject.append(('Dist L2 H', -np.sum((np.abs(trg_data - W.dot(H))**2. ), axis=0)))
        reject.append(('Dist L2 H2', -np.sum((np.abs(trg_data - W.dot(H2))**2. ), axis=0)))
        reject.append(('Dist L1 H', -np.sum(np.abs(trg_data - W.dot(H)), axis=0)))
        reject.append(('Dist L1 H2', -np.sum(np.abs(trg_data - W.dot(H2)), axis=0)))
        self.trg_reject_option = reject

    def reject_classifier(self, K, kurts):
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
