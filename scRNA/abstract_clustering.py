from abc import ABCMeta, abstractmethod

import numpy as np
import pdb


class AbstractClustering(object):
    __metaclass__ = ABCMeta

    cell_filter_list = None
    gene_filter_list = None
    data_transf = None

    data = None
    gene_ids = None
    num_cells = -1
    num_transcripts = -1

    pp_data = None
    cluster_labels = None

    remain_cell_inds = None
    remain_gene_inds = None

    def __init__(self, data, gene_ids=None):
        # init lists
        self.cell_filter_list = list()
        self.gene_filter_list = list()
        self.data_transf = lambda x: x
        self.gene_ids = gene_ids
        self.data = data
        self.num_transcripts, self.num_cells = data.shape
        if self.gene_ids is None:
            #print('No gene ids provided.')
            self.gene_ids = np.arange(self.num_transcripts)
        self.cluster_labels = np.zeros((self.num_cells, 1))
        #print('Number of cells = {0}, number of transcripts = {1}'.format(self.num_cells, self.num_transcripts))

    def set_data_transformation(self, data_transf):
        self.data_transf = data_transf

    def add_cell_filter(self, cell_filter):
        if self.cell_filter_list is None:
            self.cell_filter_list = list(cell_filter)
        else:
            self.cell_filter_list.append(cell_filter)

    def add_gene_filter(self, gene_filter):
        if self.gene_filter_list is None:
            self.gene_filter_list = list(gene_filter)
        else:
            self.gene_filter_list.append(gene_filter)

    def pre_processing(self):
        self.pp_data, self.remain_gene_inds, self.remain_cell_inds = self.pre_processing_impl(self.data)
        return self.pp_data

    def pre_processing_impl(self, data):
        transcripts, cells = data.shape
        # 1. cell filter
        remain_cell_inds = np.arange(0, cells)

        for c in self.cell_filter_list:
            res = c(data)
            remain_cell_inds = np.intersect1d(remain_cell_inds, res)
        #print('1. Remaining number of cells after filtering: {0}/{1}'.format(remain_cell_inds.size, cells))
        A = data[:, remain_cell_inds]

        # 2. gene filter
        remain_gene_inds = np.arange(0, transcripts)
        for g in self.gene_filter_list:
            res = g(data)
            remain_gene_inds = np.intersect1d(remain_gene_inds, res)
        #print('2. Remaining number of transcripts after filtering: {0}/{1}'.format(remain_gene_inds.size, transcripts))

        # 3. data transformation
        B = A[remain_gene_inds, :]
        #print '3. Data transformation'
        #print 'Before data transformation: '
        #print '- Mean\median\max values: ', np.mean(B), np.median(B), np.max(B)
        #print '- Percentiles: ', np.percentile(B, [50, 75, 90, 99])
        X = self.data_transf(B)
        #print 'After data transformation: '
        #print '- Mean\median\max values: ', np.mean(X), np.median(X), np.max(X)
        #print '- Percentiles: ', np.percentile(X, [50, 75, 90, 99])
        return X, remain_gene_inds, remain_cell_inds

    @abstractmethod
    def apply(self):
        pass

    def __str__(self):
        if self.cluster_labels is None:
            return 'Empty cluster pipeline.'
        ret = 'Cluster Pipeline ({1} processed datapoints, {0} processed features):\n'.format(
            self.pp_data.shape[0], self.pp_data.shape[1])
        ret = '{0}-------------------------------------\n'.format(ret)
        lbls = np.unique(self.cluster_labels)
        for i in range(lbls.size):
            inds = np.where(self.cluster_labels == lbls[i])[0]
            ret = '{2}({1})[{0}'.format(inds[0], lbls[i], ret)
            for j in range(1, inds.size):
                ret = '{0},{1}'.format(ret, inds[j])
            ret = '{0}]\n'.format(ret)
            ret = '{0}-------------------------------------\n'.format(ret)
        return ret