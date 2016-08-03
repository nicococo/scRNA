import numpy as np
import pdb


class SC3Pipeline(object):
    """ Meta-class for single-cell clustering based on the SC3 pipeline.
        Nico Goernitz, TU Berlin, 2016
    """
    cell_filter_list = None
    gene_filter_list = None
    data_transf = None
    dists_list  = None
    dimred_list = None
    intermediate_clustering_list = None
    consensus_clustering = None

    gene_ids = None
    data = None
    num_cells = -1
    num_transcripts = -1

    cluster_labels = None
    remain_cell_inds = None
    remain_gene_inds = None
    filtered_transf_data = None

    consensus = None
    dists = None
    intermediate_label_matrix = None

    def __init__(self, data, gene_ids=None):
        # init lists
        self.cell_filter_list = list()
        self.gene_filter_list = list()
        self.data_transf = lambda X: X
        self.dists_list = list()
        self.dimred_list = list()
        self.intermediate_clustering_list = list()
        self.consensus_clustering = lambda X: np.zeros(X.shape[0])

        self.gene_ids = gene_ids
        self.data = data
        self.num_transcripts, self.num_cells = data.shape
        if self.gene_ids is None:
            print('No gene ids provided.')
            self.gene_ids = np.arange(self.num_transcripts)

        self.cluster_labels = np.zeros((self.num_cells, 1))
        print('Number of cells = {0}, number of transcripts = {1}'.format(self.num_cells, self.num_transcripts))


    def set_consensus_clustering(self, consensus_clustering):
        self.consensus_clustering = consensus_clustering


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


    def add_distance_calculation(self, dist_calculation):
        if self.dists_list is None:
            self.dists_list = list(dist_calculation)
        else:
            self.dists_list.append(dist_calculation)


    def add_dimred_calculation(self, dimred_computation):
        if self.dimred_list is None:
            self.dimred_list = list(dimred_computation)
        else:
            self.dimred_list.append(dimred_computation)


    def add_intermediate_clustering(self, intermediate_clustering):
        if self.intermediate_clustering_list is None:
            self.intermediate_clustering_list = list(intermediate_clustering)
        else:
            self.intermediate_clustering_list.append(intermediate_clustering)


    def __str__(self):
        if self.cluster_labels is None:
            return 'Empty cluster pipeline.'
        ret = 'Cluster Pipeline ({1} processed datapoints, {0} processed features):\n'.format(
            self.filtered_transf_data.shape[0], self.filtered_transf_data.shape[1])
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


    def apply(self, pc_range=[4, 10]):
        # 1. cell filter
        self.remain_cell_inds = np.arange(0, self.num_cells)
        for c in self.cell_filter_list:
            res = c(self.data)
            self.remain_cell_inds = np.intersect1d(self.remain_cell_inds, res)
        print('1. Remaining number of cells after filtering: {0}/{1}'.format(self.remain_cell_inds.size, self.num_cells))
        A = self.data[:, self.remain_cell_inds]

        # 2. gene filter
        self.remain_gene_inds = np.arange(0, self.num_transcripts)
        for g in self.gene_filter_list:
            res = g(self.data)
            self.remain_gene_inds = np.intersect1d(self.remain_gene_inds, res)
        print('2. Remaining number of transcripts after filtering: {0}/{1}'.format(self.remain_gene_inds.size, self.num_transcripts))

        # 3. data transformation
        B = A[self.remain_gene_inds, :]
        gene_ids = self.gene_ids[self.remain_gene_inds]
        print '3. Data transformation'
        print 'Before data transformation: '
        print '- Mean\median\max values: ', np.mean(B), np.median(B), np.max(B)
        print '- Percentiles: ', np.percentile(B, [50, 75, 90, 99])
        X = self.data_transf(B)
        self.filtered_transf_data = X.copy()
        print 'After data transformation: '
        print '- Mean\median\max values: ', np.mean(X), np.median(X), np.max(X)
        print '- Percentiles: ', np.percentile(X, [50, 75, 90, 99])

        # 4. distance calculations
        print '4. Distance calculations ({0} methods).'.format(len(self.dists_list))
        dists = list()
        for d in self.dists_list:
            M = d(X, gene_ids)
            dists.append(M)

        # 5. transformations (dimension reduction)
        print '5. Distance transformations ({0} transformations * {1} distances = {2} in total).'.format(
            len(self.dimred_list), len(self.dists_list), len(self.dists_list)*len(self.dimred_list))
        transf = list()
        for d in dists:
            for t in self.dimred_list:
                dres, deigv = t(d)
                transf.append((dres, deigv))

        # 6. intermediate  clustering
        print '6. Intermediate clustering.'
        labels = list()
        for cluster in self.intermediate_clustering_list:
            for t in range(len(transf)):
                _, deigv = transf[t]
                range_inds = range(pc_range[0], pc_range[1]+1)
                if len(range_inds) > 15:
                    # subsample 15 inds from this range
                    range_inds = np.random.permutation(range_inds)[:15]
                for d in range_inds:
                    labels.append(cluster(deigv[:, :d].reshape((deigv.shape[0], d))))

        print '\nrange inds:\n', range_inds

        # 7. consensus clustering
        print '7. Consensus clustering.'
        self.intermediate_label_matrix = np.array(labels)
        lbl, consensus, dists  = self.consensus_clustering(np.array(labels))
        self.cluster_labels = lbl
        self.consensus = consensus
        self.dists = dists