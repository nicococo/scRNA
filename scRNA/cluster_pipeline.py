import numpy as np

class ClusterPipeline(object):
    """ Meta-class for single-cell clustering based on the SC3 pipeline.
        Nico Goernitz, TU Berlin, 2016
    """
    cell_filter_list = None
    gene_filter_list = None
    data_transf = None
    dists_list  = None
    dimred_list = None
    intermediate_clustering = None
    consensus_clustering = None

    data = None
    num_cells = -1
    num_transcripts = -1

    def __init__(self, data):
        # init lists
        self.cell_filter_list = list()
        self.gene_filter_list = list()
        self.data_transf = lambda X: X
        self.dists_list = list()
        self.dimred_list = list()
        self.intermediate_clustering = lambda X: np.zeros(X.shape[0])
        self.consensus_clustering = lambda X: np.zeros(X.shape[0])

        self.data = data
        self.num_transcripts, self.num_cells = data.shape
        print('Number of cells = {0}, number of transcripts = {1}'.format(self.num_cells, self.num_transcripts))

    def set_intermediate_clustering(self, intermediate_clustering):
        self.intermediate_clustering = intermediate_clustering

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

    def apply(self, pc_range=[4, 10]):
        # 1. cell filter
        remain_cell_inds = np.arange(0, self.num_cells)
        for c in self.cell_filter_list:
            res = c(self.data)
            remain_cell_inds = np.intersect1d(remain_cell_inds, res)
        print('1. Remaining number of cells after filtering: {0}/{1}'.format(remain_cell_inds.size, self.num_cells))
        A = self.data[:, remain_cell_inds]

        # 2. gene filter
        remain_gene_inds = np.arange(0, self.num_transcripts)
        for g in self.gene_filter_list:
            res = g(self.data)
            remain_gene_inds = np.intersect1d(remain_gene_inds, res)
        print('2. Remaining number of transcripts after filtering: {0}/{1}'.format(remain_gene_inds.size, self.num_transcripts))

        # 3. data transformation
        B = A[remain_gene_inds, :]
        print '3. Data transformation'
        print 'Before data transformation: '
        print '- Mean\median\max values: ', np.mean(B), np.median(B), np.max(B)
        print '- Percentiles: ', np.percentile(B, [50, 75, 90, 99])
        X = self.data_transf(B)
        print 'After data transformation: '
        print '- Mean\median\max values: ', np.mean(X), np.median(X), np.max(X)
        print '- Percentiles: ', np.percentile(X, [50, 75, 90, 99])

        # 4. distance calculations
        print '4. Distance calculations ({0} methods).'.format(len(self.dists_list))
        dists = list()
        for d in self.dists_list:
            M = d(X)
            dists.append(M)

        # 5. transformations (dimension reduction)
        print '5. Distance transformations ({0} transformations * {1} distances = {2} in total).'.format(
            len(self.dimred_list), len(self.dists_list), len(self.dists_list)*len(self.dimred_list))
        transf = list()
        for d in dists:
            for t in self.dimred_list:
                dres, deigv = t(M)
                transf.append((dres, deigv))

        # 6. intermediate  clustering
        print '6. Intermediate clustering.'
        labels = list()
        for t in range(len(transf)):
            dres, deigv = transf[t]
            for d in range(pc_range[0], pc_range[1]+1):
                labels.append(self.intermediate_clustering(deigv[:, :d].reshape((deigv.shape[0], d))))

        # 7. consensus clustering
        print '7. Consensus clustering.'
        lbl = self.consensus_clustering(np.array(labels))