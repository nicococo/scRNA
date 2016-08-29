import numpy as np

from abstract_clustering_pipeline import AbstractClusteringPipeline


class SC3Pipeline(AbstractClusteringPipeline):
    """ Meta-class for single-cell clustering based on the SC3 pipeline.
        Nico Goernitz, TU Berlin, 2016
    """
    dists_list  = None
    dimred_list = None
    intermediate_clustering_list = None
    build_consensus_matrix = None
    consensus_clustering = None

    dists = None
    pc_range = None
    sub_sample = None
    consensus_mode = None

    def __init__(self, data, gene_ids=None,
                 pc_range=[4, 10], sub_sample=True, consensus_mode=0):
        super(SC3Pipeline, self).__init__(data, gene_ids=gene_ids)
        # init lists
        self.dists_list = list()
        self.dimred_list = list()
        self.intermediate_clustering_list = list()
        self.consensus_clustering = lambda X: np.zeros(X.shape[0])
        self.pc_range = pc_range
        self.sub_sample = sub_sample
        self.consensus_mode = consensus_mode

    def set_consensus_clustering(self, consensus_clustering):
        self.consensus_clustering = consensus_clustering

    def set_build_consensus_matrix(self, build_consensus_matrix):
        self.build_consensus_matrix = build_consensus_matrix

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

    def apply(self):
        # check range
        assert self.pc_range[0] > 0
        assert self.pc_range[1] < self.num_cells

        X = self.pre_processing()

        # 4. distance calculations
        print '4. Distance calculations ({0} methods).'.format(len(self.dists_list))
        dists = list()
        for d in self.dists_list:
            dists.append(d(X, self.gene_ids[self.remain_gene_inds]))

        # 5. transformations (dimension reduction)
        print '5. Distance transformations ({0} transformations * {1} distances = {2} in total).'.format(
            len(self.dimred_list), len(self.dists_list), len(self.dists_list)*len(self.dimred_list))
        transf = list()
        for d in dists:
            for t in self.dimred_list:
                dres, deigv = t(d)
                transf.append((dres, deigv))

        # 6. intermediate  clustering and consensus matrix generation
        print '6. Intermediate clustering and consensus matrix generation.'
        range_inds = range(self.pc_range[0], self.pc_range[1]+1)
        if self.sub_sample and len(range_inds) > 15:
            # subsample 15 inds from this range
            range_inds = np.random.permutation(range_inds)[:15]
            print 'Subsample 15 eigenvectors for intermediate clustering: ', range_inds
        else:
            print('Using complete range of eigenvectors from {0} to {1}.'.format(
                self.pc_range[0], self.pc_range[1]))

        cnt = 0.
        consensus2 = np.zeros((self.remain_cell_inds.size, self.remain_cell_inds.size))
        for cluster in self.intermediate_clustering_list:
            for t in range(len(transf)):
                _, deigv = transf[t]

                labels = list()
                for d in range_inds:
                    labels.append(cluster(deigv[:, 0:d].reshape((deigv.shape[0], d))))
                    if self.consensus_mode == 0:
                        consensus2 += self.build_consensus_matrix(np.array(labels[-1]))
                        cnt += 1.

                if self.consensus_mode == 1:
                    consensus2 += self.build_consensus_matrix(np.array(labels))
                    cnt += 1.
        consensus2 /= cnt

        # 7. consensus clustering
        print '7. Consensus clustering.'
        self.cluster_labels, self.dists = self.consensus_clustering(consensus2)
