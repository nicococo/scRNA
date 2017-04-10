import unittest
import numpy as np

from functools import partial

from scRNA.nmf_clustering import NmfClustering, DaNmfClustering
from scRNA.sc3_clustering_impl import cell_filter, gene_filter


class MyTestCase(unittest.TestCase):

    def test_source_preprocessing(self):
        src = [
            [1,2, 0,0,0],
            [0,0,10,11,0],
            [0,0,0,0,0]
        ]
        src = np.array(src, dtype=np.float)
        print src

        ids = np.array(['lbl0','lbl1','lbl2'])
        nmf = NmfClustering(src, ids, 2)
        nmf.add_cell_filter(partial(cell_filter, num_expr_genes=1, non_zero_threshold=1))
        nmf.add_gene_filter(partial(gene_filter, perc_consensus_genes=0.96, non_zero_threshold=1))
        nmf.apply()

        print nmf.pp_data
        print nmf.remain_cell_inds
        print nmf.remain_gene_inds

        print nmf.cluster_labels
        # numpy array testing
        np.testing.assert_array_equal(nmf.pp_data, src[:2,:4])
        np.testing.assert_array_equal(nmf.remain_gene_inds, np.arange(2))
        np.testing.assert_array_equal(nmf.remain_cell_inds, np.arange(4))


    def test_target(self):
        src = [
            [1,2, 0,0,0],
            [0,0,10,11,0],
            [1,2,1,2,0],
            [0,0,0,0,0]
        ]
        src = np.array(src, dtype=np.float)
        trg = [
            [2,4,6,6,0],
            [2,1,2,2,0],
            [1.1,2.3,1.2,2.1,0],
            [0,0,0,0,0]
        ]
        trg = np.array(trg, dtype=np.float)

        ids = np.array(['lbl0','lbl1','lbl2','lbl3'])
        nmf = NmfClustering(src, ids, 2)
        nmf.add_cell_filter(partial(cell_filter, num_expr_genes=1, non_zero_threshold=1))
        nmf.add_gene_filter(partial(gene_filter, perc_consensus_genes=0.96, non_zero_threshold=1))
        nmf.apply()

        trg_ids = np.array(['lbl0','lbl1','lbl20','lbl3'])
        da_nmf = DaNmfClustering(nmf, trg, trg_ids, 2)
        da_nmf.add_cell_filter(partial(cell_filter, num_expr_genes=1, non_zero_threshold=1))
        da_nmf.add_gene_filter(partial(gene_filter, perc_consensus_genes=0.96, non_zero_threshold=1))
        mixed, _, _ = da_nmf.get_mixed_data(mix=0.)

        print '-------------'
        print da_nmf.src.pp_data
        print '-------------'
        print da_nmf.pp_data
        print '-------------'
        print mixed
        print '-------------'
        W, H, H2 = da_nmf.intermediate_model
        print W.dot(H)


        print '-------------'
        print da_nmf.remain_gene_inds
        print da_nmf.src.remain_gene_inds

        # print nmf.remain_cell_inds
        # print nmf.remain_gene_inds

        # print nmf.cluster_labels
        # numpy array testing
        # np.testing.assert_array_equal(nmf.pp_data, src[:2,:4])
        # np.testing.assert_array_equal(nmf.remain_gene_inds, np.arange(2))
        # np.testing.assert_array_equal(nmf.remain_cell_inds, np.arange(4))


if __name__ == '__main__':
    unittest.main()
