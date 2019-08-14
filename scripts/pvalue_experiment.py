# import matplotlib.pyplot as plt

from nmf_clustering import NmfClustering, DaNmfClustering
from utils import *
import pdb


def delete_nonunique_genes(data, gene_ids):
    if not np.unique(gene_ids).size == gene_ids.size:
        _, index, counts = np.unique(gene_ids, return_index=True, return_counts=True)
        index_keep = [i for i, j in enumerate(index) if counts[i] == 1]
        data = data[index[index_keep], :]
        gene_ids = gene_ids[index[index_keep]]
    return data, gene_ids


def cell_filter(data, num_expr_genes=500, non_zero_threshold=0):
    """
    The cell filter returns indices of cells containing more than a specified number of non-zero (or > non_zero_threshold) genes.
    Per default minimum number of expressed genes is 2000.
    :param data: transcripts x cells data matrix
    :return: indices of valid cells
    """
    print(('SC3 cell filter with num_expr_genes={0} and non_zero_threshold={1}'.format(num_expr_genes, non_zero_threshold)))
    ai, bi = np.where(np.isnan(data))
    data[ai, bi] = 0
    res = np.sum(data > non_zero_threshold, axis=0)
    return np.where(np.isfinite(res) & (res >= num_expr_genes))[0]


def gene_filter(data, perc_consensus_genes=0.99, non_zero_threshold=0):
    """
    The gene filter returns indices of genes that are either expressed or absent in less than 94% of cells. (ubiquitous and rare genes are not informative)
    :param data: transcripts x cells data matrix
    :return: indices of valid transcripts
    """
    print(('SC3 gene filter with perc_consensus_genes={0} and non_zero_threshold={1}'.format(perc_consensus_genes, non_zero_threshold)))
    ai, bi = np.where(np.isnan(data))
    data[ai, bi] = 0
    _, num_cells = data.shape
    res = np.sum(data > non_zero_threshold, axis=1)
    lower_bound = np.float(num_cells)*(1.-perc_consensus_genes)
    upper_bound = np.float(num_cells)*perc_consensus_genes
    return np.where((res >= lower_bound) & (res <= upper_bound))[0]


def data_transformation_log2(data):
    """
    :param data: transcripts x cells data matrix
    :return: log2 transformed data
    """
    print('SC3 log2 data transformation.')
    return np.log2(data + 1.)

if __name__ == "__main__":
    path_src = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/Ting-data.tsv"
    path_geneids_src = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/Ting-geneids.tsv"
    path_trg = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/Usoskin-data.tsv"
    path_geneids_trg = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/Usoskin-geneids.tsv"

    #path_src = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/fout_source_data_excl_T1_200_S1_800.tsv"
    #path_geneids_src = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/fout_geneids_excl.tsv"
    #path_trg = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/fout_target_data_excl_T1_200_S1_800.tsv"
    #path_geneids_trg = "C:/Users/Bettina/PycharmProjects2/scRNA_new/scRNA/fout_geneids_excl.tsv"

    n_source_cluster = 16
    n_target_cluster = 16
    n_source = 1024
    n_target = 400

    # Load Source Data
    data_src = np.loadtxt(path_src)
    gene_ids_src = np.loadtxt(path_geneids_src, dtype=np.str)
    # Delete non-unique genes
    data_src, gene_ids_src = delete_nonunique_genes(data_src, gene_ids_src)
    # Apply cell filter
    valid_cells = cell_filter(data_src)
    # Apply gene filter
    valid_genes = gene_filter(data_src)

    # Create filtered data
    data_src = data_src[:, valid_cells]
    data_src = data_src[valid_genes, :]
    gene_ids_src = gene_ids_src[valid_genes]
    # Log transform data
    data_src = data_transformation_log2(data_src)

    # Load Target data
    data_trg = np.loadtxt(path_trg)
    gene_ids_trg = np.loadtxt(path_geneids_trg, dtype=np.str)
    # Delete non-unique genes
    data_trg, gene_ids_trg = delete_nonunique_genes(data_trg, gene_ids_trg)
    # Apply cell filter
    valid_cells = cell_filter(data_trg)
    # Apply gene filter
    valid_genes = gene_filter(data_trg)

    # Create filtered data
    data_trg = data_trg[:, valid_cells]
    data_trg = data_trg[valid_genes, :]
    gene_ids_trg = gene_ids_trg[valid_genes]
    # Log transform data
    data_trg = data_transformation_log2(data_trg)

    # train source and test performance
    source_nmf = NmfClustering(data_src, gene_ids_src, num_cluster=n_source_cluster)
    source_nmf.apply(k=n_source_cluster, max_iter=100, rel_err=1e-3)

    # Number of repetitions can be changed in line 153 of utils.py
    target_nmf = DaNmfClustering(source_nmf, data_trg.copy(), gene_ids_trg, num_cluster=n_target_cluster)
    target_nmf.apply(k=n_target_cluster, calc_transferability=True)
    # target_nmf.transferability_pvalue

    # np.savez(fname, source_ari=source_ari, target_ari=target_ari, n_mix=n_mix, n_source=n_source, n_target=n_target, n_source_cluster=n_source_cluster,
    # n_target_cluster=n_target_cluster)
