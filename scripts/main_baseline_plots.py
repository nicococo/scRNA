import numpy as np


def plot_overlapping_cluster(fig_num, res, accs_desc, m_desc, percs, genes, n_src, n_trg, common):
    ind_genes = 0
    ind_src = 0
    ind_percs = 4
    print n_src, ind_src

    ari_1 = np.mean(res[ind_src, ind_genes, :, 1, :, ind_percs, 0], axis=1)
    print ari_1
    ari_1 = np.mean(res[ind_src, ind_genes, :, 1, :, ind_percs, 1], axis=1)
    print ari_1
    ari_1 = np.mean(res[ind_src, ind_genes, :, 1, :, ind_percs, 2], axis=1)
    print ari_1

    print ''



if __name__ == "__main__":
    foo = np.load('intermediate.npz')
    # methods = foo['methods']
    # acc_funcs = foo['acc_funcs']
    res = foo['res']  # n_src x genes x common x acc_funcs x reps x percs x methods
    accs_desc = foo['accs_desc']
    method_desc = foo['method_desc']
    percs = foo['percs']
    # reps = foo['reps']
    genes = foo['genes']
    n_src = foo['n_src']
    n_trg = foo['n_trg']
    common = foo['common']
    print 'n_src x genes x common x acc_funcs x reps x percs x methods'
    print 'Result dimensionality: ', res.shape

    # Plot experiment results
    plot_overlapping_cluster(1, res, accs_desc, method_desc, percs, genes, n_src, n_trg, common)


    print('Done')