import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import sklearn.manifold as manifold
import numpy as np

import sc3_pipeline_impl as sc
from utils import *

if __name__ == "__main__":
    # foo = np.load('pfizer_data.npz')
    #
    # gene_names = np.loadtxt('/Users/nicococo/Documents/scRNA/pfizer/gene_names.txt', skiprows=1, dtype='object')
    #
    # print gene_names.shape
    # print np.unique(gene_names[:, 0]).shape, np.unique(gene_names[:, 1]).shape
    #
    # gene_id_map = dict()
    # for i in range(gene_names.shape[0]):
    #     gene_id_map[gene_names[i, 1]] = gene_names[i, 0]

    # dataset = 'Ting'
    print("Starting...")
    dataset = 'Pfizer'
    # dataset = 'Usoskin'
    data, gene_ids = load_dataset_by_name(dataset)

    # foo = np.load('/Users/nicococo/Documents/scRNA-data/Usoskin.npz')
    # foo = np.load('/Users/nicococo/Documents/scRNA-data/Ting.npz')
    # data = foo['data']  # transcripts x cells

    # filtered_inds=foo['filtered_inds']
    # data  = foo['data']
    # rpm_data = foo['rpm_data']
    # transcripts = foo['transcripts']

    # print transcripts[:10,0]
    # num_transcripts = transcripts.shape[0]
    # print num_transcripts, np.unique(transcripts[:,0]).shape[0]

    # transcripts_header = foo['transcripts_header']
    # xlsx_data = foo['xlsx_data']
    # xlsx_header = foo['xlsx_header']

    num_transcripts, num_cells = data.shape
    remain_inds = np.arange(0, num_cells)

    res = sc.cell_filter(data, num_expr_genes=2000, non_zero_threshold=1)
    remain_inds = np.intersect1d(remain_inds, res)
    A = data[:, remain_inds]

    remain_inds = np.arange(0, num_transcripts)
    res = sc.gene_filter(data, perc_consensus_genes=0.98, non_zero_threshold=1)
    remain_inds = np.intersect1d(remain_inds, res)
    X = sc.data_transformation(A[remain_inds, :])
    print X.shape, np.min(X), np.max(X)
    num_transcripts, num_cells = X.shape

    nmf = decomp.NMF(alpha=10.1, init='nndsvdar', l1_ratio=0.9, max_iter=1000,
        n_components=10, random_state=0, shuffle=True, solver='cd', tol=0.00001, verbose=0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    print nmf.reconstruction_err_

    print nmf
    print 'Absolute elementwise reconstruction error: ', np.sum(np.abs(X - W.dot(H)))/np.float(X.size)
    print 'Fro-norm reconstruction error: ', np.sqrt(np.sum((X - W.dot(H))*(X - W.dot(H))))
    print 'dim(W): ', W.shape
    print 'dim(H): ', H.shape

    # plt.figure(1)
    # plt.imshow(W)
    plt.figure(2)

    labels = np.argmax(H, axis=0)
    print np.unique(labels)
    inds = np.argsort(labels)

    A = np.zeros((num_cells, num_cells))
    for j in range(num_cells):
        A[:, j] = (labels[inds] == labels[inds[j]])
        A[j, :] = (labels[inds] == labels[inds[j]])

    dists =sc.distances(X[:, inds], None, metric='spearman')
    # plt.imshow(sc.distances(X[:, inds], None, metric='euclidean'), cmap='bwr')
    O = H.copy()
    O /= np.repeat(np.sum(O, axis=0).reshape((1, O.shape[1])), O.shape[0], axis=0)
    plt.imshow(O.T.dot(O), cmap='rainbow')

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.imshow(X[:1000, :])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Cells')
    plt.ylabel('Transcripts')
    plt.title('Data')

    plt.subplot(1, 2, 2)
    plt.imshow(W.dot(H)[:1000, :])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Cells')
    plt.ylabel('Transcripts')
    plt.title('Reconstruction')


    plt.figure(4)
    #
    # plt.subplot(1, 2, 1)
    # model = manifold.TSNE(n_components=2, perplexity=37, init='pca')
    # np.set_printoptions(suppress=True)
    # out = model.fit_transform(dists)
    # print out.shape
    #
    # plt.scatter(out[:, 0], out[:, 1], s=20, c=labels)


    # plt.subplot(1, 2, 2)
    model = manifold.TSNE(n_components=2, perplexity=37, init='pca')
    out = model.fit_transform(W.dot(H).T)
    print out.shape

    plt.scatter(out[:, 0], out[:, 1], s=20, c=labels)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('t-SNE (colonic dataset)')

    plt.show()


    print('Done.')
