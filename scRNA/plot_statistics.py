import numpy as np
import matplotlib.pyplot as plt


def plot_abundances(data):
    dmean = np.mean(data, axis=1)
    dstd = np.std(data, axis=1)
    dmax = np.max(data, axis=1)
    print dmax


    plt.figure(figsize=(20, 5.5))
    plt.subplot(1, 3, 1)
    plt.title('Mean Transcript Abundances ({0}/{1} nnz)'.format(np.sum(dmean>0.01), dmean.size))
    inds = np.argsort(-dmean)
    plt.plot(np.arange(dmean.size), dmean[inds], '-r', linewidth=4)
    plt.grid('on')
    plt.axis('tight')

    plt.subplot(1, 3, 2)
    plt.title('Max Transcript Abundances ({0}/{1} nnz)'.format(np.sum(dmax>0.01), dmax.size))
    inds = np.argsort(-dmax)
    plt.plot(np.arange(dmax.size), dmax[inds], '-b', linewidth=4)
    plt.grid('on')
    plt.axis('tight')

    plt.subplot(1, 3, 3)
    plt.title('Std Transcript Abundances ({0}/{1} nnz)'.format(np.sum(dstd>0.00001), dstd.size))
    inds = np.argsort(-dstd)
    plt.plot(np.arange(dstd.size), dstd[inds], '-g', linewidth=4)
    plt.grid('on')
    plt.axis('tight')

    plt.show()


if __name__ == "__main__":
    PATH = '/Users/nicococo/Documents/scRNA-data/'
    datasets = ['Ting', 'Zeisel', 'Usoskin']

    foo = np.load('{0}{1}.npz'.format(PATH, datasets[2]))
    cells = foo['cells']
    transcripts = foo['transcripts']
    data = foo['data']  # transcripts x cells
    name = foo['name']
    encoding = foo['encoding']

    print cells.shape
    print transcripts.shape
    print data.shape
    print encoding

    plot_abundances(data)

    foo = np.load('pfizer_data.npz')
    transcripts = foo['transcripts']
    data = foo['rpm_data']  # transcripts x cells
    filtered_inds = foo['filtered_inds']  # filtered inds of cells with enough coverage

    print transcripts.shape
    print data.shape

    plot_abundances(data[:, filtered_inds])