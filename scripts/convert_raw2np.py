import pandas as pd
import numpy as np


def load_Zeisel(path):
    frame = pd.read_table(PATH + 'GSE60361_C1-3005-Expression.txt')
    table = frame.as_matrix()
    print table[:3,:10]

    name = 'Zeisel'
    encoding = 'Read Counts'
    transcripts = table[:, 0]
    cells = frame.columns.values[1:]
    data = np.array(table[:, 1:], dtype=np.float64)

    print '----------- Summary ------------'
    print 'General: ', name, encoding, data.shape, cells.shape
    print 'Statistics: ', np.min(data), np.mean(data), np.max(data)
    print 'Transcripts: ', transcripts.shape, transcripts[:4]
    print 'Cells: ', cells.shape, cells[:4]
    print 'Data:', data.shape, data[:4,:4]
    print '--------------------------------'
    return name, encoding, transcripts, cells, data


def load_Ting(path):
    frame = pd.read_table(PATH + 'GSE51372_readCounts.txt')
    table = frame.as_matrix()

    name = 'Ting'
    encoding = 'Read Counts'
    transcripts = table[:, 3]
    cells = frame.columns.values[6:]
    data = np.array(table[:, 6:], dtype=np.float64)

    print '----------- Summary ------------'
    print 'General: ', name, encoding, data.shape, cells.shape
    print 'Statistics: ', np.min(data), np.mean(data), np.max(data)
    print 'Transcripts: ', transcripts.shape, transcripts[:4]
    print 'Cells: ', cells.shape, cells[:4]
    print 'Data:', data.shape, data[:4,:4]
    print '--------------------------------'
    return name, encoding, transcripts, cells, data


def load_Usoskin(path):
    frame = pd.read_table(PATH + 'GSE59739_DataTable.txt')
    table = frame.as_matrix()

    name = 'Usoskin'
    encoding = 'RPM'
    transcripts = table[4:, 0]
    cells = frame.columns.values[1:]
    data = np.array(table[4:, 1:], dtype=np.float64)

    print '----------- Summary ------------'
    print 'General: ', name, encoding, data.shape, cells.shape
    print 'Statistics: ', np.min(data), np.mean(data), np.max(data)
    print 'Transcripts: ', transcripts.shape, transcripts[:4]
    print 'Cells: ', cells.shape, cells[:4]
    print 'Data:', data.shape, data[:4,:4]
    print '--------------------------------'
    return name, encoding, transcripts, cells, data


if __name__ == "__main__":
    PATH = '/Users/nicococo/Documents/scRNA-data/'

    datasets = ['Ting', 'Zeisel', 'Usoskin']
    props = []
    for i in range(len(datasets)):
        name, encoding, transcripts, cells, data = eval('load_{0}'.format(datasets[i]))(PATH)
        props.append([name, encoding, data.shape[0], data.shape[1]])
        np.savez('{0}{1}.npz'.format(PATH, name), name=name, encoding=encoding, transcripts=transcripts, cells=cells, data=data)
        # Save as tab-separated-values
        np.savetxt('{0}{1}-data.tsv'.format(PATH, name), data, delimiter='\t', fmt='%.4f')
        np.savetxt('{0}{1}-cellids.tsv'.format(PATH, name), cells, delimiter='\t', fmt='%s')
        np.savetxt('{0}{1}-geneids.tsv'.format(PATH, name), transcripts, delimiter='\t', fmt='%s')

    print props
    # name, encoding, transcripts, cells, data = load_Ting(PATH)
    # name, encoding, transcripts, cells, data = load_Zeisel(PATH)
    # name, encoding, transcripts, cells, data = load_Usoskin(PATH)