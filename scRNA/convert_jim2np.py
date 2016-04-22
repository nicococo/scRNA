import pandas as pd
import numpy as np


def load_excel_description(fname):
    frame = pd.read_excel(fname)
    table = frame.as_matrix()
    print '-------------------'
    print frame.columns.values
    print table[0, :]
    return table, frame.columns.values


def load_counts(fname):
    frame = pd.read_csv(fname, sep='\t', skiprows=1)
    table = frame.as_matrix()

    transcripts_header = frame.columns.values[:6]
    transcripts = table[:, :6]
    data = np.array(table[:, 6:], dtype=np.int)

    print '-----------------------------'
    print 'Transcript header: ', transcripts_header
    print 'Transcript sample: ', transcripts[2, :]
    print 'Data shape       : ', data.shape
    print 'Data sample      : ', data[0, :]
    print '-----------------------------'
    return data, transcripts, transcripts_header


def filter_low_coverage_cells(X, threshold=0):
    # Assume X \in N^{Transcripts x Cells}
    transcripts, cells = X.shape
    inds = np.where(np.sum(X, axis=0) > threshold)[0]
    print 'Filtered {0}/{1} cells.'.format(cells-inds.size, cells)
    return inds


def readcounts_to_rpm(X):
    # Assume X \in N^{Transcripts x Cells}
    transcripts, cells = X.shape
    per_mio = np.array(np.sum(X, axis=0), dtype=np.float) / 1000000.
    # print per_mio
    # print np.min(per_mio)
    # print np.sort(np.sum(X, axis=0))
    # print np.where(np.sum(X, axis=0) <= 1e-20)[0]
    div = np.repeat(per_mio[np.newaxis, :], transcripts, axis=0)
    return np.array(X, dtype=np.float) / div


if __name__ == "__main__":
    PATH = '/Users/nicococo/Documents/scRNA/pfizer/'
    XSL_FILE = '{0}Single-cellRNAseqMetaData_All.xlsx'.format(PATH)
    CNT_FILE = '{0}counts.txt'.format(PATH)

    xlsx_data, xlsx_header = load_excel_description(XSL_FILE)
    data, transcripts, transcripts_header = load_counts(CNT_FILE)

    filtered_inds = filter_low_coverage_cells(data)
    rpm_data = readcounts_to_rpm(data)
    print rpm_data[:4,:8]

    np.savez_compressed('pfizer_data.npz', filtered_inds=filtered_inds, data=data, rpm_data=rpm_data,
                        transcripts=transcripts, transcripts_header=transcripts_header,
                        xlsx_data=xlsx_data, xlsx_header=xlsx_header)

    print('Done.')