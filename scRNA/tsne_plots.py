import pdb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd


if __name__ == "__main__":

    # load Pfizer + Uso data
    # Pfizer + Usoskin
    data_pfizer_uso = np.loadtxt('C:\Users\Bettina\ml\scRNAseq\data\Pfizer Uso data\data_pfizer_uso.txt')
    data_array_pfizer_uso = np.transpose(np.nan_to_num(data_pfizer_uso))
    cell_names_pfizer_uso = np.loadtxt('C:\Users\Bettina\ml\scRNAseq\data\Pfizer Uso data\cell_names_pfizer_uso.txt', dtype='string')
    transcript_names_pfizer_uso = np.loadtxt('C:\Users\Bettina\ml\scRNAseq\data\Pfizer Uso data\\transcript_names_pfizer_uso.txt', dtype='string')

    # Split in Pfizer and Usoskin
    data_array_pfizer = data_array_pfizer_uso[0:417, ]
    data_array_uso = data_array_pfizer_uso[417:, ]

    # Load Cluster Results
    cluster_labels_inf = np.loadtxt("C:/Users/Bettina/ml/scRNAseq/Results/SC3 results/Pfizer + Uso data/cluster_4_labels_pfizer_uso_after_scaling.txt",
                                dtype={'names': ('cluster', 'cell_name'), 'formats': ('i2', 'S8')}, skiprows=1)

    cluster_labels_raw = cluster_labels_inf['cluster']
    cells_to_keep_raw = cluster_labels_inf['cell_name']
    cells_to_keep = cells_to_keep_raw[cells_to_keep_raw!='NA']
    cluster_labels = cluster_labels_raw[cells_to_keep_raw!='NA']

    new_data_pfizer_uso = np.asarray([data_array_pfizer_uso[cell_names_pfizer_uso.tolist().index(cell),] for cell in cells_to_keep])

    model = TSNE(n_components=2, random_state=0)
    tsne_comps = model.fit_transform(new_data_pfizer_uso)

    df = pd.DataFrame(dict(x=tsne_comps[:,0], y=tsne_comps[:,1], label=cluster_labels))

    groups = df.groupby('label')

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    ax.legend()

    plt.show()

    # pdb.set_trace()
