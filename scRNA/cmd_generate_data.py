import argparse, sys

from utils import *

# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--fout", help="Output filename", default='fout.npz', type=str)

parser.add_argument("--num_genes", help="Number of genes/transcripts per cell (default 5000)", default=5000, type = int)
parser.add_argument("--num_cells", help="Number of cells (default 400)", default=400, type = int)
parser.add_argument("--num_cluster", help="Number of cluster (default 5)", default=5, type = int)
parser.add_argument("--dir_cluster_size", help="Dirichlet parameter cluster size (default 1)", default=1, type = int)
parser.add_argument("--shape_pl", help="Shape power law (default 0.1)", default=0.1, type = float)
parser.add_argument("--upper_counts", help="Upper bound counts (default 300000)", default=300000, type = int)
parser.add_argument("--dir_counts", help="Dirichlet parameter counts (default 1)", default=1, type = int)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print arguments

# 1. GENERATE TOY DATA
print('\nGenerate artificial single-cell RNA-seq data.')
data, labels = generate_toy_data(num_genes=arguments.num_genes,
                                 num_cells= arguments.num_cells,
                                 num_clusters= arguments.num_cluster,
                                 dirichlet_parameter_cluster_size=arguments.dir_cluster_size,
                                 shape_power_law=arguments.shape_pl,
                                 upper_bound_counts=arguments.upper_counts,
                                 dirichlet_parameter_counts=arguments.dir_counts)
print 'Data dimension: ', data.shape

# 2. GENERATE GENE NAMES
print('Generating corresponding gene names.')
transcripts = np.arrange(arguments.num_genes)

# 3. SAVE RESULTS
print('Saving results to \'{0}\'.'.format(arguments.fout))
np.savez(arguments.fout, type='Toy', data=data, labels=labels, transcripts=transcripts)

print('Done.')
