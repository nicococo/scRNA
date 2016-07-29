import argparse
import sys
from simulation import generate_toy_data, split_source_target
from utils import *

# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--fout_target_data", help="Output filename target data", default='fout_target_data.tsv', type=str)
parser.add_argument("--fout_source_data", help="Output filename source data", default='fout_source_data.tsv', type=str)
parser.add_argument("--fout_geneids", help="Output filename geneids", default='fout_geneids.tsv', type=str)
parser.add_argument("--fout_target_labels", help="Output filename target labels", default='fout_target_labels.tsv', type=str)
parser.add_argument("--fout_source_labels", help="Output filename source labels", default='fout_source_labels.tsv', type=str)

parser.add_argument("--num_genes", help="Number of genes/transcripts per cell (default 10000)", default=10000, type = int)
parser.add_argument("--num_cells", help="Number of cells (default 1000)", default=1000, type = int)
parser.add_argument("--num_cluster", help="Number of clusters (default 4)", default=4, type = int)
parser.add_argument("--generation_mode", help="How are total counts generated? 1 = Power law, 2 = Negative Binomial Distribution, 3 = simulation from Len et al. (default 3)", default=3, type = int)
parser.add_argument("--dir_cluster_size", help="Dirichlet parameter cluster size (default 10)", default=10, type = int)
parser.add_argument("--shape_pl", help="Shape power law (default 0.1)", default=0.1, type = float)
parser.add_argument("--upper_counts", help="Upper bound counts (default 1000000)", default=1000000, type = int)
parser.add_argument("--dir_counts", help="Dirichlet parameter counts (default 0.05)", default=0.05, type = int)
parser.add_argument("--proportion_target", help="How much of data will be target data? This will not be exact for splitting mode 3 and 4! (default 0.4)", default=0.4, type = float)
parser.add_argument("--splitting_mode", help="Splitting mode, 1 = split randomly, 2 = split randomly, but stratified, 3 = Have some overlapping and some exclusive clusters, 4 = Have only exclusive clusters (default 2)", default=2, type = int)
parser.add_argument("--binomial_parameter", help="Parameter p of negative binomial distribution, between 0-1 (default 1e-05)", default=1e-05, type = float)

arguments = parser.parse_args(sys.argv[1:])
print('Command line arguments:')
print arguments

# 1. GENERATE TOY DATA
print('\nGenerate artificial single-cell RNA-seq data.')
data, labels = generate_toy_data(num_genes=arguments.num_genes,
                                 num_cells= arguments.num_cells,
                                 num_clusters= arguments.num_cluster,
                                 dirichlet_parameter_cluster_size=arguments.dir_cluster_size,
                                 mode= arguments.generation_mode,
                                 shape_power_law=arguments.shape_pl,
                                 upper_bound_counts=arguments.upper_counts,
                                 dirichlet_parameter_counts=arguments.dir_counts,
                                 binomial_parameter=arguments.binomial_parameter)
print 'Data dimension: ', data.shape

# 2. SPLIT TOY DATA IN TARGET AND SOURCE DATA
print('\nSplit artificial single-cell RNA-seq data in target and source data.')
data_source, data_target, true_labels_source, true_labels_target = split_source_target(data, labels, proportion_target=arguments.proportion_target,
                                                                                       mode=arguments.splitting_mode)
print 'Target data dimension: ', data_target.shape
print 'Source data dimension: ', data_source.shape

# 3. GENERATE GENE AND CELL NAMES
print('Generating corresponding gene names.')
transcripts = np.arange(arguments.num_genes)

# 4. SAVE RESULTS
print('Saving target data to \'{0}\'.'.format(arguments.fout_target_data))
np.savetxt(arguments.fout_target_data, data_target, fmt='%u', delimiter='\t')
np.savetxt(arguments.fout_geneids, transcripts, fmt='%u', delimiter='\t')
np.savetxt(arguments.fout_target_labels, true_labels_target, fmt='%u', delimiter='\t')

print('Saving source data to \'{0}\'.'.format(arguments.fout_source_data))
np.savetxt(arguments.fout_source_data, data_source, fmt='%u', delimiter='\t')
np.savetxt(arguments.fout_source_labels, true_labels_source, fmt='%u', delimiter='\t')

print('Done.')
