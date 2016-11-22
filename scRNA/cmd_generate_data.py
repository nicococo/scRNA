import argparse
import sys
import ast
import os

from simulation import generate_toy_data, split_source_target
from utils import *

# 0. PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--fout_target_data",
    help = "Output filename target data",
    default = 'fout_target_data.tsv',
    type = str
)
parser.add_argument(
    "--fout_source_data",
    help = "Output filename source data",
    default = 'fout_source_data.tsv',
    type = str
)
parser.add_argument(
    "--fout_geneids",
    help = "Output filename geneids",
    default = 'fout_geneids.tsv',
    type = str
)
parser.add_argument(
    "--fout_target_labels",
    help = "Output filename target labels",
    default = 'fout_target_labels.tsv',
    type = str
)
parser.add_argument(
    "--fout_source_labels",
    help = "Output filename source labels",
    default = 'fout_source_labels.tsv',
    type = str
)

parser.add_argument(
    "--num_genes",
    help = "Number of genes/transcripts per cell (default 10000)",
    default = 10000,
    type = int
)
parser.add_argument(
    "--num_cells",
    help = "Number of cells (default 1000)",
    default = 2000,
    type = int
)

parser.add_argument(
    "--cluster_spec",
    help = "Cluster specification as Python list",
    default = "[1, 2, 3, [4, 5], [6, [7, 8]]]",
    type = str
)
parser.add_argument(
    "--dir_cluster_size",
    help = "Dirichlet parameter cluster size (default 10)",
    default = 10,
    type = float
)

parser.add_argument(
    "--gamma_shape",
    help = "Gamma distribution shape parameter (default 2)",
    default = 2,
    type = float
)
parser.add_argument(
    "--gamma_rate",
    help = "Gamma distribution rate parameter (default 2)",
    default = 2,
    type = float
)
parser.add_argument(
    "--nb_dispersion",
    help = "Negative binomial distribution dispersion parameter (default 0.1)",
    default = 0.1,
    type = float
)
parser.add_argument(
    "--min_prop_genes_de",
    help = "Minimum proportion of genes DE in each cluster (default 0.1)",
    default = 0.1,
    type = float
)
parser.add_argument(
    "--max_prop_genes_de",
    help = "Maximum proportion of genes DE in each cluster (default 0.4)",
    default = 0.4,
    type = float
)
parser.add_argument(
    "--mean_de_logfc",
    help = "Mean log2 fold change of DE genes (default 1)",
    default = 1,
    type = float
)
parser.add_argument(
    "--sd_de_logfc",
    help = "Standard deviation of log2 fold change of DE genes (default 0.5)",
    default = 0.5,
    type = float
)

parser.add_argument(
    "--target_ncells",
    help = "How much of data will be target data (default 200)",
    default = 100,
    type = int
)
parser.add_argument(
    "--source_ncells",
    help = "How much of data will be source data (default 800)",
    default = 800,
    type = int
)
parser.add_argument(
  "--source_ncells_range",
      help = "How much of data will be source data (range)",
      default = "[]",
      type = str
)
parser.add_argument(
  "--target_ncells_range",
      help = "How much of data will be target data (range)",
      default = "[]",
      type = str
)
parser.add_argument(
    "--noise_target",
    help = "Add noise to target data",
    dest = "noise_target",
    action = 'store_true'
)
parser.add_argument(
    "--no-noise_target",
    help = "Do not add noise to target",
    dest = "noise_target",
    action = 'store_false'
)
parser.set_defaults(noise_target = False)
parser.add_argument(
    "--noise_sd",
    help = "Standard deviation of target noise",
    default = 0.5,
    type = float
)

parser.add_argument(
    "--splitting_mode",
    help = "Splitting mode:\n\t- 1 = split randomly\n\t- 2 = split randomly, but stratified\n\t- 3 = Split randomly but antistratified\n\t- 4 = Have some overlapping and some exclusive clusters\n\t- 5 = Have only exclusive clusters\n\t- 6 = Have some defined clusters as the source\n\t(default 2)",
    default = 4,
    type = int
)
parser.add_argument(
    "--source_clusters",
    help = "Clusters to use as source when splitting by mode 6. Define as Python list",
    default = "[",
    type = str
)

parser.add_argument(
    "--normalise",
    help = "Normalise data to log2(fpkm+1)",
    dest = "normalise",
    action = 'store_true'
)
parser.add_argument(
    "--no-normalise",
    help = "Disable normalise data to log2(fpkm+1)",
    dest = "normalise",
    action = 'store_false'
)
parser.set_defaults(normalise = False)

args = parser.parse_args(sys.argv[1:])
print('Command line argumentss:')
print args

try:
    cluster_spec = ast.literal_eval(args.cluster_spec)
except SyntaxError:
    sys.stderr.write("Error: Invalid cluster specification.")
    sys.exit()


try:
    source_clusters = None
    if args.splitting_mode == 6:
        source_clusters = ast.literal_eval(args.source_clusters)
except SyntaxError:
    sys.stderr.write("Error: Invalid source cluster specification.")
    sys.exit()


# 1. GENERATE TOY DATA
print('\nGenerate artificial single-cell RNA-seq data.')
data, labels = generate_toy_data(
    num_genes                        = args.num_genes,
    num_cells                        = args.num_cells,

    cluster_spec                     = cluster_spec,
    dirichlet_parameter_cluster_size = args.dir_cluster_size,

    gamma_shape                      = args.gamma_shape,
    gamma_rate                       = args.gamma_rate,
    nb_dispersion                    = args.nb_dispersion,
    min_prop_genes_de                = args.min_prop_genes_de,
    max_prop_genes_de                = args.max_prop_genes_de,
    mean_de_logfc                    = args.mean_de_logfc,
    sd_de_logfc                      = args.sd_de_logfc
)
print 'Data dimension: ', data.shape

output_fmt = "%u"

#Perform FPKM and log2 normalisation if required
if args.normalise:
    data = np.log2(data.astype(float) / (np.sum(data, 0) / 1e6) + 1)
    output_fmt = "%f"

# 2. SPLIT TOY DATA IN TARGET AND SOURCE DATA
try:
    source_ncells_range = ast.literal_eval(args.source_ncells_range)
    target_ncells_range = ast.literal_eval(args.target_ncells_range)
except SyntaxError:
    sys.stderr.write("Error: Invalid source/target size specification.")
    sys.exit()

if len(source_ncells_range) == 0:
    source_ncells_range = [args.source_ncells]

if len(target_ncells_range) == 0:
    target_ncells_range = [args.target_ncells]

for sidx, source_ncells in enumerate(source_ncells_range):
    for tidx, target_ncells in enumerate(target_ncells_range):

        print('\nSplit artificial single-cell RNA-seq data in target and source data.')
        data_source, data_target, true_labels_source, true_labels_target = \
            split_source_target(
                data,
                labels,
                target_ncells = target_ncells,
                source_ncells = source_ncells,
                source_clusters = source_clusters,
                noise_target = args.noise_target,
                noise_sd = args.noise_sd,
                mode = args.splitting_mode
            )
        print 'Target data dimension: ', data_target.shape
        print 'Source data dimension: ', data_source.shape

        # 3. GENERATE GENE AND CELL NAMES
        gene_ids = np.arange(args.num_genes)

        # 4. SAVE RESULTS
        print('Saving target data to \'{0}\'.'.format(args.fout_target_data))
        np.savetxt(
            os.path.splitext(args.fout_target_data)[0] + 
            "_T" + str(tidx+1) + "_" + str(target_ncells) + 
            "_S" + str(sidx+1) + "_" + str(source_ncells) +
             os.path.splitext(args.fout_target_data)[1],
            data_target,
            fmt = output_fmt,
            delimiter = '\t'
        )
        np.savetxt(
            os.path.splitext(args.fout_target_labels)[0] + 
            "_T" + str(tidx+1) + "_" + str(target_ncells) + 
            "_S" + str(sidx+1) + "_" + str(source_ncells) +
             os.path.splitext(args.fout_target_labels)[1],
            true_labels_target,
            fmt = '%u',
            delimiter = '\t'
        )
        np.savetxt(
            args.fout_geneids,
            gene_ids,
            fmt = '%u',
            delimiter = '\t'
        )

        print('Saving source data to \'{0}\'.'.format(args.fout_source_data))
        np.savetxt(
            os.path.splitext(args.fout_source_data)[0] + 
            "_T" + str(tidx+1) + "_" + str(target_ncells) + 
            "_S" + str(sidx+1) + "_" + str(source_ncells) +
             os.path.splitext(args.fout_source_data)[1],
            data_source, 
            fmt = output_fmt, 
            delimiter = '\t'
        )
        np.savetxt(
            os.path.splitext(args.fout_source_labels)[0] + 
            "_T" + str(tidx+1) + "_" + str(target_ncells) + 
            "_S" + str(sidx+1) + "_" + str(source_ncells) +
             os.path.splitext(args.fout_source_labels)[1],
            true_labels_source,
            fmt = '%u',
            delimiter = '\t'
        )

print('Done.')
