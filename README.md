**scRNA**
***************

Python framework for single-cell RNA-seq clustering with special 
focus on transfer learning (multitask/domain adaptation). 
This package contains methods for generating artifical data,
clustering, and blending datasets (domain adaptation).

This software was written by Nico Goernitz, Bettina Mieth, Marina Vidovic, Alex Gutteridge. 

### News
- The software outputs
- First version that can be conveniently installed using the _pip install git+https://github.com/nicococo/scRNA.git_ 
command. Enjoy :)
- Command line script available


Getting started
===============

Installation
------------
After installing the software package using the _pip install git+https://github.com/nicococo/scRNA.git_
command, three command line arguments will be available for **MacOS and Linux only**: 

### 1. Generating Artificial Data 
_scRNA-generate-data.sh_

### 2. Setup the Source Dataset
_scRNA-source.sh_ 

### 3. Target Dataset Clustering 
_scRNA-source.sh_ 


|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--src-fname  | Source *.npz result filename from Step 2       | 
|--fname      | Target data (TSV file)                         |
|--fgene-ids  | Target gene ids (TSV file)                     |
|--fout       | Result files will use this prefix              |
|--flabels    | (optional) Target cluster labels (TSV file)    |

Data pre-processing Gene/cell filtering 
--min_expr_genes", help="(Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type=int)
--non_zero_threshold", help="(Cell/gene filter) Threshold for zero expression per gene (default 1.0)", default=1.0, type=float)
--perc_consensus_genes", help="(Gene filter) Filter genes that coincide across a percentage of cells (default 0.98)", default=0.98, type=float)

--cluster-range", help="Comma separated list of clusters (default 6,7,8)", dest='cluster_range', default='5,6,7,8,9', type=str)
--mixtures", help="Comma separated list of convex combination src-trg mixture coefficient (0.=no transfer, default 0.1)", default="0.0,0.1,0.4,0.8", type = str)

--sc3-dists", dest='sc3_dists', help="(SC3) Comma-separated MTL distances (default euclidean)", default='euclidean', type = str)
--sc3-transf", dest='sc3_transf', help="(SC3) Comma-separated transformations (default pca)", default='pca', type = str)


