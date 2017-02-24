scRNA
=====
***************

Python framework for single-cell RNA-seq clustering with special 
focus on transfer learning (multitask/domain adaptation). 
This package contains methods for generating artifical data,
clustering, and blending datasets (domain adaptation).

This software was written by Nico Goernitz, Bettina Mieth, Marina Vidovic, Alex Gutteridge. 

### News
- First version that can be conveniently installed using the _pip install git+https://github.com/nicococo/scRNA.git_ 
command. Enjoy :)
- Command line script available


Getting started
===============

Installation
------------
After installing the software package using the _pip install git+https://github.com/nicococo/scRNA.git_
command, three command line arguments will be available for **MacOS and Linux only**: 


Simulating scRNA-seq Data
-------------------------

### Generating Artificial Data 
_scRNA-generate-data.sh_


Domain Adaption for scRNA-seq Data
----------------------------------

### 2. Setup the Source Dataset
_scRNA-source.sh_ 

Input and output files: 
|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--src-fname  | Source *.npz result filename from Step 2       | 
|--fname      | Target data (TSV file)                         |
|--fgene-ids  | Target gene ids (TSV file)                     |
|--fout       | Result files will use this prefix              |
|--flabels    | (optional) Target cluster labels (TSV file)    |


Data pre-processing Gene/cell filtering arguments (SC3 inspired):
|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--min_expr_genes      | (Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type=int) |
|--non_zero_threshold  | (Cell/gene filter) Threshold for zero expression per gene (default 1.0)|
|--perc_consensus_genes| (Gene filter) Filter genes that coincide across a percentage of cells (default 0.98) |



### 3. Target Dataset Clustering 
_scRNA-source.sh_ 

Input and output files: 
|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--src-fname  | Source *.npz result filename from Step 2       | 
|--fname      | Target data (TSV file)                         |
|--fgene-ids  | Target gene ids (TSV file)                     |
|--fout       | Result files will use this prefix              |
|--flabels    | (optional) Target cluster labels (TSV file)    |


Data pre-processing Gene/cell filtering arguments (SC3 inspired):
|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--min_expr_genes      | (Cell filter) Minimum number of expressed genes (default 2000)", default=2000, type=int) |
|--non_zero_threshold  | (Cell/gene filter) Threshold for zero expression per gene (default 1.0)|
|--perc_consensus_genes| (Gene filter) Filter genes that coincide across a percentage of cells (default 0.98) |


SC3-specific distances and transformations:
|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--sc3-dists  |(SC3) Comma-separated MTL distances (default euclidean) |
|--sc3-transf |(SC3) Comma-separated transformations (default pca) |


Test settings: The software will cluster any combination of the cluster-range
and mixtures and store results separately.

|Command line arguments|Description                            |
|----------------------|:--------------------------------------|
|--cluster-range | Comma separated list of clusters (default 6,7,8) |
|--mixtures | Comma separated list of convex combination src-trg mixture coefficient (0.=no transfer, default 0.0,0.1,0.2)| 



