# About
Python framework for single-cell RNA-seq clustering with special 
focus on transfer learning (multitask/domain adaptation). This package contains methods
for generating and clustering (single task and multitask).

This software was written by Nico Goernitz, Bettina Mieth, Alex Gutteridge. TU Berlin, 2016.

# News
- First version that can be conveniently installed using the _pip install git+https://github.com/nicococo/scRNA.git_ 
command. Enjoy :)
- Command line script are available

# How to use
Basically, the software breaks down into 2 distinct parts: data generation and clustering.
After the installation using _pip_ is complete, there are
a couple of (bash) command line scripts that should be accessible from anywhere. Below are some examples and details 
for each of the parts. 

Calling those scripts without any parameter will give a short list of parameters and corresponding descriptions.

## 1. Generating Artificial Data 
scRNA-generate-data.sh

## 2.a. Single Task Clustering
(a) scRNA-sc3.sh \\
(b) scRNA-nmf.sh

Here, at least a input expression matrix is needed as well as the number of clusters.
There are some more model parameters that can be chosen, e.g. the list of transformations ("pca", "spectral")
as well as the list of distances ("Pearson", "Euclidean", etc pp) for the SC3 variant. For the 
NMF variant, the overall regularization strength (alpha) and the l1-influence can be defined.
Both methods use by default cell- and gene-filter as well as a log2-transformation of the data.

If cluster labels are available then these can be passed to the programm as well. If present, the scripts
will print the adjusted Rand index (ARI), comparing these labels with the inferred ones, at the end.


## 2.b. Multitask Clustering
(a) scRNA-mtl-nmf.sh \\
(b) scRNA-mtl-sc3.sh 

In addition to the parameters discussed in 2.b., a source dataset needs to be defined as well as 
gene id description files for the source and the target dataset. Only matching gene ids will be considered
in multitask learning. Hence, it is the users duty to ensure a high as possible percentage of gene ids in
both dataset overlap.

For the SC3 variant, a mixture parameter can be passed which controls the influence of the multitask
learning (0.0=no influence, 1.0=full influence).
