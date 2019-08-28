scRNA
=====
Python framework for single-cell RNA-seq clustering with special 
focus on transfer learning. This package contains methods for 
generating artificial data, clustering, and transfering knowledge
from a source to a target datasets.

This software package is developed by Nico Goernitz, Bettina Mieth, Marina Vidovic, and Alex Gutteridge. 

![Travis-CI](https://travis-ci.org/nicococo/scRNA.svg?branch=master)

#### Publication
This framework and website are part of a publication currently under peer-review at Nature Scientific Reports. Links to the published paper and online supplementary material will be included here once available.

Abstract: In many research areas scientists are interested in clustering objects within small datasets while making use of prior knowledge from large reference datasets. We propose a method to apply the machine learning concept of transfer learning to unsupervised clustering problems and show its effectiveness in the field of single-cell RNA sequencing (scRNA-Seq). The goal of scRNA-Seq experiments is often the definition and cataloguing of cell types from the transcriptional output of individual cells. To improve the clustering of small disease- or tissue-specific datasets, for which the identification of rare cell types is often problematic, we propose a transfer learning method to utilize large and well-annotated reference datasets, such as those produced by the Human Cell Atlas. Our approach modifies the dataset of interest while incorporating key information from the larger reference dataset via Non-negative Matrix Factorization (NMF). The modified dataset is subsequently provided to a clustering algorithm. We empirically evaluate the benefits of our approach on simulated scRNA-Seq data as well as on publicly available datasets. Finally, we present results for the analysis of a recently published small dataset and find improved clustering when transferring knowledge from a large reference dataset. 

#### News
* (2019.08) Information on the experimental results presented in our paper (_under review_) can be accessed in the Section "Replicating experiments"  
* (2019.08) We added example application using Jupyter notebooks (cf. Section "Example application")
* (2019.08) Added Python 3 support (scRNA no longer supports Python 2)
* (2019.08) Finalized version
* (2017.02) Added Travis-CI
* (2017.02) Added string label support
* (2017.02) Simple example [available](#example)
* (2017.02) [Website](http://nicococo.github.io/scRNA/) is up and running
* (2017.02) [Wiki](https://github.com/nicococo/scRNA/wiki) with detailed information (e.g. command line arguments)
* (2017.01) Please report [Bugs](https://github.com/nicococo/scRNA/issues) or other inconveniences
* (2017.01) scRNA can now be conveniently installed using the _pip install git+https://github.com/nicococo/scRNA.git_ 
command (see [Installation](#installation) for further information)
* (2017.01) Command line script available



Getting started
---------------

### Installation
We assume that Python is installed and the _pip_ command is
callable from the command line. If starting from scratch, we recommend installing 
the [Anaconda](https://www.continuum.io/downloads) open data science 
platform (w/ Python 3) which comes with a bunch of most useful packages
for scientific computing.

The *scRNA* software package can be installed using the _pip install git+https://github.com/nicococo/scRNA.git_
command. After successful completion, three command line arguments will be 
available for **MacOS and Linux only**: 

* _scRNA-generate-data.sh_
* _scRNA-source.sh_ 
* _scRNA-target.sh_ 


### Example 
**Step 1**: Installation with _pip install git+https://github.com/nicococo/scRNA.git_
![Installation with _pip install git+https://github.com/nicococo/scRNA.git_](doc/screen_install_pip.png)



**Step 2**: Check the scripts
![Check for the scripts](doc/screen_install_scripts.png)



**Step 3**: Create directory /foo. Go to directory /foo. Generate some artificial data
by simply calling the _scRNA-generate-data.sh_ (using only default parameters).


![Generate artificial data](doc/screen_install_generate.png)


This will result in a number of files:
* Gene ids
* Source- and target data
* Source- and target ground truth labels



**Step 4**: NMF of source data using the provided gene ids and source data. Ie. we want
 to turn off the cell- and gene-filter as well as the log transformation.
 You can provide source labels to be used as a starting point for NMF. If not those labels
 will be generated via NMF Clustering.
Potential problems:
* If a ''Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.''
occurs and Anaconda open data science platform is used, then use _conda install mkl_ first.
* Depending on the data and cluster range, this step can take time. However, you can
 speed up the process by tuning off the t-SNE plots using the _--no-tsne_ command 
 (see [Wiki](https://github.com/nicococo/scRNA/wiki) for further information)

![Cluster the source data](doc/screen_install_source.png)


This will result in a number of files:
* t-SNE plots (.png) for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* Output source model in .npz format for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* A summarizing .png figure
* True cluster labels - either as provided from user or as generated via NMF Clustering - (and corresponding cell id) in .tsv format for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* Model cluster labels after NMF (and corresponding cell id) in .tsv format for every number of cluster as specified in the --cluster-range argument (default 6,7,8)




**Step 5**: Now, it is time to cluster the target data and transfer knowledge from the source model to our target data. Therefore, we need to
choose a source data model which was generated in **Step 4**. In this example, we will 
pick the model with 8 cluster (*src_c8.npz*).

* Depending on the data, the cluster range and the mixture range, this step can take a long
time. However, you can  speed up the process by tuning off the t-SNE plots using the _--no-tsne_ command 
(see [Wiki](https://github.com/nicococo/scRNA/wiki) for further information)

![Cluster the target data](doc/screen_install_target.png)

Which results in a number of files (for each value in the cluster range).
* Predicted cluster labels after transfer learning (and corresponding cell id) in .tsv format for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* t-SNE plots with predicted labels (.png)
* Data and gene ids in .tsv files

In addition there is a summarizing .png figure of all accs and a t-SNE plot with the real target labels, if they were provided.

![Cluster the target data](doc/screen_install_result.png)

Command line output shows a number of results: unsupervised and supervised (if no ground truth labels 
are given this will remain 0.) accuracy measures.

Example application
---------------

Using Jupyter notebooks, we showcase the main workflow as well as the abilities of the application.
The main features are 
* generating read-count data 
* data splits using various scenarios
* source data clustering with and without accompanying labels
* augmented clustering of the target data with user defined mix-in of the source data influence.

A minimal working example can be accessed under [https://github.com/nicococo/scRNA/blob/master/notebooks/example.ipynb][example_notebook]


[example_notebook]: https://github.com/nicococo/scRNA/blob/master/notebooks/example.ipynb


Replicating experiments 
---------------
Here, we present the information that is essential to fully reproduce the experiments of our study published at 'link to paper'.

Links to scripts reproducing all experiments of the paper: 
* Script for reproducing [experiments on generated datasets](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_generated_data.py)
* Script for reproducing [experiments on subsampled Tasic data with labels from the original publication](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_tasic.py)
* Script for reproducing [experiments on subsampled Tasic data with NMF labels](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_tasic_NMF_labels.py)
* Script for reproducing [experiments on Hockley data with Usoskin as source data](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_hockley.py)
* Script for reproducing [robustness experiments on Hockley data with Usoskin as source data](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_hockley_robustness.py)
* Script for reproducing [robustness experiments on Hockley data with usoskin as source data and pre-processing through Seurats batch effect removal method](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_hockley_robustness_seurat.py)
 - include link to the corresponding R Script!
* Script for reproducing [robustness experiments on Hockley data with usoskin as source data and pre-processing through MAGIC imputation](https://github.com/nicococo/scRNA/blob/master/scripts/experiments/main_wrapper_hockley_robustness_magic.py)
- include link to the corresponding Matlab Script!

Links to scripts producing figures of the paper. 
* Plot scripts of [Figure 2](https://github.com/nicococo/scRNA/blob/master/scripts/plots/main_plots_generated_data.py)
* Plot scripts of [Figure 3](https://github.com/nicococo/scRNA/blob/master/scripts/plots/main_plots_tasic.py)
* Plot scripts of [Figure 4](https://github.com/nicococo/scRNA/blob/master/scripts/plots/main_plots_hockley.py)
 
Links to scripts producing table 1 of the paper.
* [Without preprocessing](https://github.com/nicococo/scRNA/blob/master/scripts/plots/evaluate_hockley_robustness.py)
* [With pre-processing](https://github.com/nicococo/scRNA/blob/master/scripts/plots/evaluate_hockley_robustness_magic_seurat.py)

Include a section designated to parameter selection where we also reference where to find the pre-processing parameter settings of the generated datasets and the Tasic, Hockley and Usoskin datasets (Supplementary Sections 2.1, 3.1 and 4.1). Details on all other parameters of the respective datasets are also described and linked to the corresponding sections of the supplementary online material (Supplementary Sections 2.2, 3.2 and 4.2).
