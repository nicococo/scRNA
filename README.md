scRNA
=====
Python framework for single-cell RNA-seq clustering with special 
focus on transfer learning. This package contains methods for 
generating artifical data, clustering, and blending datasets (domain adaptation).

This software was written by Nico Goernitz, Bettina Mieth, Marina Vidovic, Alex Gutteridge. 


#### News
* [Website](http://nicococo.github.io/scRNA/) is up and running
* [Wiki](https://github.com/nicococo/scRNA/wiki) with detailed information (e.g. command line arguments)
* Please report [Bugs](https://github.com/nicococo/scRNA/issues) 
* First version that can be conveniently installed using the _pip install git+https://github.com/nicococo/scRNA.git_ 
command. Enjoy :)
* Command line script available



Getting started
---------------

### Installation
We assume that Python >2.7 is installed and the _pip_ command is
callable from the command line. If starting from scratch, we recommend installing 
the [Anaconda](https://www.continuum.io/downloads) open data science 
platform (w/ Python 2.7) which comes with a bunch of most useful packages
for scientific computing.

The *scRNA* software package can be installed using the _pip install git+https://github.com/nicococo/scRNA.git_
command. After successful completion, three command line arguments will be 
available for **MacOS and Linux only**: 

* _scRNA-generate-data.sh_
* _scRNA-source.sh_ 
* _scRNA-target.sh_ 


### Example 
Step 1: Installation with _pip install git+https://github.com/nicococo/scRNA.git_
![Installation with _pip install git+https://github.com/nicococo/scRNA.git_](doc/screen_install_pip.png)



Step 2: Check the scripts
![Check for the scripts](doc/screen_install_scripts.png)



Step 3: Create directory /foo. Go to directory /foo. Generate some artifical data
by simply calling the _scRNA-generate-data.sh_ (using only default parameters).


![Generate artifical data](doc/screen_install_generate.png)


This will result in a number of files:
* Gene ids
* Source- and target data
* Source- and target ground truth labels



Step 4: Cluster the source data using the provided gene ids and source data. Ie. we want
 to turn off the cell- and gene-filter as well as the log transformation.
Potential problems:
* If a ''Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.''
occurs and Anaconda open data science platform is used, then use _conda install mkl_ first.


![Cluster the source data](doc/screen_install_source.png)


This will result in a number of files:
* t-SNE plots (.png) for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* Output cluster labels (and corresponding cell id) in .tsv format for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* Output source model in .npz format for every number of cluster as specified in the --cluster-range argument (default 6,7,8)
* A summarizing .png figure


