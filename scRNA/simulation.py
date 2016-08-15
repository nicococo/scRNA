import numpy as np
import random
import ast
import sys
import pdb
from sklearn.cross_validation import train_test_split

def recursive_dirichlet(cluster_spec, num_cells,
                        dirichlet_parameter_cluster_size):

    num_clusters = len(cluster_spec)
        
    cluster_sizes = np.ones(num_clusters)
    
    while min(cluster_sizes) == 1:
        cluster_sizes = \
          np.floor(
            np.random.dirichlet(
              np.ones(num_clusters) * 
              dirichlet_parameter_cluster_size, 
              size=None
            ) * num_cells
          )

    #Because of the floor call we always have a little too few cells
    if np.sum(cluster_sizes) != num_cells:
        cluster_sizes[0] = \
          cluster_sizes[0] - (np.sum(cluster_sizes) - num_cells)
          
    assert min(cluster_sizes) > 1
    assert sum(cluster_sizes) == num_cells
    
    cluster_sizes = cluster_sizes.astype(int).tolist()
    for i, spec in enumerate(cluster_spec):
        if type(spec) is list:
             cluster_sizes[i] = recursive_dirichlet(
               spec, 
               cluster_sizes[i],
               dirichlet_parameter_cluster_size
             )
             
    return(cluster_sizes)
            
def generate_de_logfc(ngenes, prop_genes_de, de_logfc):

    nde_genes = int(np.floor(ngenes * prop_genes_de))
    up_down = np.sign(np.random.normal(size = nde_genes))    
    logfc = map((lambda x: x * de_logfc), up_down)
        
    logfc = logfc + [0] * (ngenes - nde_genes)
    random.shuffle(logfc)
    
    return(logfc)

def recursive_generate_counts(cluster_nums, num_genes, true_means,
                              parent_logfc, nb_dispersion,
                              min_prop_genes_de, max_prop_genes_de,
                              mean_de_logfc, sd_de_logfc):

    cluster_counts = [0] * len(cluster_nums)
        
    for i,num_cells in enumerate(cluster_nums):

        #Set DE for this cluster or set of clusters
        prop_genes_de = np.random.uniform(min_prop_genes_de, max_prop_genes_de)
        de_logfc      = np.random.normal(mean_de_logfc, sd_de_logfc)
        logfc = np.add(
          parent_logfc, 
          generate_de_logfc(num_genes, prop_genes_de, de_logfc)
        )

        if type(num_cells) is list:
            cluster_counts[i] = \
              recursive_generate_counts(
                num_cells, num_genes, true_means, logfc, nb_dispersion,
                min_prop_genes_de, max_prop_genes_de,
                mean_de_logfc, sd_de_logfc
              )
        else:
            cluster_counts[i] = \
              generate_counts(
                num_cells, num_genes, true_means, logfc, nb_dispersion
              )
            
    return(np.hstack(cluster_counts))
    
def generate_counts(num_cells, num_genes, true_means, logfc, nb_dispersion):
    
    #Per cell noise
    all_facs = np.power(
      2, 
      np.random.normal(
        loc = 0, scale = 0.5, size = num_cells
      )
    )
    effective_means = np.outer(true_means, all_facs)

    effective_means = np.transpose(
      np.multiply(np.transpose(effective_means), np.power(2, logfc))
    )
    
    # Generate data
    sample = np.random.negative_binomial(
      p = (1 / nb_dispersion) / ((1/nb_dispersion) + effective_means),
      n = 1 / nb_dispersion, size = [num_genes, num_cells]
    )
    
    return(sample)
    
def generate_toy_data(
                      num_genes = 10000, num_cells = 1000, 

                      cluster_spec = None,
                      dirichlet_parameter_cluster_size = 10,

                      gamma_shape = 2, gamma_rate = 2,
                      nb_dispersion = 0.1,
                      min_prop_genes_de = 0.1,
                      max_prop_genes_de = 0.4,
                      mean_de_logfc     = 1,
                      sd_de_logfc       = 0.5,
                     ):
                      
    # Toy experiment parameters
    # Data generation parameters

    # num_genes = 10000  # 10000, number of genes
    # num_cells = 1000  # 1000, number of cells

    # Cluster spec = None # Definition of cluster hierarchy
    # dirichlet_parameter_cluster_size = 10  # 10, Dirichlet parameter for cluster sizes, between 0 and inf, bigger values make cluster sizes more similar

    # Generate Cluster sizes
    cluster_sizes = recursive_dirichlet(
      cluster_spec,
      num_cells,
      dirichlet_parameter_cluster_size
    )
    
    #Define the 'true' population mean expression levels
    true_means = np.random.gamma(
      gamma_shape, scale=1 / float(gamma_rate), size=num_genes
    )

    counts = recursive_generate_counts(
      cluster_sizes,
      num_genes,
      true_means,
      [0] * num_genes,
      nb_dispersion,
      min_prop_genes_de,
      max_prop_genes_de,
      mean_de_logfc,
      sd_de_logfc
    )
    
    def flatten(l): 
        if type(l) is list:
            return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else [])
        else:
            return([l])

    flat_sizes = flatten(cluster_sizes)
    flat_labels = flatten(cluster_spec)

    labels = []

    for x in zip(flat_labels, flat_sizes):
        labels = labels + ([x[0]] * x[1])

    return [counts, labels]

def split_source_target(toy_data, true_toy_labels, 
                        target_ncells=1000, source_ncells=1000,
                        mode=2, source_clusters = None,
                        noise_target=False, noise_sd=0.5):
    # Parameters for splitting data in source and target set:
    # target_ncells = 1000 # How much of the data will be target data?
    # source_ncells = 1000 # How much of the data will be source data?
    # splitting_mode = 2
    # Splitting mode: 1 = split randomly,
    #                 2 = split randomly, but stratified,
    #                 3 = split randomly, but anti-stratified [not implemented]
    #                 4 = Have some overlapping and some exclusive clusters,
    #                 5 = have only non-overlapping clusters
    #                 6 = Define source matrix clusters
    # source_clusters = None # Array of cluster ids to use in mode 6
    # noise_target = False # Add some per gene gaussian noise to the target?
    # noise_sd = 0.5 # SD of gaussian noise
    
    assert (target_ncells + source_ncells <= toy_data.shape[1])

    #First split the 'truth' matrix into a set we will use and a set we wont
    #For mode 6 we do this differently
    if target_ncells + source_ncells < toy_data.shape[1] and mode != 6:
      
        toy_data, _, true_toy_labels, _ = \
            train_test_split(
                np.transpose(toy_data),
                true_toy_labels,
                test_size = toy_data.shape[1] - (target_ncells + source_ncells),
                stratify = true_toy_labels
            )
        toy_data = np.transpose(toy_data)
    
    proportion_target = float(target_ncells) / (source_ncells + target_ncells) 
    
    if mode == 1:
        toy_data_source, \
        toy_data_target, \
        true_toy_labels_source, \
        true_toy_labels_target = \
            train_test_split(
                np.transpose(toy_data),
                true_toy_labels,
                test_size = target_ncells
            )
        toy_data_source = np.transpose(toy_data_source)
        toy_data_target = np.transpose(toy_data_target)

    elif mode == 2:
        toy_data_source, \
        toy_data_target, \
        true_toy_labels_source, \
        true_toy_labels_target = \
            train_test_split(
                np.transpose(toy_data),
                true_toy_labels,
                test_size = target_ncells,
                stratify = true_toy_labels
            )
        toy_data_source = np.transpose(toy_data_source)
        toy_data_target = np.transpose(toy_data_target)

    elif mode == 3:
        print "Mode 3 not implemented!"
        toy_data_source=[]
        toy_data_target=[]
        true_toy_labels_source=[]
        true_toy_labels_target = []

    elif mode == 4:
        cluster_names = np.unique(true_toy_labels)

        # Assign exclusive clusters
        # To source data set
        source_cluster = cluster_names[0]
        source_indices = (true_toy_labels == source_cluster)
        toy_data_source_exclusive = toy_data[:, source_indices]
        true_toy_labels_source_exclusive = true_toy_labels[source_indices]

        # To target data set
        target_cluster = cluster_names[1]
        target_indices = (true_toy_labels == target_cluster)
        toy_data_target_exclusive = toy_data[:, target_indices]
        true_toy_labels_target_exclusive = true_toy_labels[target_indices]

        # Distribute the rest randomly
        random_clusters = cluster_names[2:]
        random_indices_mat = []
        for i in range(len(random_clusters)):
            random_indices_mat.append(true_toy_labels == random_clusters[i])
        random_indices = np.any(random_indices_mat, 0)
        toy_data_random = toy_data[:, random_indices]
        true_toy_labels_random = true_toy_labels[random_indices]

        toy_data_source_random, 
        toy_data_target_random,
        true_toy_labels_source_random,
        true_toy_labels_target_random = \
            train_test_split(
                np.transpose(toy_data_random),
                true_toy_labels_random,
                test_size = target_ncells
            )

        # Combine exclusive and random data
        toy_data_source = np.concatenate(
            (toy_data_source_exclusive,
             np.transpose(toy_data_source_random)
            ), axis = 1)
        toy_data_target = np.concatenate(
            (toy_data_target_exclusive,
             np.transpose(toy_data_target_random)
            ), axis = 1)
        true_toy_labels_source = np.concatenate(
            (true_toy_labels_source_exclusive, 
             np.transpose(true_toy_labels_source_random)
            ))
        true_toy_labels_target = np.concatenate(
            (true_toy_labels_target_exclusive, 
             np.transpose(true_toy_labels_target_random)
            ))

    elif mode == 5:
        proportion_source = 1-proportion_target
        cluster_names = np.unique(true_toy_labels)
        source_clusters = cluster_names[
            0: int(len(cluster_names) * proportion_source)
        ]
        target_clusters = cluster_names[
            int(len(cluster_names) * proportion_source):len(cluster_names)
        ]

        source_indices_mat = []
        for i in range(len(source_clusters)):
            source_indices_mat.append(true_toy_labels == source_clusters[i])
        source_indices = np.any(source_indices_mat, 0)
        toy_data_source = toy_data[:, source_indices]
        true_toy_labels_source = true_toy_labels[source_indices]

        target_indices_mat = []
        for i in range(len(target_clusters)):
            target_indices_mat.append(true_toy_labels == target_clusters[i])
        target_indices = np.any(target_indices_mat,0)
        toy_data_target = toy_data[:,target_indices]
        true_toy_labels_target = true_toy_labels[target_indices]

    elif mode == 6:

        assert(source_clusters != None)
        
        source_cluster_indices = \
            [i for i, x in enumerate(true_toy_labels) if x in source_clusters]
                
        toy_data_source = toy_data[:, source_cluster_indices]
        true_toy_labels_source = \
            [true_toy_labels[i] for i in source_cluster_indices]
                
        try:
            assert(source_ncells <= toy_data_source.shape[1])
        except AssertionError:
            print("There aren't enough cells in the source clusters. Raise ncells")
            sys.exit()
        
        #Now take the source from these clusters only in a stratified way
        toy_data_source, _, true_toy_labels_source, _ = \
            train_test_split(
                np.transpose(toy_data_source),
                true_toy_labels_source,
                test_size = toy_data_source.shape[1] - source_ncells,
                stratify = true_toy_labels_source
            )
        toy_data_source = np.transpose(toy_data_source)
                
        #Now take the target from the whole dataset again stratified
        toy_data_target, _, true_toy_labels_target, _ = \
            train_test_split(
                np.transpose(toy_data),
                true_toy_labels,
                test_size = toy_data.shape[1] - target_ncells,
                stratify = true_toy_labels
            )
        toy_data_target = np.transpose(toy_data_target)


    else:
        print "Unknown mode!"
        toy_data_source=[]
        toy_data_target=[]
        true_toy_labels_source=[]
        true_toy_labels_target = []

    if noise_target:
        toy_data_target = np.transpose(
            np.transpose(toy_data_target) + 
            np.random.normal(size = toy_data.shape[0], scale = noise_sd)
        )

    #Some modes can by chance gives us n+1 column matrices. This just neatens
    #this by throwing away the additional column
    toy_data_source = toy_data_source[:,0:source_ncells]
    toy_data_target = toy_data_target[:,0:target_ncells]
    true_toy_labels_source = true_toy_labels_source[0:source_ncells]
    true_toy_labels_target = true_toy_labels_target[0:target_ncells]

    return toy_data_source, toy_data_target, \
           true_toy_labels_source, true_toy_labels_target
