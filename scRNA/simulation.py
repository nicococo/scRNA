import numpy as np
import random
import ast
import sys
import pdb
from sklearn.model_selection import train_test_split


def recursive_dirichlet(cluster_spec, num_cells,
                        dirichlet_parameter_cluster_size):

    num_clusters = len(cluster_spec)
        
    cluster_sizes = np.ones(num_clusters)
    
    while min(cluster_sizes) == 1:
        cluster_sizes = np.floor(np.random.dirichlet(np.ones(num_clusters) * dirichlet_parameter_cluster_size, size=None) * num_cells)

    #Because of the floor call we always have a little too few cells
    if np.sum(cluster_sizes) != num_cells:
        cluster_sizes[0] = cluster_sizes[0] - (np.sum(cluster_sizes) - num_cells)
    #if min(cluster_sizes)<=1:
	#		pdb.set_trace()
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


def flatten(xs):
    res = []

    def loop(ys):
        for i in ys:
            if isinstance(i, list):
                loop(i)
            else:
                res.append(i)
    loop(xs)
    if res == []:
        return res
    else:
        return np.hstack(res)


def split_source_target(toy_data, true_toy_labels,
                        target_ncells=1000, source_ncells=1000,
                        mode=2, source_clusters = None,
                        noise_target=False, noise_sd=0.5, common=2, cluster_spec = None):
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
    #		          7 = Define number of overlapping clusters
    # source_clusters = None # Array of cluster ids to use in mode 6
    # noise_target = False # Add some per gene gaussian noise to the target?
    # noise_sd = 0.5 # SD of gaussian noise
    # nscr = 2 # number of source clusters
    # ntrg = 2 # number of target clusters
    # common = 2 # number of shared clusters

    assert (target_ncells + source_ncells <= toy_data.shape[1])

    #First split the 'truth' matrix into a set we will use and a set we wont
    #For mode 6,4,7 we do this differently
    if target_ncells + source_ncells < toy_data.shape[1] and mode != 6 and mode != 4 and mode != 7:

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

        true_toy_labels = np.array(true_toy_labels)
        cluster_names, counts = np.unique(true_toy_labels,return_counts=True)
        # Assign cluster that is not in source
        source_cluster = cluster_names[np.argmax(counts)]
        source_indices = (true_toy_labels != source_cluster)
        toy_data_source_exclusive = toy_data[:, source_indices]
        true_toy_labels_source_exclusive = true_toy_labels[source_indices]

        # Assign cluster that is not in target
        counts[np.argmax(counts)]=0
        target_cluster = cluster_names[np.argmax(counts)]
        target_indices = (true_toy_labels != target_cluster)
        toy_data_target_exclusive = toy_data[:, target_indices]
        true_toy_labels_target_exclusive = true_toy_labels[target_indices]

        _, toy_data_source, _, true_toy_labels_source = train_test_split(np.transpose(toy_data_source_exclusive), true_toy_labels_source_exclusive, test_size = source_ncells, stratify = true_toy_labels_source_exclusive)

        _, toy_data_target, _, true_toy_labels_target = train_test_split(np.transpose(toy_data_target_exclusive), true_toy_labels_target_exclusive, test_size = target_ncells, stratify = true_toy_labels_target_exclusive)

        toy_data_source = np.transpose(toy_data_source)
        toy_data_target = np.transpose(toy_data_target)


    elif mode == 5:
        proportion_source = 1.-proportion_target
        cluster_names = np.unique(true_toy_labels)
        source_clusters = cluster_names[
            0: int(len(cluster_names) * proportion_source)
        ]
        target_clusters = cluster_names[
            int(len(cluster_names) * proportion_source):len(cluster_names)
        ]


        pdb.set_trace()

        source_indices_mat = []
        for i in range(len(source_clusters)):
            source_indices_mat.append(true_toy_labels == source_clusters[i])
        source_indices = np.any(source_indices_mat, 0)
        toy_data_source = toy_data[:, source_indices]

        source_indices_ind = np.argwhere(source_indices == True)[:, 0]
        true_toy_labels_arr = np.array(true_toy_labels, dtype=np.int)

        true_toy_labels_source = true_toy_labels_arr[source_indices_ind]

        target_indices_mat = []
        for i in range(len(target_clusters)):
            target_indices_mat.append(true_toy_labels == target_clusters[i])
        target_indices = np.any(target_indices_mat,0)
        toy_data_target = toy_data[:,target_indices]

        target_indices_ind = np.argwhere(target_indices == True)[:, 0]

        true_toy_labels_target = true_toy_labels_arr[target_indices_ind]

    elif mode == 6:
        assert(source_clusters != None)

        source_cluster_indices = [i for i, x in enumerate(true_toy_labels) if x in source_clusters]

        toy_data_source = toy_data[:, source_cluster_indices]
        true_toy_labels_source = [true_toy_labels[i] for i in source_cluster_indices]

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

    elif mode == 7:
        '''
        In splitting mode 7, the data are splitted  as follows:
            common: number of shared clusters
            nsrc:	number of source clusters
            ntrg:	number of target clusters
            cluster_spec:
                = None: no subcluster structures
                = [1,2,[3,4],[5,6,7],8]: indicate the subcluster structure. The first level cluster structures are taken as one
                cluster - here for instance we would have 5 clusters: [1,2,3,4,5], where cluster 3 automatically involve the subcluster
                3 and 4 from the original cluster_spec
        '''

        if cluster_spec == None:
            nclusters = np.unique(true_toy_labels)
            ntrg = np.int(np.floor((len(nclusters) + common)/2.))
            nsrc = len(nclusters) - ntrg

            assert(nsrc + ntrg - common <= len(nclusters))
            Cidx = np.random.choice(nclusters,common,False)
            Sidx = np.concatenate((np.array(Cidx).copy(),np.random.choice(np.setdiff1d(nclusters,Cidx),nsrc-common)),axis=0)
            Tidx = np.concatenate((np.array(Cidx).copy(),np.random.choice(np.setdiff1d(nclusters,Sidx),ntrg-common)),axis=0)
        else:
            #print cluster_spec
            nclusters = np.arange(len(cluster_spec))  # compute cluster dependence for the first level cluster structure
            #print nclusters
            ntrg = np.int(np.floor((len(nclusters) + common)/2.))   # number of clusters in target
            nsrc = len(nclusters) - ntrg + common   # number of clusters in source
            assert(nsrc + ntrg - common <= len(nclusters))
            Cidx = np.random.choice(nclusters,common,False) # Indices of common clusters, chosen at random
            if not nsrc-common==0:
                Sidx = np.random.choice(np.setdiff1d(nclusters,Cidx),nsrc-common,False) # Indices of exclusive source clusters
                if not ntrg-common==0:
                    Tidx = np.random.choice(np.setdiff1d(nclusters,np.union1d(Sidx,Cidx)),ntrg-common,False) # Indices of exclusive target clusters
                else:
                    Tidx = []
            else:
                Sidx = []
                Tidx = []
            #np.concatenate((np.array(Cidx).copy(),np.random.choice(np.setdiff1d(nclusters,Sidx),ntrg-common)),axis=0)

            Cidx = flatten([cluster_spec[c] for c in  Cidx])

            if ntrg>common:
                Tidx = flatten([cluster_spec[c] for c in  Tidx])
            else:
                Tidx = []
            if nsrc>common:
                Sidx = flatten([cluster_spec[c] for c in  Sidx])
            else:
                Sidx = []

            #excl_t_cells = sum(np.in1d(true_toy_labels, Tidx))
            #excl_s_cells = sum(np.in1d(true_toy_labels,Sidx))
            #excl_c_cells = sum(np.in1d(true_toy_labels, Cidx))

            ## Make sure that the clusters have enough cells to fill up source or target data
            #try:
            #    assert (source_ncells <= toy_data_source.shape[1])
            #except AssertionError:
            #    print("There aren't enough cells in the source clusters. Raise ncells")
            #    sys.exit()
            '''
            get shared cluster split for source and target data
            '''
            shared_idx=np.in1d(true_toy_labels,Cidx)
            # shared_trg_size = target_ncells - sum(np.in1d(true_toy_labels,Tidx))
            shared_trg_size = int(target_ncells * float(len(Cidx))/(len(Tidx)+len(Cidx)))
            # shared_src_size = source_ncells - sum(np.in1d(true_toy_labels,Sidx))   #
            shared_src_size = int(source_ncells * float(len(Cidx))/(len(Sidx)+len(Cidx)))
            if shared_trg_size+shared_src_size >=  sum(shared_idx):
                to_take_away = np.ceil((shared_trg_size+shared_src_size-sum(shared_idx))/2)+1
                shared_trg_size=np.int(shared_trg_size-to_take_away)
                shared_src_size=np.int(shared_src_size-to_take_away)
            if shared_trg_size == 0:
                data_shared_target, data_shared_source, labels_shared_target, labels_shared_source = [],[],[],[]
            else:
                data_shared_target, data_shared_source, labels_shared_target, labels_shared_source = train_test_split(toy_data[:,shared_idx].transpose(),np.array(true_toy_labels)[shared_idx],train_size=shared_trg_size,test_size=shared_src_size)

            '''
            get cluster split for target data
            '''
            if ntrg > common:
                trg_idx = np.in1d(true_toy_labels, Tidx)
                add_trg_size = int(target_ncells - shared_trg_size)
                toy_data_target, _, true_toy_labels_target, _ = train_test_split(toy_data[:, trg_idx].transpose(), np.array(true_toy_labels)[trg_idx],train_size=add_trg_size, test_size=0)
                if shared_trg_size != 0:
                    toy_data_target = np.concatenate((data_shared_target,toy_data_target))
                    true_toy_labels_target = np.concatenate((labels_shared_target,true_toy_labels_target))

            else:
                toy_data_target = data_shared_target
                true_toy_labels_target = labels_shared_target
            '''
            get cluster split for source data
            '''
            if nsrc>common:
                src_idx = np.in1d(true_toy_labels, Sidx)
                add_src_size = int(source_ncells - shared_src_size)
                toy_data_source, _, true_toy_labels_source, _ = train_test_split(toy_data[:, src_idx].transpose(), np.array(true_toy_labels)[src_idx],train_size=add_src_size, test_size=0)
                if shared_src_size != 0:
                    toy_data_source = np.concatenate((data_shared_source,toy_data_source))
                    true_toy_labels_source = np.concatenate((labels_shared_source,true_toy_labels_source))
            else:
                toy_data_source = data_shared_source
                true_toy_labels_source 	= labels_shared_source

            toy_data_source = toy_data_source.transpose()
            toy_data_target = toy_data_target.transpose()

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


def split_source_target_2labels(toy_data, true_toy_labels_1, true_toy_labels_2,
                            target_ncells=1000, source_ncells=1000,
                            mode=2, source_clusters=None,
                            noise_target=False, noise_sd=0.5, common=2, cluster_spec=None):

    # Parameters for splitting data in source and target set:
    # target_ncells = 1000 # How much of the data will be target data?
    # source_ncells = 1000 # How much of the data will be source data?
    # splitting_mode = 2
    # Splitting mode: 1 = split randomly,
    # source_clusters = None # Array of cluster ids to use in mode 6
    # noise_target = False # Add some per gene gaussian noise to the target?
    # noise_sd = 0.5 # SD of gaussian noise
    # nscr = 2 # number of source clusters
    # ntrg = 2 # number of target clusters
    # common = 2 # number of shared clusters

    assert (target_ncells + source_ncells <= toy_data.shape[1])

    # First split the 'truth' matrix into a set we will use and a set we wont
    # For mode 6,4,7 we do this differently
    if target_ncells + source_ncells < toy_data.shape[1] and mode != 6 and mode != 4 and mode != 7:
        toy_data, _, true_toy_labels_1, _, true_toy_labels_2,_ = \
            train_test_split(
                np.transpose(toy_data),
                true_toy_labels_1, true_toy_labels_2,
                test_size=toy_data.shape[1] - (target_ncells + source_ncells),
                stratify=true_toy_labels_1
            )
        toy_data = np.transpose(toy_data)

    if mode == 1:
        toy_data_source, \
        toy_data_target, \
        true_toy_labels_source_1, \
        true_toy_labels_target_1, \
        true_toy_labels_source_2, \
        true_toy_labels_target_2 = \
            train_test_split(
                np.transpose(toy_data),
                true_toy_labels_1,
                true_toy_labels_2,
                test_size=target_ncells
            )
        toy_data_source = np.transpose(toy_data_source)
        toy_data_target = np.transpose(toy_data_target)

    else:
        print "Unknown mode!"
        toy_data_source = []
        toy_data_target = []
        true_toy_labels_source_1 = []
        true_toy_labels_target_1 = []
        true_toy_labels_source_2 = []
        true_toy_labels_target_2 = []

    if noise_target:
        toy_data_target = np.transpose(
            np.transpose(toy_data_target) +
            np.random.normal(size=toy_data.shape[0], scale=noise_sd)
        )

    # Some modes can by chance gives us n+1 column matrices. This just neatens
    # this by throwing away the additional column
    toy_data_source = toy_data_source[:, 0:source_ncells]
    toy_data_target = toy_data_target[:, 0:target_ncells]
    true_toy_labels_source_1 = true_toy_labels_source_1[0:source_ncells]
    true_toy_labels_target_1 = true_toy_labels_target_1[0:target_ncells]
    true_toy_labels_source_2 = true_toy_labels_source_2[0:source_ncells]
    true_toy_labels_target_2 = true_toy_labels_target_2[0:target_ncells]

    return toy_data_source, toy_data_target, \
           true_toy_labels_source_1, true_toy_labels_target_1, \
           true_toy_labels_source_2, true_toy_labels_target_2
