import numpy as np
import sys
import ast

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
    
    cluster_sizes = cluster_sizes.tolist()
    for i, spec in enumerate(cluster_spec):
        if type(spec) is list:
             cluster_sizes[i] = recursive_dirichlet(
               spec, 
               cluster_sizes[i],
               dirichlet_parameter_cluster_size
             )
             
    return(cluster_sizes)
    
cluster_sizes = recursive_dirichlet(
  ast.literal_eval(sys.argv[1]),
  int(sys.argv[2]),
  float(sys.argv[3])
)

print cluster_sizes