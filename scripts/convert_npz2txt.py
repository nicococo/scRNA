import numpy as np
import pdb
import pandas as pd

def intersect(a, b):
    return list(set(a) & set(b))

# Load Pfizer data
data_pfizer = np.load('C:\Users\Bettina\ml\scRNAseq\Data\Pfizer data\pfizer_data.npz')
cell_names_pfizer = data_pfizer['filtered_inds']
gene_IDs_pfizer = tuple(x[0] for x in data_pfizer['transcripts'])
data_array_pfizer_raw = data_pfizer['rpm_data']
pdb.set_trace()
# Load Usoskin data
data_uso = np.load('C:\Users\Bettina\ml\scRNAseq\data\Usoskin.npz')
cell_names_uso = data_uso['cells']
transcript_names_uso = data_uso['transcripts']
data_array_uso = data_uso['data']

# Convert gene IDs to gene names
raw_gene_names = pd.read_table('C:\Users\Bettina\ml\scRNAseq\Data\Pfizer data\gene_names.txt')
gene_names = raw_gene_names.as_matrix()
gene_IDs_ensembl_order = tuple(x[1] for x in gene_names)

# Select the right genes in Pfizer data
transcript_names_pfizer_temp=[]
data_array_pfizer_temp=[]
for x in gene_IDs_pfizer:
    if x in gene_IDs_ensembl_order:
        transcript_names_pfizer_temp.append(gene_names[gene_IDs_ensembl_order.index(x),0])
        data_array_pfizer_temp.append(data_array_pfizer_raw[gene_IDs_ensembl_order.index(x)])

data_array_pfizer = np.asarray(data_array_pfizer_temp)
transcript_names_pfizer = np.asarray(transcript_names_pfizer_temp)

transcript_names_uso_now = transcript_names_uso.tolist()
common_genes = intersect(transcript_names_pfizer_temp, transcript_names_uso)

# Combine the two datasets
data_pfizer_uso=[]
for x in common_genes:
    data_pfizer_uso_temp = data_array_pfizer[transcript_names_pfizer_temp.index(x)].tolist()
    data_pfizer_uso_temp_2 = data_array_uso[transcript_names_uso_now.index(x)].tolist()
    data_pfizer_uso_temp.extend(data_pfizer_uso_temp_2)
    data_pfizer_uso.append(data_pfizer_uso_temp)

data_array_pfizer_uso = np.asarray(data_pfizer_uso)
cell_names_pfizer_uso = np.asarray(list(cell_names_pfizer) + list(cell_names_uso))
transcript_names_pfizer_uso = np.asarray(common_genes)

# Save data
# Pfizer
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\Pfizer data\data_pfizer.txt", data_array_pfizer, delimiter=" ", fmt="%10.5f")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\Pfizer data\cell_names_pfizer.txt", cell_names_pfizer, delimiter=" ", fmt="%s")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\Pfizer data\\transcript_names_pfizer.txt", transcript_names_pfizer, delimiter=" ", fmt="%s")

# Pfizer + Usoskin
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\Pfizer data\data_pfizer_uso.txt", data_array_pfizer_uso, delimiter=" ", fmt="%10.5f")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\Pfizer data\cell_names_pfizer_uso.txt", cell_names_pfizer_uso, delimiter=" ", fmt="%s")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\Pfizer data\\transcript_names_pfizer_uso.txt", transcript_names_pfizer_uso, delimiter=" ", fmt="%s")

# Usoskin
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\data_uso.txt", data_array_uso, delimiter=" ", fmt="%10.5f")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\cell_names_uso.txt", cell_names_uso, delimiter=" ", fmt="%s")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\\transcript_names_uso.txt", transcript_names_uso, delimiter=" ", fmt="%s")


# Ting
#data_ting = np.load('C:\Users\Bettina\ml\scRNAseq\data\Ting.npz')
#cell_names_ting = data_ting['cells']
#transcript_names_ting = data_ting['transcripts']
#data_array_ting = data_ting['data']

#np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\data_ting.txt", data_array_ting, delimiter=" ", fmt="%i")
#np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\cell_names_ting.txt", cell_names_ting, delimiter=" ", fmt="%s")
#np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\_transcript_names_ting.txt", transcript_names_ting, delimiter=" ", fmt="%s")


# Zeisel
#data_zeisel = np.load('C:\Users\Bettina\ml\scRNAseq\data\Zeisel.npz')
#cell_names_zeisel = data_zeisel['cells']
#transcript_names_zeisel = data_zeisel['transcripts']
#data_array_zeisel = data_zeisel['data']

#np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\data_zeisel.txt", data_array_zeisel, delimiter=" ", fmt="%i")
#np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\cell_names_zeisel.txt", cell_names_zeisel, delimiter=" ", fmt="%s")
#np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\_transcript_names_zeisel.txt", transcript_names_zeisel, delimiter=" ", fmt="%s")


# Are there any duplicates?
#set([x for x in transcript_names_pfizer if transcript_names_pfizer.count(x) > 1])
