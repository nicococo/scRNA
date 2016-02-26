import numpy as np

# Ting
data_ting = np.load('C:\Users\Bettina\ml\scRNAseq\data\Ting.npz')
cell_names_ting = data_ting['cells']
transcript_names_ting = data_ting['transcripts']
data_array_ting = data_ting['data']

np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\data_ting.txt", data_array_ting, delimiter=" ", fmt="%i")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\cell_names_ting.txt", cell_names_ting, delimiter=" ", fmt="%s")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\_transcript_names_ting.txt", transcript_names_ting, delimiter=" ", fmt="%s")

# Usoskin
data_uso = np.load('C:\Users\Bettina\ml\scRNAseq\data\Usoskin.npz')
cell_names_uso = data_uso['cells']
transcript_names_uso = data_uso['transcripts']
data_array_uso = data_uso['data']

np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\data_uso.txt", data_array_uso, delimiter=" ", fmt="%i")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\cell_names_uso.txt", cell_names_uso, delimiter=" ", fmt="%s")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\_transcript_names_uso.txt", transcript_names_uso, delimiter=" ", fmt="%s")

# Zeisel
data_zeisel = np.load('C:\Users\Bettina\ml\scRNAseq\data\Zeisel.npz')
cell_names_zeisel = data_zeisel['cells']
transcript_names_zeisel = data_zeisel['transcripts']
data_array_zeisel = data_zeisel['data']

np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\data_zeisel.txt", data_array_zeisel, delimiter=" ", fmt="%i")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\cell_names_zeisel.txt", cell_names_zeisel, delimiter=" ", fmt="%s")
np.savetxt("C:\Users\Bettina\ml\scRNAseq\data\_transcript_names_zeisel.txt", transcript_names_zeisel, delimiter=" ", fmt="%s")


