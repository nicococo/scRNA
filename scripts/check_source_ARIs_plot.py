from utils import *
import matplotlib.pyplot as plt
from scipy import stats

#  generate plot
foo = np.load('C:\Users\Bettina\PycharmProjects2\scRNA_new\scripts\\ari_pre_experiment_full_n_src_final.npz')
source_aris_NMF_NMF = foo['source_aris_NMF_NMF']
source_aris_NMF_SC3 = foo['source_aris_NMF_SC3']
source_aris_SC3_NMF = foo['source_aris_SC3_NMF']
source_aris_SC3_SC3 = foo['source_aris_SC3_SC3']
n_src = foo['n_src']

ste_NMF_NMF = stats.sem(source_aris_NMF_NMF, ddof=0, axis=1)
ste_NMF_SC3 = stats.sem(source_aris_NMF_SC3, ddof=0, axis=1)
ste_SC3_NMF = stats.sem(source_aris_SC3_NMF, ddof=0, axis=1)
ste_SC3_SC3 = stats.sem(source_aris_SC3_SC3, ddof=0, axis=1)

markers, caps, bars = plt.errorbar(n_src, np.mean(source_aris_NMF_NMF, axis=1), fmt='k', yerr=ste_NMF_NMF, linewidth=2.0)
[bar.set_alpha(0.3) for bar in bars]
[cap.set_alpha(0.3) for cap in caps]
markers, caps, bars = plt.errorbar(n_src, np.mean(source_aris_NMF_SC3, axis=1), fmt='--k', yerr=ste_NMF_SC3, linewidth=2.0)
[bar.set_alpha(0.3) for bar in bars]
[cap.set_alpha(0.3) for cap in caps]
markers, caps, bars = plt.errorbar(n_src, np.mean(source_aris_SC3_NMF, axis=1), fmt='--r', yerr=ste_SC3_NMF, linewidth=2.0)
[bar.set_alpha(0.3) for bar in bars]
[cap.set_alpha(0.3) for cap in caps]
markers, caps, bars = plt.errorbar(n_src, np.mean(source_aris_SC3_SC3, axis=1), fmt='r', yerr=ste_SC3_SC3, linewidth=2.0)
[bar.set_alpha(0.3) for bar in bars]
[cap.set_alpha(0.3) for cap in caps]

plt.legend(['method NMF, truth NMF', 'method NMF, truth SC3', 'method SC3, truth NMF', 'method SC3, truth SC3'], loc=4)
plt.xlim([0,600])
plt.ylim([0,1])
plt.xlabel('Data size (truth from whole dataset of 1670)')
plt.ylabel('ARI')

plt.show()

print('Done.')

