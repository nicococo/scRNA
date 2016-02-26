import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

import pdb
# import matplotlib.pyplot as plt

# import pickle
# pickel.dump und Pickel.load
# for data type: type()


# data.keys()
# pdb.set_trace()
data_ting = np.load('C:\Users\Bettina\ml\scRNAseq\data\Ting.npz')
cell_names_ting = data_ting['cells']
transcript_names_ting = data_ting['transcripts']
data_array_ting = data_ting['data']

data_uso = np.load('C:\Users\Bettina\ml\scRNAseq\data\Usoskin.npz')
cell_names_uso = data_uso['cells']
transcript_names_uso = data_uso['transcripts']
data_array_uso = data_uso['data']

data_zeisel = np.load('C:\Users\Bettina\ml\scRNAseq\data\Zeisel.npz')
cell_names_zeisel = data_zeisel['cells']
transcript_names_zeisel = data_zeisel['transcripts']
data_array_zeisel = data_zeisel['data']


# pdb.set_trace()


def split_data(org_data, org_cell_names):
    num_cell = len(org_cell_names)
    indices = np.random.permutation(num_cell)
    [d1, d2] = split_list(indices, wanted_parts=2)
    new_data_1 = org_data[:, d1]
    new_cell_names_1 = org_cell_names[d1]
    new_data_2 = org_data[:, d2]
    new_cell_data_2 = org_cell_names[d2]
    labels = [0] * num_cell
    labels = np.array(labels)
    labels[d2] = [1]*len(d2)
    return [new_data_1, new_cell_names_1, new_data_2, new_cell_data_2, labels]


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i*length / wanted_parts: (i+1)*length / wanted_parts] for i in range(wanted_parts)]


def cross_validation(data, labels):
    kf = KFold(len(labels), n_folds=10, shuffle=True)
    # pdb.set_trace()
    accs = []
    for train, test in kf:
        # print("TRAIN:", train, "TEST:", test)
        x_train, x_test, y_train, y_test = data[:, train], data[:, test], labels[train], labels[test]
        clf_now = SVC()
        clf_now.fit(np.transpose(x_train), y_train)
        accs.append(clf_now.score(np.transpose(x_test), y_test))
    return accs


def intersect(a, b):
    return list(set(a) & set(b))

# Find matching transcripts
print(len(intersect(transcript_names_ting, transcript_names_uso)))
print(len(intersect(transcript_names_ting, transcript_names_zeisel)))
print(len(intersect(transcript_names_uso, transcript_names_zeisel)))

# Prediction random split in Ting
[data_1_ting, cell_names_1_ting, data_2_ting, cell_data_2_ting, labels_all_ting] = split_data(data_array_ting, cell_names_ting)
clf = SVC()
clf.fit(np.transpose(data_array_ting), labels_all_ting)
accuracies_ting = cross_validation(data_array_ting, labels_all_ting)
print("Prediction random split in Ting - Mean accuracy: ", np.mean(accuracies_ting))

# Prediction random split in Usoskin
[data_1_uso, cell_names_1_uso, data_2_uso, cell_data_2_uso, labels_all_uso] = split_data(data_array_uso, cell_names_uso)
clf = SVC()
clf.fit(np.transpose(data_array_uso), labels_all_uso)
accuracies_uso = cross_validation(data_array_uso, labels_all_uso)
print("Prediction random split in Usososkin - Mean accuracy: ", np.mean(accuracies_uso))

# Prediction random split in Zeisel
[data_1_zeisel, cell_names_1_zeisel, data_2_zeisel, cell_data_2_zeisel, labels_all_zeisel] = split_data(data_array_zeisel, cell_names_zeisel)
clf = SVC()
clf.fit(np.transpose(data_array_zeisel), labels_all_zeisel)
accuracies_zeisel = cross_validation(data_array_zeisel, labels_all_zeisel)
print("Prediction random split in Zeisel - Mean accuracy: ", np.mean(accuracies_zeisel))

# Prediction Ting vs. Usoskin
transcripts_to_keep = intersect(transcript_names_ting, transcript_names_uso)
new_data_ting = np.asarray([data_array_ting[transcript_names_ting.tolist().index(transcript)] for transcript in transcripts_to_keep])
new_data_uso = np.asarray([data_array_uso[transcript_names_uso.tolist().index(transcript)] for transcript in transcripts_to_keep])
big_data = np.concatenate([new_data_ting, new_data_uso], axis=1)
big_labels = np.concatenate([[0]*len(cell_names_ting), [1]*len(cell_names_uso)])
accuracies = cross_validation(big_data, big_labels)
# print(accuracies)
print("Prediction Ting vs. Usoskin - Mean accuracy: ", np.mean(accuracies))

# Prediction Ting vs. Zeisel
transcripts_to_keep = intersect(transcript_names_ting, transcript_names_zeisel)
new_data_ting = np.asarray([data_array_ting[transcript_names_ting.tolist().index(transcript)] for transcript in transcripts_to_keep])
new_data_zeisel = np.asarray([data_array_zeisel[transcript_names_zeisel.tolist().index(transcript)] for transcript in transcripts_to_keep])
big_data = np.concatenate([new_data_ting, new_data_zeisel], axis=1)
big_labels = np.concatenate([[0]*len(cell_names_ting), [1]*len(cell_names_zeisel)])
accuracies = cross_validation(big_data, big_labels)
# print(accuracies)
print("Prediction Ting vs. Zeisel - Mean accuracy: ", np.mean(accuracies))

# Prediction Usoskin vs. Zeisel
transcripts_to_keep = intersect(transcript_names_uso, transcript_names_zeisel)
new_data_uso = np.asarray([data_array_uso[transcript_names_uso.tolist().index(transcript)] for transcript in transcripts_to_keep])
new_data_zeisel = np.asarray([data_array_zeisel[transcript_names_zeisel.tolist().index(transcript)] for transcript in transcripts_to_keep])
big_data = np.concatenate([new_data_uso, new_data_zeisel], axis=1)
big_labels = np.concatenate([[0]*len(cell_names_uso), [1]*len(cell_names_zeisel)])
accuracies = cross_validation(big_data, big_labels)
# print(accuracies)
print("Prediction Usoskin vs. Zeisel - Mean accuracy: ", np.mean(accuracies))

pdb.set_trace()
