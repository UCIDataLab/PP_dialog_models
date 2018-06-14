__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import numpy as np
import csv


def get_lab_arr(lablist, n_labels=None):
    if not n_labels:
        maxlab = max(lablist)
        n_labels = maxlab + 1
    labarr = np.zeros((len(lablist), n_labels))
    labarr[range(len(lablist)), lablist] = 1
    return labarr


def flatten_nested_labels(nested_labs, lab_idx=None):
    if lab_idx is None:
        return [nested_labs[i][j] for i in range(len(nested_labs)) for j in range(len(nested_labs[i]))]
    else:
        return [nested_labs[i][lab_idx][j] for i in range(len(nested_labs)) for j in range(len(nested_labs[i][lab_idx]))]


def get_nested_labels(flattened_labs, len_list):
    fr, to = 0, 0
    nested_labs = []
    for slen in len_list:
        to = fr + slen
        nested_labs.append(flattened_labs[fr:to])
        fr = to
    return nested_labs


def save_sq_mat_with_labels(mat, lid2shortname, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([""] + lid2shortname)
        writer.writerows([[lid2shortname[i]]+row for i, row in enumerate(mat.tolist())])


def get_marginals(doc_label_mat):
    n_docs_in_labels = np.sum(doc_label_mat, axis=0)
    totalsum = np.sum(n_docs_in_labels)
    weights = n_docs_in_labels / float(totalsum)
    return weights


def get_marginals_from_y_seq(tr_true_y, n_labels):
    doc_lab_mat = get_lab_arr(tr_true_y, n_labels)
    return get_marginals(doc_lab_mat)
