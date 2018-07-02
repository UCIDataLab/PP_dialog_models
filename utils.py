__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import numpy as np
import csv


def get_lab_arr(lablist, n_labels=None):
    """
    Convert a list of labels (1-d array with size (N,))
    to a matrix (2-d array with size (N, L)), where
     N = number of data points
     L = number of labels

    Parameters
    ----------
    lablist : list or np.array
        List of labels.
    n_labels : int
        Number of labels that exist in lablist.

    Returns
    -------
    np.array
        One-hot-encoded 2-d array, where the column has the size of n_labels.
    """
    if not n_labels:
        maxlab = max(lablist)
        n_labels = maxlab + 1
    labarr = np.zeros((len(lablist), n_labels))
    labarr[range(len(lablist)), lablist] = 1
    return labarr


def flatten_nested_labels(nested_labs, lab_idx=None):
    """
    Convert a session-level nested list into an utterance-level flattened list.
    """
    if lab_idx is None:
        return [nested_labs[i][j] for i in range(len(nested_labs)) for j in range(len(nested_labs[i]))]
    else:
        return [nested_labs[i][lab_idx][j] for i in range(len(nested_labs)) for j in range(len(nested_labs[i][lab_idx]))]


def get_nested_labels(flattened_labs, len_list):
    """
    Given an utterance-level flattened list of labels, and the lengths (in utterance) of each sessions,
    get a session-level nested list of labels.

    Parameters
    ----------
    flattened_labs : list[int] or np.array[int]
        1-d list or np.array with size (N, )
    len_list : list[int]
        1-d list or np.array with size (S, )

    Returns
    -------
    list[list[int]]
        with size (S, N_s)

    * N: total number of utterances
    * S: number of sessions
    * N_s : number of utterances in each session
    """
    fr, to = 0, 0
    nested_labs = []
    for slen in len_list:
        to = fr + slen
        nested_labs.append(flattened_labs[fr:to])
        fr = to
    return nested_labs


def save_sq_mat_with_labels(mat, lid2shortname, filename):
    """
    Save square matrix with labels.

    Parameters
    ----------
    mat : np.array
        with size (n_labels, n_labels)
    lid2shortname : dict[int, str]
        Short names for each lab index. Used as headers of each columns and rows.
    filename : str
        Path to the file where the matrix is saved as csv file.
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([""] + lid2shortname)
        writer.writerows([[lid2shortname[i]]+row for i, row in enumerate(mat.tolist())])


def get_marginals(doc_label_mat):
    """
    Get marginal probabilities given a one-hot-encoded (n_docs X n_label) 2-d array.
    """
    n_docs_in_labels = np.sum(doc_label_mat, axis=0)
    totalsum = np.sum(n_docs_in_labels)
    weights = n_docs_in_labels / float(totalsum)
    return weights


def get_marginals_from_y_seq(tr_true_y, n_labels):
    """
    Get marginal probabilities given a sequence of label integers.
    List of label indices (tr_true_y) should range [0, n_labels).
    """
    doc_lab_mat = get_lab_arr(tr_true_y, n_labels)
    return get_marginals(doc_lab_mat)
