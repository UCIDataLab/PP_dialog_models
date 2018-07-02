__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import numpy as np
from utils import get_lab_arr, save_sq_mat_with_labels, get_marginals
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix


def R_precision(true_y, pred_y):
    """
    Given two flat np.arrays, calculate R-precision score.

    Parameters
    ----------
    true_y : np.array
        True y's (true labels)
    pred_y : np.array
        Predicted y's (predicted labels)

    Returns
    -------
    float
    """
    R = np.sum(true_y)
    trueidxs = np.where(true_y)[0]
    retrieved_R_docs = np.argsort(pred_y)[::-1][:int(R)]

    n_true = len(set(retrieved_R_docs).intersection(set(trueidxs)))
    return n_true / float(R)


def get_accuracy(true_y, pred_y):
    """
    Given two lists or np.arrays, calculate the accuracy.

    Parameters
    ----------
    true_y : np.array
        True y's (true labels)
    pred_y : np.array
        Predicted y's (predicted labels)

    Returns
    -------
    float
    """
    assert len(true_y) == len(pred_y)
    numcorr = lambda a, b: np.where(np.array(a) == np.array(b))[0].shape[0]
    return numcorr(true_y, pred_y) / float(len(true_y)) * 100.0


def get_accuracy_per_lab(true_y, pred_y, n_labels):
    """
    Given two np.arrays, calculate an accuracy per label.
    Returns a list of accuracies with size n_label.
    `true_y` and `pred_y` should have values ranging from 0 to n_labels-1

    Parameters
    ----------
    true_y : np.array
        True y's (true labels)
    pred_y : np.array
        Predicted y's (predicted labels)
    n_labels : int
        Number of labels

    Returns
    -------
    list[int]
    """
    true_y_arr = get_lab_arr(true_y, n_labels)
    yhat_arr = get_lab_arr(pred_y, n_labels)

    accs = []
    n_utter = np.sum(true_y_arr, axis=0)
    for tidx in range(n_labels):
        accs.append(get_accuracy(true_y_arr[:, tidx], yhat_arr[:, tidx]))
    return accs, n_utter


def get_binary_classification_scores(true_y, pred_y, n_labels):
    """
    Given two np.arrays, calculate per-label binary scores,
    and return as a dictionary of arrays with size n_labels.
    `true_y` and `pred_y` should have values ranging from 0 to n_labels-1

    Parameters
    ----------
    true_y : np.array
        True y's (true labels)
    pred_y : np.array
        Predicted y's (predicted labels)
    n_labels : int
        Number of labels

    Returns
    -------
    dict[str: np.array]
        Dictionary with keys "precision", "recall", "auc", "rprecision", "f1score".
        Values are np.arrays with size `n_labels`.
    """
    true_y_arr = get_lab_arr(true_y, n_labels)
    yhat_arr = get_lab_arr(pred_y, n_labels)

    precisions = []
    recalls = []
    aucs = []
    rprecisions = []
    fscores = []

    for tidx in range(n_labels):
        if sum(true_y_arr[:, tidx]) == 0:
            print("WARNING: label index %d has 0 instance in the data. Binary scores for this label is set to 0.0" % tidx)
            precisions.append(0.0)
            recalls.append(0.0)
            fscores.append(0.0)
            aucs.append(0.0)
            rprecisions.append(0.0)
        else:
            precisions.append(precision_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
            recalls.append(recall_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
            fscores.append(f1_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
            aucs.append(roc_auc_score(true_y_arr[:, tidx], yhat_arr[:, tidx]))
            rprecisions.append(R_precision(true_y_arr[:, tidx], yhat_arr[:, tidx]))

    return {"precision":np.array(precisions), "recall":np.array(recalls),
            "auc":np.array(aucs), "rprecision":np.array(rprecisions), "f1score":np.array(fscores)}


def get_overall_scores_in_diff_metrics(true_y, pred_y, tr_doc_label_mat):
    """
    Get all the scores: accuracy and
                        both weighted and non-weighted average of all the binary scores available.

    Parameters
    ----------
    true_y : np.array
        True y's (true labels)
    pred_y : np.array
        Predicted y's (predicted labels)
    tr_doc_label_mat : np.array
        Document-label matrix for training data

    Returns
    -------
    dict[str:float]

    """
    n_states = tr_doc_label_mat.shape[1]
    marginals = get_marginals(tr_doc_label_mat)
    results = {}

    acc = get_accuracy(true_y, pred_y)
    results["accuracy"] = acc

    scores = get_binary_classification_scores(true_y, pred_y, n_states)
    for sc in sorted(scores.keys()):
        weighted = get_weighted_avg(scores[sc], marginals)
        notweighted = np.mean(scores[sc])
        results[sc+"_w"] = weighted
        results[sc] = notweighted

    return results


def print_row_of_diff_metrics(model_name, result_numbers, headers=None, filename="./overall_result.csv"):

    if headers is None:
        bin_metrics = ["precision", "recall", "auc", "rprecision", "f1score"]
        headers = ["model", "accuracy"] + [met+"_w" for met in bin_metrics]  + bin_metrics
        print(",".join(headers))

    with open(filename, 'a') as f:
        f.write(model_name)
        print(model_name),
        for met in headers[1:]:
            f.write(",%.4f" % result_numbers[met])
            print(",%.4f" % result_numbers[met]),
        f.write("\n")


def get_weighted_avg(score_list, weights):
    """
    Get weighted average of scores using the weights.
    """
    return np.dot(score_list, weights)


def get_weighted_avg_from_ymat(score_list, doc_label_mat):
    """
    Get weighted average, where the weights are from the marginal probabilities of training data.
    """
    weights = get_marginals(doc_label_mat)
    weighted_avg = np.dot(score_list, weights)
    return weighted_avg


def save_confusion_matrix(true_y, pred_y, lid2shortname, filename):
    """
    Save confusion matrix given true labels and the predicted labels.

    Parameters
    ----------
    true_y : np.array
        True y's (true labels)
    pred_y : np.array
        Predicted y's (predicted labels)
    lid2shortname : dict[int, str]
        Short names for each label index
    filename : str
        Path to the file where the confusion matrix will be saved.

    Returns
    -------
    np.array[float]
        with size (n_labels, n_labels)

    """
    conf = confusion_matrix(true_y, pred_y)
    save_sq_mat_with_labels(conf, lid2shortname, filename)
    return conf
