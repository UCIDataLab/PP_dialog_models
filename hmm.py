__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import numpy as np


def viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """
    Run Viterbi algorithm.

    Parameters
    ----------
    emission_scores : np.array
        Emission scores or log probabilities
    trans_scores : np.array
        Transition matrix, scores or log probabilities
    start_scores : np.array
        Start transition scores or log probabilities
    end_scores : np.array
        End transition scores or log probabilities

    Returns
    -------
    tuple
        Returns a tuple of
            score of the best sequence,
            array of integers representing the best sequence, and
            the trellis table for calculating the scores.
    """
    L = start_scores.shape[0]  # Number of labels
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]  # Length of tokens

    # The label that led to the time i for label l
    T = np.zeros((N, L), dtype=np.float)
    backpointers = np.full((N,L), 0, dtype=np.int)
    y = []

    # Initial values
    T[0,:] = start_scores + emission_scores[0, :]

    for i in range(1, N):
        for l in range(L):
            tmp_arr = trans_scores[:, l] + T[i-1, :]
            backpointers[i, l] = np.argmax(tmp_arr)
            T[i, l] = emission_scores[i, l] + np.max(tmp_arr)

    final_scores = T[-1,:] + end_scores
    yhat = np.argmax(final_scores)
    y.append(yhat)

    for i in range(N-1, 0, -1):
        yhat = backpointers[i, yhat]
        y.append(yhat)

    return np.max(final_scores), y[::-1], T


def viterbi_with_multiple_transitions(emission_scores, trans_ids, trans_scores_list, start_scores, end_scores):
    """
    Viterbi code, when there are multiple transitions.
    trans_ids should be given in addition to the other parameters

    Parameters
    ----------
    emission_scores : np.array
    trans_ids : np.array
        List of indexes of transition matrix
    trans_scores_list : np.array
        List of transition matrices
    start_scores : np.array
    end_scores : np.array

    Returns
    -------
    tuple
        Returns a tuple of
            score of the best sequence,
            array of integers representing the best sequence, and
            the trellis table for calculating the scores.

    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores_list[0].shape[0] == L
    assert trans_scores_list[0].shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    assert trans_ids.shape[0] == N
    K = len(trans_scores_list)

    # The label that led to the time i for label l
    T = np.zeros((N, L), dtype=np.float)
    backpointers = np.full((N,L), 0, dtype=np.int)
    y = []

    # Initial values
    T[0,:] = start_scores + emission_scores[0, :]

    for i in range(1, N):
        spkrid = trans_ids[i] if trans_ids[i] < K else K-1
        for l in range(L):
            tmp_arr = trans_scores_list[spkrid][:, l] + T[i-1, :]
            backpointers[i, l] = np.argmax(tmp_arr)
            T[i, l] = emission_scores[i, l] + np.max(tmp_arr)

    final_scores = T[-1,:] + end_scores
    yhat = np.argmax(final_scores)
    y.append(yhat)

    for i in range(N-1, 0, -1):
        yhat = backpointers[i, yhat]
        y.append(yhat)

    return (np.max(final_scores), y[::-1], T)


def get_trans_prob_from_cnt(trans_cnt_mat, proportional_prior=True, prior_sum=0.5, log=True):
    """
    Used inside of the method get_transition_mat.
    From the counts of the transitions, returns the probability matrix.

    Parameters
    ----------
    trans_cnt_mat : np.array
        with size (n_states, n_states)
    proportional_prior : bool
        Use proportional Dirichlet prior instead of flat priors.
        Priors are proportional to the marginal probabilities.
    prior_sum : float
        The total sum of the priors. \sum_k prior_k
    log : bool
        Returns log probabilities if set to True (default)

    Returns
    -------
    np.array
        with size (n_states, n_states)

    """
    n_states = trans_cnt_mat.shape[0]
    if proportional_prior:
        alpha = np.sum(trans_cnt_mat, axis=0)
        alpha /= np.sum(alpha) * prior_sum
    else:
        alpha = np.ones(n_states) * prior_sum / float(n_states)

    # Use alpha as Dirichlet prior
    trans_cnts_Dir = trans_cnt_mat + alpha
    trans_prob = (trans_cnts_Dir) / (np.tile(np.sum(trans_cnts_Dir, axis=1), (n_states, 1))).T
    if log:
        return np.log(trans_prob)
    else:
        return trans_prob


def get_transition_mat(true_lab_lists, n_states, proportional_prior=True, prior_sum=0.5, log=True):
    """
    Returns a transition matrix.

    Parameters
    ----------
    true_lab_lists : list[list[int]]
        A list of sessions, where a session is a list of labels,
    n_states : int
        Number of states that exist in the data
    proportional_prior : bool
        Use proportional Dirichlet prior instead of flat priors.
        Priors are proportional to the marginal probabilities.
    prior_sum : float
        The total sum of the priors. \sum_k prior_k

    Returns
    -------
    np.array
        np.array with size (n_states, n_states)

    """
    trans_cnts = np.zeros((n_states, n_states))

    for lab_seq in true_lab_lists:
        for l1, l2 in zip(lab_seq, lab_seq[1:]):
            trans_cnts[l1, l2] += 1

    return get_trans_prob_from_cnt(trans_cnts, proportional_prior, prior_sum, log)


def get_spkr_transition_mat(true_lab_lists, spkr_lists, n_states, get_patient_mat=False,
                            proportional_prior=True, prior_sum=0.5, log=True):
    """
    Get a dictionary of transition matrices and transition count matrices, where the keys are
    'MD_trans_mat', 'MD_cnts', 'OTH_trans_mat', 'OTH_cnts'.
    Keys 'PT_trans_mat' and 'PT_cnts' only exist when the parameter get_patient_mat is set to True.

    Parameters
    ----------
    true_lab_lists : list[list[int]]
        Nested list of labels.
        A list of sessions, where a session is a list of labels.
    spkr_lists : list[list[int]]
        A list of sessions, where a session is a list of speaker IDs of the utterances in that session.
    n_states : int
        Number of states (number of different labels)
    get_patient_mat : bool
        Also get the patient matrix in addition to MD and OTH matrix.
    proportional_prior : bool
        Use proportional Dirichlet prior instead of flat priors.
        Priors are proportional to the marginal probabilities.
    prior_sum : float
        The total sum of the priors. \sum_k prior_k
    log : bool
        Returns log probabilities if set to True (default)

    Returns
    -------
    dict[int, np.array]

    """
    # 1. * -> MD (idx: 0)
    # 2. * -> not MD (others, PT + OTHER)
    # 3. * -> PT (idx: 1)
    trans_cnts_md = np.zeros((n_states, n_states))
    trans_cnts_oth = np.zeros((n_states, n_states))  # Others include PT
    if get_patient_mat:
        trans_cnts_pt = np.zeros((n_states, n_states))

    for lab_seq, spkr_seq in zip(true_lab_lists, spkr_lists):
        for i1, i2 in zip(range(len(lab_seq)), range(len(lab_seq))[1:]):
            if spkr_seq[i2] == 0: # 0 idx is MD
                trans_cnts_md[lab_seq[i1], lab_seq[i2]] += 1
            else:
                if get_patient_mat and spkr_seq[i2] == 1:
                    trans_cnts_pt[lab_seq[i1], lab_seq[i2]] += 1
                trans_cnts_oth[lab_seq[i1], lab_seq[i2]] += 1

    trans_mat_md = get_trans_prob_from_cnt(trans_cnts_md, proportional_prior, prior_sum, log)
    trans_mat_oth = get_trans_prob_from_cnt(trans_cnts_oth, proportional_prior, prior_sum, log)
    result = {"MD_cnts":trans_cnts_md, "MD_trans_mat":trans_mat_md,
              "OTH_cnts":trans_cnts_oth, "OTH_trans_mat":trans_mat_oth}
    if get_patient_mat:
        result["PT_cnts"] = trans_cnts_pt
        trans_mat_pt = get_trans_prob_from_cnt(trans_cnts_oth, proportional_prior, prior_sum, log)
        result["PT_trans_mat"] = trans_mat_pt

    return result


def get_start_end_prob(true_lab_lists, n_states, alpha=0.001):
    """
    Parameters
    ----------
    true_lab_lists : list[list[int]]
        Nested list of labels.
        A list of sessions, where a session is a list of labels.
    n_states : int
        Number of states or labels
    alpha : float
        Small number for padding (Dirichlet prior)

    Returns
    -------
    tuple(np.array, np.array)
        Each element in a tuple has size (n_states,1).
        Returns the start and the end probabilities.
    """
    initial_prob = np.zeros(n_states) + alpha
    end_prob = np.zeros(n_states) + alpha
    for lab_seq in true_lab_lists:
        # label at the start of each session
        initial_prob[lab_seq[0]] += 1
        # label at the end of each session
        end_prob[lab_seq[-1]] += 1
    initial_prob /= initial_prob.sum()
    end_prob /= end_prob.sum()
    return initial_prob, end_prob


def convert_class_prob_to_log_emission_prob(class_prob, marginals, p_utter=1e-5):
    """
    Applies Bayes rule to convert the class output probabilities
    given an utterance p(t|u) to the emission probabilities p(u|t).
    p(u|t) = p(t|u) * p(u) / p(t).
    We assume that p(u) is the same for all the utterances, and p(t) is the
    marginal probability for topic t.

    Parameters
    ----------
    class_prob : np.array
        np.array of size (N_s, T), where N_s: number of utterances in the session
        and T: number of labels or states.
    marginals : np.array
        np.array of size (T, 1).  (T: number of labels or states.)

    Returns
    -------
    np.array
        Size should be the same as class_prob.
        Returns the log emission probabilities.
    """
    N = class_prob.shape[0]
    if p_utter > 0:
        log_putter = np.log(p_utter)  # we can ignore this but ..
    else:
        log_putter = 0.0
    log_p_u_tk = np.log(class_prob) + log_putter - np.log(np.tile(marginals, (N,1)))
    return log_p_u_tk
