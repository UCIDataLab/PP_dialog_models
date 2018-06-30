__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import cPickle as cp
import numpy as np
import os

from mhddata import MHDTrainData, MHDTestData
from evaluate import get_binary_classification_scores, get_accuracy, get_weighted_avg
from utils import get_marginals_from_y_seq, flatten_nested_labels, get_lab_arr
from hmm import convert_class_prob_to_log_emission_prob, viterbi, viterbi_with_multiple_transitions,\
    get_transition_mat, get_spkr_transition_mat, get_start_end_prob

from sklearn.linear_model import LogisticRegression
import sklearn
if int(sklearn.__version__.split(".")[1]) > 17:
    from sklearn.model_selection import GridSearchCV
else:
    from sklearn.grid_search import GridSearchCV


class DialogModel():
    """
    Base class for training & testing topic classification on MHD dialog data.
    The base class is mainly used when loading the results of the model
    that was trained outside this package. (method load_results())
    Other models inherits this model.
    """

    def __init__(self):
        self.tr_data = None  # training data (MHDTrainData)
        self.te_data = None  # test data (MHDTestData)
        self.vectorizer = None  # CountVectorizer for creating BOW/TFIDF
        self.model = None  # Model (e.g. an object of LogisticRegression classifier)
        self.result = None  # Result object, of class DialogResult

    def load_model(self, model_file):
        pass

    def fit_model(self, data_file, model_file):
        pass

    def load_train_data(self, data_file, nouns_only=False, ignore_case=True,
                        remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                        label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                        min_dfreq=0, max_dfreq=0.8, min_sfreq=20,
                        token_pattern=r"(?u)[A-Za-z\?\!\-\.']+", verbose=1,
                        corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):
        """
        Loads and processes training data.
        Also creates marginal probabilities.
        Arguments are the same as MHDTrainData.
        """

        self.tr_data = MHDTrainData(data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                 remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                 label_mappings=label_mappings, ngram_range=ngram_range,
                 max_np_len=max_np_len, min_wlen=min_wlen,
                 min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                 token_pattern=token_pattern, verbose=verbose,
                 corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        self.marginals = get_marginals_from_y_seq(self.tr_data.uid2lid, self.tr_data.n_labels)

    def load_test_data(self, te_data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True,
                 proper_nouns_dir="./stopwordlists", min_wlen=1,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+", verbose=1, reload_corpus=True,
                 corpus_pkl='./corpus_te.pkl', tr_label_pkl='./label.pkl', tr_vocab_pkl='./vocab.pkl'):
        """
        Loads and processes test data.
        Arguments are the same as MHDTestData.
        """
        self.te_data = MHDTestData(te_data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                                   remove_numbers=remove_numbers, sub_numbers=sub_numbers,
                                   proper_nouns_dir=proper_nouns_dir,
                                   min_wlen=min_wlen, token_pattern=token_pattern, verbose=verbose,
                                   reload_corpus=reload_corpus,
                                   corpus_pkl=corpus_pkl, tr_label_pkl=tr_label_pkl, tr_vocab_pkl=tr_vocab_pkl)

        self.n_labels = self.te_data.n_labels

    def predict(self, te_data_file):
        pass

    def load_results(self, te_data_file, model_info='dialog_model', marginals=None,
                     predictions='./model123_pred.pkl', output_probs='./model123_prob.pkl',
                     verbose=1, output_filename='./utter_level_result.txt',
                     corpus_pkl='./corpus_te.pkl', tr_label_pkl='./label.pkl', tr_vocab_pkl='./vocab.pkl'):
        """
        This is used when you are not running predictions but just want to load the results
        to calculate scores or to plug the results into HMM.

        Parameters
        ----------
        te_data_file : str
            Path to the test data file.
        model_info : str
            Model information. e.g. "HMM_LR_OVR"
        marginals : np.array or None
            Marginal probabilities for labels. If not given, weighted average scores will not provided.
        predictions : str
            Path to the pickle file that has nested list of predictions
        output_probs : str
            Path to the pickle file that has nested list of output probabilities
        verbose : int
            The level of verbosity in range [0,3]
        corpus_pkl : str
            Path to the pickle file where corpus related test data is (will be) saved.
        tr_label_pkl : str
            Path to the pickle file where label related training data is saved
        tr_vocab_pkl : str
            Path to the pickle file where vocab related training data is saved.

        Returns
        -------
        DialogResult

        """

        self.load_test_data(te_data_file, verbose=verbose,
                            corpus_pkl=corpus_pkl, tr_label_pkl=tr_label_pkl, tr_vocab_pkl=tr_vocab_pkl)
        self.model_info = model_info

        n_labs = self.te_data.n_labels
        te_data_nested = self.te_data.get_utter_level_data_from_sids(sorted(self.te_data.sstt2uid.keys()))
        ulists, docs, labs = te_data_nested

        if os.path.exists(output_probs):
            with open(output_probs, 'rb') as f:
                prob = cp.load(f)
        else:
            prob = None
        if os.path.exists(predictions):
            with open(predictions, 'rb') as f:
                pred = cp.load(f)
        else:
            pred = None

        self.result = DialogResult(n_labs, pred, prob, marginals, model_info)

        if self.te_data.has_label:
            if verbose > 0:
                print("Calculate score")
            self.result.get_scores(labs)
            if verbose > 0:
                print("Printing utterance-level results to file "+output_filename)
            self.result.print_utter_level_results(ulists, docs, labs, self.te_data.lid2name,
                                                  filename=output_filename)
        return self.result


class LogRegDialogModel(DialogModel):
    """
    Logistic Regression model for running topic classification on MHD dialog data.
    """

    def __init__(self, lr_type="ovr"):
        DialogModel.__init__(self)
        self.lr_type = lr_type
        self.trainX, self.trainy = None, None

    def grid_search_parameter(self, data_file, C_values=None,
                              penalty_type="l2", solver='lbfgs',
                              n_fold=3, verbose=1,
                              corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):
        """
        A method that does grid search over the training data and finds the best
        regularization constant and returns the value.
        It creates self.grid_search, which is an obj of sklearn's GridSearchCV.

        Parameters
        ----------
        data_file : str
            File path to the training data.
        C_values : iterable or None
            Values to be searched to find the best C value.
            If None (default), C value is searched over the values np.arange(0.6,3,0.1)
        penalty_type : str
            Penalty type of logistic regression. "l2" is default.
        solver : str
            "lbfgs" is set to default
        n_fold : int
            The number of cross validation folds.
            Default = 3. Large number is not recommended since there are labels that are quite rare.
        verbose : int
            The level of verbosity in range [0,3]
        corpus_pkl : str
            Path to the pickle file where corpus related data is (will be) saved.
        label_pkl : str
            Path to the pickle file where label related data is (will be) saved
        vocab_pkl : str
            Path to the pickle file where vocab related data is (will be) saved.

        Returns
        -------
        float
            Best parameter (regularization constant)
        """
        if self.tr_data is None:
            self.load_train_data(data_file, verbose=verbose,
                                 corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        if self.trainX is None or self.trainy is None:
            trainX, self.vectorizer = self.tr_data.fit_bow(self.tr_data.corpus_txt, tfidf=True)
            trainy = self.tr_data.uid2lid
            self.trainX = trainX
            self.trainy = trainy
        else:
            trainX = self.trainX
            trainy = self.trainy

        # Just do 3-fold cross-validation
        grid = GridSearchCV(LogisticRegression(penalty=penalty_type, solver=solver, multi_class=self.lr_type),
                            {'C': C_values}, cv=n_fold, verbose=verbose)
        grid.fit(trainX, trainy)
        self.grid_search = grid
        print("Best regularization constant: %.2f" % grid.best_params_['C'])
        return grid.best_params_

    def fit_model(self, data_file, penalty_type="l2", reg_const=1.0,
                  solver='lbfgs', model_file='./lrdialog.pkl', verbose=1,
                  corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):
        """
        Loads training data from `data_file`, processes the data,
        fits the LR model, and then saves the model.
        Updates self.model.

        Parameters
        ----------
        data_file : str
            File path to the training data.
        penalty_type : str
            Penalty type of logistic regression. "l2" is default.
        reg_const : float
            Regularization constant. Inverse of regularization strength.
            Smaller values specify larger regularization.
            (You can plug in the number after running the method 'grid_search_parameter')
            Default is set to 1.0.
        solver : str
            "lbfgs" is set to default.
        model_file : str
            File path to the pickle file of trained model.
        verbose : int
            The level of verbosity.
        corpus_pkl : str
            Path to the pickle file where corpus related data is (will be) saved.
        label_pkl : str
            Path to the pickle file where label related data is (will be) saved
        vocab_pkl : str
            Path to the pickle file where vocab related data is (will be) saved.

        Returns
        -------
        LogisticRegression

        """
        if self.tr_data is None:
            self.load_train_data(data_file, verbose=verbose, corpus_pkl=corpus_pkl,
                                 label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        if self.trainX is None or self.trainy is None:
            trainX, self.vectorizer = self.tr_data.fit_bow(self.tr_data.corpus_txt, tfidf=True)
            trainy = self.tr_data.uid2lid
            self.trainX = trainX
            self.trainy = trainy
        else:
            trainX = self.trainX
            trainy = self.trainy

        self.model = LogisticRegression(penalty=penalty_type, C=reg_const,
                                        multi_class=self.lr_type, solver=solver)
        self.model = self.model.fit(trainX, trainy)
        # Saves model into pkl
        with open(model_file, 'wb') as f:
            print("Saving Logistic regression model to "+ model_file)
            cp.dump((self.model, self.vectorizer, self.marginals), f, protocol=cp.HIGHEST_PROTOCOL)
        return self.model

    def load_model(self, model_file='./lrdialog.pkl'):
        """
        Loads trained model from a pickle file path.
        self.model, self.vectorizer, self.marginals are updated.
        """
        with open(model_file, 'rb') as f:
            print("Loading Logistic regression model to "+ model_file)
            self.model, self.vectorizer, self.marginals = cp.load(f)

    def predict(self, te_data_file, verbose=1, output_filename='./utter_level_results.txt',
                corpus_pkl='./corpus_te.pkl', tr_label_pkl='./label.pkl', tr_vocab_pkl='./vocab.pkl'):
        """
        Loads test data from 'te_data_file' and processes the data.
        Run prediction using the trained model.
        Also calculate the scores if the test data has labels.

        Parameters
        ----------
        te_data_file
        verbose : int
            The level of verbosity in range [0,3]
        output_filename : str
            Path to the file where the utter-level result will be saved.
        corpus_pkl : str
            Path to the pickle file where corpus related test data is (will be) saved.
        tr_label_pkl : str
            Path to the pickle file where label related training data is saved
        tr_vocab_pkl : str
            Path to the pickle file where vocab related training data is saved.

        Returns
        -------

        """
        if not (self.model and self.vectorizer):
            print ("ERROR: Train or load the model first")
            return
        self.load_test_data(te_data_file, verbose=verbose,
                            corpus_pkl=corpus_pkl, tr_label_pkl=tr_label_pkl, tr_vocab_pkl=tr_vocab_pkl)

        n_labs = self.te_data.n_labels
        self.model_info = "_".join(["LogReg", self.lr_type, self.model.penalty, str(self.model.C)])

        te_data_nested = self.te_data.get_utter_level_data_from_sids(sorted(self.te_data.sstt2uid.keys()))
        ulists, docs, labs = te_data_nested

        outputprobs = []
        yhats = []
        for sidx in range(len(docs)):
            testX_s = MHDTestData.transform_bow(docs[sidx], self.vectorizer)

            outputprob = self.model.predict_proba(testX_s)
            outputprobs.append(outputprob)
            yhat = self.model.predict(testX_s)
            yhats.append(yhat)

        self.result = DialogResult(n_labs, yhats, outputprobs, self.marginals, self.model_info)
        if self.te_data.has_label:
            if verbose > 0:
                print("Calculate score")
            self.result.get_scores(labs)
            if verbose > 0:
                print("Printing utterance-level results to file "+output_filename)
            self.result.print_utter_level_results(ulists, docs, labs, self.te_data.lid2name,
                                                  filename=output_filename)
        else:
            if verbose > 0:
                print("Printing utterance-level results to file "+output_filename)
            self.result.print_utter_level_results_without_true_lab(ulists, docs, self.te_data.lid2name,
                                                                   filename=output_filename)
        return self.result


class HMMDialogModel(DialogModel):
    """
    Hidden Markov Model that is run on top of other model or a set of probabilities.
    HMM is not jointly trained with the other model that generates output probabilities.
    Pre-trained model with output probabilities are given as an input.
    """

    def __init__(self, base_model):
        """
        Parameters
        ----------
        base_model : DialogModel
            A model whose output probabilities are used for emission probabilities in HMM.
        """

        DialogModel.__init__(self)
        if base_model.result is None:
            print("ERROR: Base model for HMMDialogModel should have .result in it.")
            print("       .predict() method should be applied before running HMM")
            return
        self.base_model = base_model

        # The following variables will be updated at fit_model()
        self.log_transitions, self.marginals = None, None
        self.start_prob, self.end_prob = None, None
        self.log_start_prob, self.log_end_prob = None, None

    def fit_model(self, data_file, model_file='./hmmdialog.pkl', verbose=1,
                  corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):
        """
        Loads training data from 'data_file', processes the data,
        gets transition matrix and other probabilities and saves the model.
        self.start_prob, self.end_prob, self.log_transition

        fit
        Parameters
        ----------
        data_file : str
            Path to the training data file.
        model_file : str
            Path to the pickle file where the Hidden Markov Model related data will be saved.
        verbose : int
            The level of verbosity in range [0,3]
        corpus_pkl : str
            Path to the pickle file where corpus related data is (will be) saved.
        label_pkl : str
            Path to the pickle file where label related data is (will be) saved
        vocab_pkl : str
            Path to the pickle file where vocab related data is (will be) saved.

        Returns
        -------

        """
        self.load_train_data(data_file, verbose=verbose,
                             corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)
        self.n_labels = self.tr_data.n_labels
        tr_data_nested = self.tr_data.get_utter_level_data_from_sids(self.tr_data.sstt2uid.keys())
        ulists_tr, docs_tr, labs_tr = tr_data_nested

        self.log_transitions = get_transition_mat(labs_tr, self.n_labels,
                                                  proportional_prior=True, prior_sum=0.5, log=True)

        self.start_prob, self.end_prob = get_start_end_prob(labs_tr, self.n_labels)
        self.log_start_prob = np.log(self.start_prob)
        self.log_end_prob = np.log(self.end_prob)
        self.marginals = get_marginals_from_y_seq(self.tr_data.uid2lid, self.n_labels)

        with open(model_file, 'wb') as f:
            print("Saving model to "+ model_file)
            data_to_save = (self.log_transitions, self.start_prob, self.end_prob, self.marginals)
            cp.dump(data_to_save, f, protocol=cp.HIGHEST_PROTOCOL)

    def load_model(self, model_file='./hmmdialog.pkl'):
        """
        Loads model (trained probabilities) from a pickle
        """
        print("Loading model from "+ model_file)
        with open(model_file, 'rb') as f:
            self.log_transitions, self.start_prob, self.end_prob, self.marginals = cp.load(f)
            self.log_start_prob = np.log(self.start_prob)
            self.log_end_prob = np.log(self.end_prob)

    def predict_viterbi(self, te_data_file, verbose=1, output_filename="./utter_level_result.txt",
                        corpus_pkl='./corpus_te.pkl', tr_label_pkl='./label.pkl', tr_vocab_pkl='./vocab.pkl'):
                         # spkr_transitions=False):

        """
        Viterbi decoding using output probabilites from the base model,
        marginal probabilities, and transition probabilities.

        Parameters
        ----------
        te_data_file : str
            Path to the test data file
        verbose : int
            The level of verbosity in range [0,3]
        output_filename : str
            Path to the utterance-level result file.
        corpus_pkl : str
            Path to the pickle file where corpus related test data is (will be) saved.
        tr_label_pkl : str
            Path to the pickle file where label related training data is saved
        tr_vocab_pkl : str
            Path to the pickle file where vocab related training data is saved.
        """
        if self.log_transitions is None:
            print ("ERROR: Train or load the model first")
            return

        self.load_test_data(te_data_file, verbose=verbose,
                            corpus_pkl=corpus_pkl, tr_label_pkl=tr_label_pkl, tr_vocab_pkl=tr_vocab_pkl)

        self.model_info = "_".join(["HMM", self.base_model.model_info])

        te_data_nested = self.te_data.get_utter_level_data_from_sids(sorted(self.te_data.sstt2uid.keys()))
        ulists, docs, labs = te_data_nested

        vit_res = []
        for sidx in range(len(ulists)):
            output_prob_s = self.base_model.result.output_prob[sidx]
            log_emissions = convert_class_prob_to_log_emission_prob(output_prob_s, self.marginals)

            vit_res.append(viterbi(log_emissions, self.log_transitions,
                                   self.log_start_prob, self.log_end_prob))

        yhats = [s[1] for s in vit_res]
        output_scores = [s[0] for s in vit_res]
        self.result = DialogResult(self.n_labels, yhats, None, self.marginals,
                                   self.model_info, output_scores)
        if self.te_data.has_label:
            if verbose > 0:
                print("Calculate score")
            self.result.get_scores(labs)
            if verbose > 0:
                print("Printing utterance-level results to file "+output_filename)
            self.result.print_utter_level_results(ulists, docs, labs, self.te_data.lid2name,
                                                  filename=output_filename)
        else:
            if verbose > 0:
                print("Printing utterance-level results to file "+output_filename)
            self.result.print_utter_level_results_without_true_lab(ulists, docs, self.te_data.lid2name,
                                                                   filename=output_filename)
        return self.result


class DialogResult():
    """
    Class that stores all the results and calculates different scores.

    * S : number of sessions
    * N_s: number of utterances in the session. different number for each session.
    * T : number of labels

    Variables
    ---------
    n_labels : int
        Number of labels
    predictions : list[list[int]]
        Nested list of size (S, N_s, 1). List of sessions, where session is a list of
        predictions of utterances in that session
    predictions_f : list[int]
        Flattened version of predictions. Length = S x N_s
    output_prob : list[np.array]
        List of output probabilities for each session.
        Length S list of np.array with size (N_s, T)
        Total size = (S, N_s, T)
    output_score : list[list[float]]
        Nested list of output scores of size (S, N_s, 1).
        Only used for HMM.
    marginals : np.array
        np.array, size (T,)
        Marginal probabilities calculated from the training data.
    model_info : str
        String that describes the model
    scores : dict[str, float]
        Dictionary that stores accuracy, and all the averaged and weighted averaged binary scores.
        '_w' means weighted. Weights are the marginal probabilities from training data.
        Keys are: 'accuracy', 'auc', 'auc_w', 'f1score', 'f1score_w', 'precision', 'precision_w',
                   'recall', 'recall_w', 'rprecision', 'rprecision_w'
    bin_scores : dict[str, list[float]]
        Dictionary that has different binary score lists.
        Keys are 'precision', 'recall', 'auc', 'rprecision', 'f1score'
        Value is a list of scores for each label. length=T.
    """

    def __init__(self, n_labels, predictions, output_prob=None, marginals=None,
                 model_info="", output_score=None):
        self.n_labels = n_labels
        self.predictions = predictions
        self.predictions_f = flatten_nested_labels(predictions)
        self.output_prob = output_prob
        if output_prob is not None:
            self.output_prob_f = flatten_nested_labels(output_prob)
        self.output_score = output_score
        self.marginals = marginals
        self.model_info = model_info
        self.scores = None
        self.bin_scores = None

    def get_scores(self, true_y):
        """
        Calculates scores given true labels.

        Parameters
        ----------
        true_y : list[list[int]]
            Nested list of size (S, Ns, 1)
            (A list list of labels in each session.)

        Returns
        -------

        """
        print("Calculating scores..")
        scores = {}

        true_y_f = flatten_nested_labels(true_y)
        acc = get_accuracy(true_y_f, self.predictions_f)
        scores["accuracy"] = acc

        # Only take averages using non-zero instance labels.
        nonzero_labs = np.where(np.sum(get_lab_arr(true_y_f, self.n_labels), axis=0))[0]
        bin_scores = get_binary_classification_scores(true_y_f, self.predictions_f, self.n_labels)
        for sc in sorted(bin_scores.keys()):
            if self.marginals is None:
                weighted = 0.0
            else:
                weighted = get_weighted_avg(bin_scores[sc][nonzero_labs], self.marginals[nonzero_labs])
            notweighted = np.mean(bin_scores[sc][nonzero_labs])
            scores[sc + "_w"] = weighted
            scores[sc] = notweighted

        self.scores = scores
        self.bin_scores = bin_scores
        return self.scores

    def print_scores(self, true_y=None, filename='./result_in_diff_metrics.csv'):
        """
        Prints and writes scores into a csv file.
        If self.scores does not exist and 'true_y' is given, it first runs get_scores().

        Parameters
        ----------
        true_y : list[list[int]]
            Nested list of size (S, N_s, 1)
            (A list list of labels in each session.)
            If .scores have already been calculated, it can be None (Default).
        filename : str
            File path to the output.
        """
        if self.scores is None:
            if true_y is None:
                print("ERROR: First run get_scores() or input true labels (true_y)")
                return
            else:
                self.get_scores(true_y)

        bin_metrics = ["precision", "recall", "auc", "rprecision", "f1score"]
        orders = ["model", "accuracy"] + [met + "_w" for met in bin_metrics] + bin_metrics
        print(",".join(orders))
        with open(filename, 'w') as f:
            f.write(",".join(orders) + "\n")
        self.print_row_of_diff_metrics(self.model_info, self.scores, orders, filename)

    def print_row_of_diff_metrics(self, modelname, result_numbers, headers, fname):
        """
        Prints a row of scores.
        Used inside of print_scores()
        """
        with open(fname, 'a') as f:
            f.write(modelname)
            print(modelname),
            for met in headers[1:]:
                f.write(",%.4f" % result_numbers[met])
                print(",%.4f" % result_numbers[met]),
            f.write("\n")

    def print_utter_level_results(self, uids, docs, true_y, label_names, print_output=False,
                                  filename='./per_utter_result.txt'):
        """
        Prints the utterance level prediction results into a file.
        Result file will have columns 'SessionID', 'UtteranceID', 'Utterance',
        'True label name', 'Predicted Label name by the model'
        and each row is the utterance.

        Parameters
        ----------

        uids : list[list[int]]
            Nested list of utterance IDs with size (S, N_s, 1) for test data
        docs : list[list[str]]
            Nested list of utterances (text) with size (S, N_s, 1) for test data
        true_y : list[list[int]]
            Nested list of labels with size (S, N_s, 1) for test data
        label_names : list or dict[int, str]
            Mapping from a label index to the label name
        filename : str
            Path to the file name
        """
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as f:
            f.write("SessionIndex|UtteranceID|Utterance|TrueLab|"+self.model_info+"\n")
            if print_output:
                print("SessionIndex|UtteranceID|Utterance|TrueLab|"+self.model_info)

            for sidx in range(len(true_y)):
                truel = true_y[sidx]
                predl = self.predictions[sidx]

                for i in range(len(truel)):
                    f.write("%d|%d|%s|%s|%s\n" % (sidx, uids[sidx][i], docs[sidx][i],
                                                label_names[truel[i]], label_names[predl[i]]))
                    if print_output:
                        print("%d|%d|%s|%s|%s" % (sidx, uids[sidx][i], docs[sidx][i],
                                                      label_names[truel[i]], label_names[predl[i]]))

    def print_utter_level_results_without_true_lab(self, uids, docs, label_names, print_output=False,
                                                   filename='./per_utter_result.txt'):
        """
        Prints the utterance level prediction results into a file.
        Result file will have columns 'SessionID', 'UtteranceID', 'Utterance',
        'True label name', 'Predicted Label name by the model'
        and each row is the utterance.

        Parameters
        ----------

        uids : list[list[int]]
            Nested list of utterance IDs with size (S, N_s, 1) for test data
        docs : list[list[str]]
            Nested list of utterances (text) with size (S, N_s, 1) for test data
        label_names : list or dict[int, str]
            Mapping from a label index to the label name
        filename : str
            Path to the file name
        """
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, 'w') as f:
            f.write("SessionIndex|UtteranceID|Utterance|"+self.model_info+"\n")
            if print_output:
                print("SessionIndex|UtteranceID|Utterance|"+self.model_info)

            for sidx in range(len(uids)):
                predl = self.predictions[sidx]

                for i in range(len(uids[sidx])):
                    f.write("%d|%d|%s|%s|\n" % (sidx, uids[sidx][i], docs[sidx][i],
                                                  label_names[predl[i]]))
                    if print_output:
                        print("%d|%d|%s|%s|" % (sidx, uids[sidx][i], docs[sidx][i],
                                                  label_names[predl[i]]))