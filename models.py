__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import cPickle as cp
import numpy as np

from sklearn.linear_model import LogisticRegression

from mhddata import MHDTrainData, MHDTestData
from evaluate import get_binary_classification_scores, get_accuracy, get_weighted_avg
from utils import get_marginals_from_y_seq, flatten_nested_labels


class DialogModel():

    def __init__(self):
        self.tr_data = None
        self.te_data = None
        self.vectorizer = None
        self.model = None
        self.result = None

    def load_model(self, model_file):
        pass

    def fit_model(self, data_file, model_file):
        pass

    def load_train_data(self, data_file, nouns_only=False, ignore_case=True,
                        remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                        label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                        min_dfreq=0, max_dfreq=0.8, min_sfreq=20,
                        token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                        corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):

        self.tr_data = MHDTrainData(data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                 remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                 label_mappings=label_mappings, ngram_range=ngram_range,
                 max_np_len=max_np_len, min_wlen=min_wlen,
                 min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                 token_pattern=token_pattern,
                 corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        self.marginals = get_marginals_from_y_seq(self.tr_data.uid2lid, self.tr_data.n_labels)

    def load_test_data(self, te_data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                 min_dfreq=0, max_dfreq=0.8, min_sfreq=20,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                 corpus_pkl='./corpus_te.pkl', tr_label_pkl='./label.pkl', tr_vocab_pkl='./vocab.pkl'):

        self.te_data = MHDTestData(te_data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                     remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                     label_mappings=label_mappings, ngram_range=ngram_range,
                     max_np_len=max_np_len, min_wlen=min_wlen,
                     min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                     token_pattern=token_pattern,
                     corpus_pkl=corpus_pkl, tr_label_pkl=tr_label_pkl, tr_vocab_pkl=tr_vocab_pkl)

        self.n_labels = self.te_data.n_labels

    def predict(self, te_data_file):
        pass


class LogRegDialogModel(DialogModel):

    def __init__(self, lr_type="ovr"):
        DialogModel.__init__(self)
        self.lr_type = lr_type

    def fit_model(self, data_file, penalty_type="l2", reg_const=0.9,
                    solver='lbfgs', model_file='./lrdialog.pkl'):
        self.load_train_data(data_file)
        trainX, self.vectorizer = self.tr_data.fit_bow(self.tr_data.corpus_txt, tfidf=True)
        trainy = self.tr_data.uid2lid

        self.model = LogisticRegression(penalty=penalty_type, C=reg_const,
                                        multi_class=self.lr_type, solver=solver)
        self.model = self.model.fit(trainX, trainy)
        self.trainX = trainX
        self.trainy = trainy
        with open(model_file, 'wb') as f:
            print("Saving Logistic regression model to "+ model_file)
            cp.dump((self.model, self.vectorizer, self.marginals), f, protocol=cp.HIGHEST_PROTOCOL)
        return self.model

    def load_model(self, model_file='./lrdialog.pkl'):
        with open(model_file, 'rb') as f:
            print("Loading Logistic regression model to "+ model_file)
            self.model, self.vectorizer, self.marginals = cp.load(f)

    def predict(self, te_data_file):
        if not (self.model and self.vectorizer):
            print ("ERROR: Train or load the model first")
            return
        self.load_test_data(te_data_file)

        n_labs = self.te_data.n_labels
        self.model_info = "_".join(["LogReg", self.lr_type, self.model.penalty, str(self.model.C)])

        te_data_nested = self.te_data.get_utter_level_data_from_sids(sorted(self.te_data.sstt2uid.keys()))
        ulists, docs, labs = te_data_nested
        labs_f = flatten_nested_labels(labs)

        outputprobs = []
        yhats = []
        for sidx in range(len(docs)):
            testX_s = MHDTestData.transform_bow(docs[sidx], self.vectorizer)
            # if self.te_data.has_label:
            #     testy_s = labs[sidx]

            outputprob = self.model.predict_proba(testX_s)
            outputprobs.append(outputprob)
            yhat = self.model.predict(testX_s)
            yhats.append(yhat)

        self.result = DialogResult(n_labs, yhats, outputprobs, self.marginals, self.model_info)
        if self.te_data.has_label:
            self.result.get_scores(labs_f)
        return self.result


from hmm import convert_class_prob_to_log_emission_prob, viterbi, viterbi_with_multiple_transitions,\
    get_transition_mat, get_spkr_transition_mat, get_start_end_prob


class HMMDialogModel(DialogModel):

    def __init__(self, base_model):
        """
        Parameters
        ----------
        dialogmodel : DialogModel
        """
        DialogModel.__init__(self)
        if base_model.result is None:
            print("ERROR: Base model for HMMDialogModel should have .result in it.")
            print("       .predict() method should be applied before running HMM")
            return
        self.base_model = base_model
        self.log_transitions, self.start_prob, self.end_prob, self.marginals = None, None, None, None

    def fit_model(self, data_file, model_file='./hmmdialog.pkl'):
        self.load_train_data(data_file)
        self.n_labels = self.tr_data.n_labels
        tr_data_nested = self.tr_data.get_utter_level_data_from_sids(self.tr_data.sstt2uid.keys())
        ulists_tr, docs_tr, labs_tr = tr_data_nested

        self.log_transitions = get_transition_mat(labs_tr, self.n_labels,
                                                  proportional_prior=True, prior_sum=0.5, log=True)

        # log_spkr_transitions = get_spkr_transition_mat(labs_tr, spkr_tr,
        #                                                self.n_labels, get_patient_mat=False,
        #                                                proportional_prior=True, prior_sum=0.5, log=True)
        # self.log_transitions_md = log_spkr_transitions["MD_trans_mat"]
        # self.log_transitions_oth = log_spkr_transitions["OTH_trans_mat"]

        self.start_prob, self.end_prob = get_start_end_prob(labs_tr, self.n_labels)
        self.log_start_prob = np.log(self.start_prob)
        self.log_end_prob = np.log(self.end_prob)
        self.marginals = get_marginals_from_y_seq(self.tr_data.uid2lid, self.n_labels)

        with open(model_file, 'wb') as f:
            print("Saving model to "+ model_file)
            data_to_save = (self.log_transitions, self.start_prob, self.end_prob, self.marginals)
            cp.dump(data_to_save, f, protocol=cp.HIGHEST_PROTOCOL)


    def load_model(self, model_file='./hmmdialog.pkl'):
        print("Loading model from "+ model_file)
        with open(model_file, 'rb') as f:
            self.log_transitions, self.start_prob, self.end_prob, self.marginals = cp.load(f)

    def predict_viterbi(self, te_data_file):# spkr_transitions=False):
        """
        Viterbi decoding
        """
        if self.log_transitions is None:
            print ("ERROR: Train or load the model first")
            return
        self.load_test_data(te_data_file)
        self.model_info = "_".join(["HMM", self.base_model.model_info])

        te_data_nested = self.te_data.get_utter_level_data_from_sids(sorted(self.te_data.sstt2uid.keys()))
        ulists, docs, labs = te_data_nested
        labs_f = flatten_nested_labels(labs)

        vit_res = []
        for sidx in range(len(ulists)):
            output_prob_s = self.base_model.result.output_prob[sidx]
            log_emissions = convert_class_prob_to_log_emission_prob(output_prob_s, self.marginals)

            vit_res.append(viterbi(log_emissions, self.log_transitions,
                                   self.log_start_prob, self.log_end_prob))

            # if spkr_transitions:
            #     vit_res.append(viterbi_with_multiple_transitions(log_emissions, np.array(self.splits.spkr_te[sidx]),
            #                                                      [self.log_transitions_md, self.log_transitions_oth],
            #                                                      self.log_start_prob, self.log_end_prob))
            # else:
            #     vit_res.append(viterbi(log_emissions, self.log_transitions,
            #                            self.log_start_prob, self.log_end_prob))

        yhats = [s[1] for s in vit_res]
        output_scores = [s[0] for s in vit_res]
        self.result = DialogResult(self.n_labels, yhats, None, self.marginals,
                                   self.model_info, output_scores)
        if self.te_data.has_label:
            self.result.get_scores(labs_f)
        return self.result



class DialogResult():

    def __init__(self, n_labels, predictions, output_prob=None, marginals=None, model_info="", output_score=None):
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

    def get_scores(self, true_y):
        print("Calculating scores..")
        scores = {}
        # self.marginals = get_marginals_from_y_seq(train_true_y, self.n_labels)

        acc = get_accuracy(true_y, self.predictions_f)
        scores["accuracy"] = acc

        bin_scores = get_binary_classification_scores(true_y, self.predictions_f, self.n_labels)
        for sc in sorted(bin_scores.keys()):
            weighted = get_weighted_avg(bin_scores[sc], self.marginals)
            notweighted = np.mean(bin_scores[sc])
            scores[sc + "_w"] = weighted
            scores[sc] = notweighted

        self.scores = scores
        return self.scores

    def print_scores(self, true_y=None, marginals=None, filename='./result_in_diff_metrics.csv'):
        if self.scores is None:
            if true_y is None or marginals is None:
                print("ERROR: First run get_scores() or input true_labels")
                return
            else:
                self.get_scores(true_y, marginals)

        bin_metrics = ["precision", "recall", "auc", "rprecision", "f1score"]
        orders = ["model", "accuracy"] + [met + "_w" for met in bin_metrics] + bin_metrics
        print(",".join(orders))
        with open(filename, 'w') as f:
            f.write(",".join(orders) + "\n")
        self.print_row_of_diff_metrics(self.model_info, self.scores, orders, filename)

    def print_row_of_diff_metrics(self, modelname, result_numbers, headers, fname):
        with open(fname, 'a') as f:
            f.write(modelname)
            print(modelname),
            for met in headers[1:]:
                f.write(",%.4f" % result_numbers[met])
                print(",%.4f" % result_numbers[met]),
            f.write("\n")






