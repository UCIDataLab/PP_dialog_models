import cPickle as cp
import numpy as np

from sklearn.linear_model import LogisticRegression

from mhddata import MHDTrainData, MHDTestData

class DialogModel():

    def __init__(self):
        self.tr_data = None
        self.te_data = None
        self.vectorizer = None
        self.model = None

    def load_model(self, model_file):
        pass

    def train_model(self, model_file):
        pass

    def load_train_data(self, tr_data_file):
        pass

    def load_test_data(self, te_data_file):
        pass

    def predict(self, te_data_file):
        pass


class LogRegDialogModel(DialogModel):

    def __init__(self, lr_type="ovr"):
        DialogModel.__init__(self)
        self.lr_type = lr_type

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

    def train_model(self, data_file, penalty_type="l2", reg_const=0.9,
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
            cp.dump((self.model, self.vectorizer), f, protocol=cp.HIGHEST_PROTOCOL)
        return self.model

    def load_model(self, model_file='./lrdialog.pkl'):
        with open(model_file, 'rb') as f:
            print("Loading Logistic regression model to "+ model_file)
            self.model, self.vectorizer = cp.load(f)

    def predict(self, te_data_file):
        if not (self.model and self.vectorizer):
            print ("ERROR: Train or load the model first")
            return
        self.load_test_data(te_data_file)
        testX = MHDTestData.transform_bow(self.te_data.corpus_txt, self.vectorizer)
        self.testX = testX
        if self.te_data.has_label:
            testy = self.te_data.uid2lab
            self.testy = testy

        outputprob = self.model.predict_proba(testX)
        yhat = self.model.predict(testX)
        n_labs = self.te_data.n_labels
        model_info = "_".join(["LogReg", self.lr_type, self.model.penalty, str(self.model.C)])

        self.result = DialogResult(n_labs, yhat, outputprob, model_info)
        return self.result


class HMMDialogModel(DialogModel):

    def __init__(self):
        DialogModel.__init__(self)



from evaluate import get_binary_classification_scores, get_accuracy, get_weighted_avg
from utils import get_marginals_from_y_seq

class DialogResult():

    def __init__(self, n_labels, predictions, output_prob=None, model_info=""):
        self.n_labels = n_labels
        self.predictions = predictions
        self.output_prob = output_prob
        self.model_info = model_info
        self.scores = None

    def get_scores(self, true_y, train_true_y):
        print("Calculating scores..")
        scores = {}
        self.marginals = get_marginals_from_y_seq(train_true_y, self.n_labels)

        acc = get_accuracy(true_y, self.predictions)
        scores["accuracy"] = acc

        bin_scores = get_binary_classification_scores(true_y, self.predictions, self.n_labels)
        for sc in sorted(bin_scores.keys()):
            weighted = get_weighted_avg(bin_scores[sc], self.marginals)
            notweighted = np.mean(bin_scores[sc])
            scores[sc + "_w"] = weighted
            scores[sc] = notweighted

        self.scores = scores
        return self.scores

    def print_scores(self, true_y=None, train_true_y=None, filename='./result_in_diff_metrics.csv'):
        if self.scores is None:
            if true_y is None or train_true_y is None:
                print("ERROR: First run get_scores() or input true_labels")
                return
            else:
                self.get_scores(true_y, train_true_y)

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






