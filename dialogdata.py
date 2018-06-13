__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import cPickle as cp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import preprocess as preproc


class DialogData():

    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,2), max_np_len=3, min_wlen=1,
                 min_dfreq=0.0001, max_dfreq=0.9, min_sfreq=5,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                 corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):

        self.n_labels = 0

        self.data_file = data_file
        self.nouns_only = nouns_only

        self.ngram_range = ngram_range
        self.min_dfreq = min_dfreq
        self.max_dfreq = max_dfreq

        if self.nouns_only:
            self.max_np_len = 1
        else:
            self.max_np_len = min(2, max_np_len)

        self.max_wlen = max(self.ngram_range[-1], self.max_np_len)

        self.token_pattern = token_pattern

        self.stopwords = preproc.get_stopwords(stopwords_dir=stopwords_dir)
        self.stopwords_all = self.stopwords.union() # do union for copying

        self.load_corpus(data_file, sep="|", min_wcnt=min_wlen,
                         min_np_len=self.max_np_len, max_np_len=max_np_len,
                         token_pattern=token_pattern,
                         ignore_case=ignore_case, remove_numbers=remove_numbers,
                         sub_numbers=sub_numbers,
                         corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        self.assign_codenames()
        self.assign_spkr_codes()

    def assign_codenames(self):
        pass

    def assign_spkr_codes(self):
        pass

    def load_corpus(self, corpus_file, sep, min_wcnt=1, min_np_len=2, max_np_len=3,
                    token_pattern=r"(?u)\b\w[A-Za-z']*\b",
                    ignore_case=True, remove_numbers=False, sub_numbers=True, parser=None, stemmer_type=None,
                    corpus_pkl='./corpus.pkl', label_pkl='./labels.pkl', vocab_pkl='./vocab.pkl'):
        pass

    def _save_lab_pkl(self, lid2lab_all, segid2lt, label_pkl):
        lid2lab_all = sorted(list(lid2lab_all))
        lab2lid_all = {ll: i for i, ll in enumerate(lid2lab_all)}

        self.ltid2lt = sorted(list(set(segid2lt.values())))
        self.lt2ltid = {ll: i for i, ll in enumerate(self.ltid2lt)}

        self.lab2lid_all = lab2lid_all
        self.lid2lab_all = lid2lab_all

        lab_data_to_save = (lid2lab_all, lab2lid_all, self.ltid2lt, self.lt2ltid)
        with open(label_pkl, 'wb') as f:
            cp.dump(lab_data_to_save, f, protocol=cp.HIGHEST_PROTOCOL)

    def _save_vocab_pkl(self, nps, vocab, stw_all, vocab_pkl):

        if self.nouns_only:
            nps_dict = {}
            for i, nounp in enumerate(nps):
                nps_dict[nounp] = i
            self.vocabulary = nps_dict
        else:
            self.vocabulary = vocab
        self.stopwords_all = stw_all

        vocab_data_to_save = (self.vocabulary, stw_all)
        with open(vocab_pkl, 'wb') as f:
            cp.dump(vocab_data_to_save, f, protocol=cp.HIGHEST_PROTOCOL)

    def _save_corpus_pkl(self, corpus_df, nps, np2cnt, uid2sstt, sstt2uid,
                         uid2segid, segid2uids, sid2labs_all, segid2lab_all,
                         segid2lt, uid2lab_all, corpus_pkl):
        self.nps = nps
        self.np2cnt = np2cnt
        self.uid2sstt = uid2sstt
        self.sstt2uid = sstt2uid
        self.uid2segid = uid2segid
        self.segid2uids = segid2uids
        self.corpus_txt = list(corpus_df.text_cleaned)

        self.segid2lab_all = segid2lab_all
        self.segid2lt = segid2lt
        self.sid2labs_all = sid2labs_all
        self.uid2lab_all = uid2lab_all

        data_to_save = (corpus_df, nps, np2cnt, uid2sstt, sstt2uid, uid2segid, segid2uids,
                        sid2labs_all, segid2lab_all, segid2lt, uid2lab_all)
        if not os.path.isdir(os.path.dirname(corpus_pkl)):
            os.makedirs(os.path.dirname(corpus_pkl))
        with open(corpus_pkl, 'wb') as f:
            cp.dump(data_to_save, f, protocol=cp.HIGHEST_PROTOCOL)

    def _load_vocab_pkl(self, vocab_pkl):
        # if the pickle file exists, load the vocab
        if os.path.exists(vocab_pkl):
            print("Loading the vocabulary file from " + vocab_pkl)
            print(" (Delete the file if you want to re-generate the vocabulary)")
            vocab_obj = cp.load(open(vocab_pkl))
            if len(vocab_obj) > 2:
                self.vocabulary = vocab_obj
            elif len(vocab_obj) == 2:
                self.vocabulary, self.stopwords_all = vocab_obj
            else:
                return False
            return True
        else:
            return False

    def _load_corpus_pkl(self, corpus_pkl):
        # if the pickle file exists, load the corpus
        if os.path.exists(corpus_pkl):
            print("Loading the processed file from " + corpus_pkl)
            print(" (Delete the file if you want to re-process the corpus)")
            corpus_data_list = cp.load(open(corpus_pkl, 'rb'))
            if len(corpus_data_list) == 11:
                self.corpus_df, self.nps, self.np2cnt, self.uid2sstt, \
                self.sstt2uid, self.uid2segid, self.segid2uids, \
                self.sid2labs_all, self.segid2lab_all, self.segid2lt, self.uid2lab_all = corpus_data_list

                self.corpus_txt = list(self.corpus_df.text_cleaned)
            else:
                print("ERROR: Cannot load the corpus file." + corpus_pkl + " has different format!")
                return False
            return True
        else:
            return False


    def _load_lab_pkl(self, label_pkl):
        if os.path.exists(label_pkl):
            print("Loading labels file from " + label_pkl)
            print(" (Delete the file if you want to re-generate the labels)")

            lab_data_list = cp.load(open(label_pkl, 'rb'))
            if len(lab_data_list) == 4:
                self.lid2lab_all, self.lab2lid_all, \
                self.ltid2lt, self.lt2ltid  = lab_data_list
            else:
                print("ERROR: Cannot load the label file." + label_pkl + " has different format!")
                return False
            return True
        else:
            return False

    def load_labels(self, min_sess_freq=20, label_mappings=None):
        pass

    def _update_session_labs(self, sid2labs, lab2lid_new, label_mappings, code_other):
        sid2labs_new = {}
        sid2lidarr_new = {}
        for sessid in sid2labs:
            labarr = np.zeros(self.n_labels, dtype=np.int8)
            sid2labs_new[sessid] = []
            for lab in sid2labs[sessid]:
                if label_mappings.get(lab, None) is not None:
                    lab = label_mappings[lab]
                lid = lab2lid_new.get(lab, lab2lid_new[code_other])
                #  To avoid appending the same txtlabels
                if labarr[lid] == 0:
                    labarr[lid] = 1
                    sid2labs_new[sessid].append(lab)
            # only save the sessions with labels
            if np.sum(labarr) > 0:
                sid2lidarr_new[sessid] = labarr
        return sid2labs_new, sid2lidarr_new


    def _update_segment_labs(self, segid2lab, lab2lid_new, segid2lt,
                            label_mappings, code_other):
        segid2lab_new = {}
        segid2lid_new = {}
        segid2lidarr = {}
        segid2ltid = {}
        for segid in segid2lab:
            labarr = np.zeros(self.n_labels)
            lab = segid2lab[segid]

            # also update tletter
            tletter = segid2lt[segid]
            tltid = self.lt2ltid[tletter]

            if label_mappings.get(lab, None) is not None:
                lab = label_mappings[lab]
            lid = lab2lid_new.get(lab, lab2lid_new[code_other])

            segid2ltid[segid] = tltid
            segid2lab_new[segid] = lab
            labarr[lid] = 1
            segid2lidarr[segid] = labarr
            segid2lid_new[segid] = lid
        return segid2lab_new, segid2lid_new, segid2lidarr, segid2ltid


    def _update_utter_labs(self, uid2lab, lab2lid_new, label_mappings, code_other):
        uid2lab_new = []
        uid2lid_new = []
        for uid, lab in enumerate(uid2lab):
            if label_mappings.get(lab, None) is not None:
                lab = label_mappings[lab]
            lid = lab2lid_new.get(lab, lab2lid_new[code_other])
            uid2lab_new.append(lab)
            uid2lid_new.append(lid)
        return uid2lab_new, uid2lid_new


    def get_valid_data(self):
        pass


    def fit_bow(self, train_doc, tfidf=True, vocabulary=None, stop_words=None,
                token_pattern=r"(?u)[A-Za-z\?\!\-\.']+", max_wlen=0):

        if max_wlen == 0:
            max_wlen = self.max_wlen

        if vocabulary is None:
            vocabulary = self.vocabulary

        if stop_words is None:
            stop_words = self.stopwords_all

        if token_pattern != self.token_pattern:
            token_pattern = self.token_pattern

        if tfidf is True:
            vectorizer = TfidfVectorizer(ngram_range=(1, max_wlen),
                                         token_pattern=token_pattern,
                                         stop_words=stop_words,
                                         vocabulary=vocabulary)
        else:
            vectorizer = CountVectorizer(ngram_range=(1, max_wlen),
                                         token_pattern=token_pattern,
                                         stop_words=stop_words,
                                         vocabulary=vocabulary)

        train_bow = vectorizer.fit_transform(train_doc)
        self.bow = train_bow
        self.vectorizer = vectorizer
        return train_bow, vectorizer

    @staticmethod
    def transform_bow(test_doc, vectorizer):
        if test_doc is not None and len(test_doc) > 0:
            return vectorizer.transform(test_doc)
        else:
            return np.array([])

    @staticmethod
    def get_nested_bow(nested_docs, vectorizer):
        """
        Works when nested_docs is a list[list[string]].
        """
        nested_bows = []
        for doc in nested_docs:
            nested_bows.append(DialogData.transform_bow(doc, vectorizer))
        return nested_bows
