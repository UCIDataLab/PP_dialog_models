__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import cPickle as cp
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import preprocess as preproc


class MHDData():

    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,2), max_np_len=3, min_wlen=1,
                 min_dfreq=0.0001, max_dfreq=0.9, min_sfreq=5,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                 corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):

        self.n_labels = 0
        self.uid2lid = []

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



    def get_spkr_code(self, spkr, include_ra=False, include_nurse=False):
        spkr = re.sub(r"[^A-Za-z]", "", spkr)
        if re.match(r".*P[Tt].*", spkr) or re.match(r"[Pp]atient", spkr):
            return self.spkr2spkrid["PT"]
        elif re.match(r"clinician", spkr) or re.match(r".*MD.*", spkr):
            return self.spkr2spkrid["MD"]
        else:
            if include_ra and re.match(r".*RA.*", spkr):
                return self.spkr2spkrid["RA"]
            if include_nurse and (re.match(r".*NURSE.*", spkr) or re.match(r".*RN.*", spkr)):
                return self.spkr2spkrid["RN"]
            return self.spkr2spkrid["OTHER"]

    def get_spkr_list(self, uids, get_spkr_category, nested=True):
        if nested: # if uids are nested
            spkrs_list = []
            for ses_uids in uids:
                spkrs = map(get_spkr_category, map(str, self.corpus_df.iloc[ses_uids].speaker))
                spkrs_list.append(spkrs)
        else:
            spkrs_list = map(get_spkr_category, map(str, self.corpus_df.iloc[uids].speaker))
        return spkrs_list


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
            nested_bows.append(MHDData.transform_bow(doc, vectorizer))
        return nested_bows


    def get_utter_level_data_from_sids(self, sesid_list):
        """
        Get nested data using session IDs

        Parameters
        ----------
        sesid_list

        Returns
        -------

        """
        ulists = []
        docs = []
        labs = []
        for sesid in sesid_list:
            tmp_uids = self.convert_docids_level([sesid], from_level='session', to_level='utterance',
                                                 insert_st_end=False)
            doc_tmp, lab_tmp, _ = self.get_utter_level_subset(tmp_uids, chunk_size=1, overlap=False)
            ulists.append(tmp_uids)
            docs.append(doc_tmp)
            labs.append(lab_tmp)
        return ulists, docs, labs


    def convert_docids_level(self, ids_list, from_level='session', to_level='utterance',
                             insert_st_end=False):
        """

        Parameters
        ----------
        ids_list
        from_level
        to_level
        insert_st_end : bool
            If True, -1 and -2 will be inserted for start and end of the session or segment
            (depending on the 'from_level').

        Returns
        -------

        """

        def get_ulist(ids_list, orig_list_type='session', insert_st_end=False):
            """ from session/segment to utterance"""
            ulist = []
            if orig_list_type == 'session':
                for sid in ids_list:
                    if insert_st_end:
                        ulist.append(-1)
                    for tt in self.sstt2uid[sid]:
                        ulist.append(self.sstt2uid[sid][tt])
                    if insert_st_end:
                        ulist.append(-2)
            elif orig_list_type == 'segment':
                for segid in ids_list:
                    if insert_st_end:
                        ulist.append(-1)
                    for uid in self.segid2uids[segid]:
                        ulist.append(uid)
                    if insert_st_end:
                        ulist.append(-2)
            else:
                print "ERROR: [get_ulist] orig_list_type can be either 'session' or 'segment'!"
                exit()
            return ulist

        def get_seglist(sids_list, insert_st_end=False):
            """ from session to segment """
            seglist = []
            for sid in sids_list:
                if insert_st_end:
                    seglist.append(-1)
                for segid in self.sid2segids[sid]:
                    seglist.append(segid)
                if insert_st_end:
                    seglist.append(-2)
            return seglist

        if to_level == 'utterance':
            return get_ulist(ids_list, orig_list_type=from_level, insert_st_end=insert_st_end)
        elif to_level == 'segment':
            return get_seglist(ids_list, insert_st_end=insert_st_end)
        else:
            print "ERROR: [convert_docids_level] to_level can be either 'utterance' or 'segment'!"
            exit()

    def get_utter_level_subset(self, uid_list, chunk_size=1, overlap=False):
        """

        Parameters
        ----------
        uid_list
        chunk_size : int
            size of the window/chunk. Unit : sentences
        overlap : bool
            Currently not supported

        Returns
        -------

        """

        text_list = []
        lab_list = []
        tmp_txt = []
        new_uid_list = []
        tmp_uids = []
        labid = -1
        prev_sid = 0
        i = 0

        for uid in uid_list:
            segid = self.uid2segid[uid]
            # Whenever you see a new segment or a new chunk,
            # only when there is some text to save
            if (len(tmp_txt) > 0) and (i % chunk_size == 0 or segid != prev_sid):
                i = 0
                text_list.append('|'.join(tmp_txt))
                lab_list.append(labid)
                if chunk_size > 1:
                    new_uid_list.append(tmp_uids)
                    tmp_uids = []
                tmp_txt = []

            if chunk_size > 1:
                tmp_uids.append(uid)
            tmp_txt.append(self.corpus_txt[uid])
            if len(self.uid2lid) > uid:
                labid = self.uid2lid[uid] #segid2lid[segid] # both could work
            else:
                labid = -1
            prev_sid = segid
            i += 1

        # Take care of the last one
        if len(tmp_txt) > 0:
            lab_list.append(labid)
            text_list.append(' '.join(tmp_txt))

        if chunk_size == 1:
            new_uid_list = uid_list
        elif chunk_size > 1 and len(tmp_txt) > 0:
            new_uid_list.append(tmp_uids)

        return text_list, lab_list, new_uid_list
