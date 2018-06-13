__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import cPickle as cp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import preprocess as preproc
from dialogdata import DialogData


class MHDTrainData(DialogData):

    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                 min_dfreq=0.0, max_dfreq=0.9, min_sfreq=20,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                 corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):

        DialogData.__init__(self, data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                 remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                 label_mappings=label_mappings, ngram_range=ngram_range,
                 max_np_len=max_np_len, min_wlen=min_wlen,
                 min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                 token_pattern=token_pattern,
                 corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        cl_lab_pkl = label_pkl.split(".pkl")[0] + "_cleaned.pkl"
        self.load_labels(min_sess_freq=min_sfreq, label_mappings=label_mappings)
        self._save_cleaned_lab_pkl(self.lid2lab, self.lab2lid, self.label_mappings, cl_lab_pkl)
        self.get_valid_data()

        self.n_utters = len(self.uid2sstt)
        self.n_vocab = len(self.vocabulary)
        self.n_labels = len(self.lid2lab)

        self.bow = None
        self.vectorizer = None


    def print_stats(self):
        print("Number of sessions: %d (ones that have text)" % len(self.sstt2uid))
        print("Number of sessions: %d (ones that have labels)" % len(self.sid2labs_all))
        print("Number of sessions: %d (ones that have both text and labels)" % len(self.valid_sids))
        print("Number of segments: %d (ones that have both text and labels)" % len(self.valid_segids))
        print("Number of utterances: %d (ones that have both text and labels)" % len(self.valid_uids))
        print("Number of labels that originally had: %d (including the ones that appear in the sessions without text)" % len(self.lid2lab_all))
        print("Number of labels: %d (after cleaning the labels)" % self.n_labels)
        print("Vocabulary size: %d" % self.n_vocab)
        print("Number of user-defined stopwords: %d" % len(self.stopwords))
        print("Number of stopwords used in total: %d (including the words with low dfs and high dfs)" % len(self.stopwords_all))


    def get_valid_data(self):
        # Get session IDs and utterance IDs that both have the text and the labels
        print("Getting lists of valid session/utterance IDs that have both text and labels")

        # First get the self.sid2segids
        sid2segids = {}
        segid2sid = {}
        for sid in self.sstt2uid:
            segidset = set()
            for tt in self.sstt2uid[sid]:
                uid = self.sstt2uid[sid][tt]
                segid = self.uid2segid[uid]
                segidset.add(segid)
                if segid2sid.get(segid, None) is None:
                    segid2sid[segid] = sid
            sid2segids[sid] = sorted(list(segidset))

        self.sid2segids = sid2segids
        self.segid2sid = segid2sid

        valid_sids = []
        valid_segids = []
        valid_uids = []

        for sid in sid2segids:
            if len(self.sid2labs.get(sid, [])) > 0:
                valid_sids.append(sid)
                for segid in sid2segids[sid]:  # segids list
                    lid = self.segid2lid.get(segid, self.n_labels)  # When there is no label, assign the number n_label
                    if lid < self.n_labels and self.lid2lab[lid] != 'nan':  # the last label is not always 'nan'. We're currently including 'nan' for now.
                        valid_segids.append(segid)
                        for uid in self.segid2uids[segid]:
                            valid_uids.append(uid)
        self.valid_sids = sorted(valid_sids)
        self.valid_segids = sorted(valid_segids)
        self.valid_uids = sorted(valid_uids)

    def assign_codenames(self):
        shortnames = ['BiomedHistorySymptom', 'MusSkePain', 'DizzyFallMemoryDentHearVision',
                      'GynGenitoUrinary', 'Prognosis', 'TestDiagnostics', 'TherapeuticIntervention',
                      'Medication', 'PreventiveCare', 'Diet',
                      'Weight', 'Exercise', 'Alcohol', 'Sex', 'Cigarette',
                      'OtherAddictions', 'Sleep', 'Death', 'Bereavement', 'PainSuffering',
                      'Depression', 'GeneralAnxieties', 'HealthCareSystem', 'ActivityDailyLiving',
                      'WorkLeisureActivity', 'Unemployment', 'MoneyBenefits', 'Caregiver', 'HomeEnv', 'Family',
                      'Religion', 'Age', 'LivingWillAdvanceCarePlanning', 'MDLife', 'MDPT-Relationship',
                      'SmallTalk', 'Other', 'VisitFlowManagement', 'PhysicalExam']
        self.lab2shortname = {str(i + 1): shortnames[i] for i in range(len(shortnames))}


    def assign_spkr_codes(self):
        self.spkrid2spkr = ["MD", "PT", "OTHER"]
        self.spkr2spkrid = {sp: i for i, sp in enumerate(self.spkrid2spkr)}


    def load_corpus(self, corpus_file, sep, min_wcnt=1, min_np_len=2, max_np_len=3,
                    token_pattern=r"(?u)\b\w[A-Za-z']*\b",
                    ignore_case=True, remove_numbers=False, sub_numbers=True, parser=None, stemmer_type=None,
                    corpus_pkl='./corpus.pkl', label_pkl='./labels.pkl', vocab_pkl='./vocab.pkl'):

        print('Loading and preprocessing the corpus with labels')
        if not (self._load_corpus_pkl(corpus_pkl) and self._load_vocab_pkl(vocab_pkl) and
                    self._load_lab_pkl(label_pkl)):
            uidx = 0
            segidx = -1
            uid2sstt = []  # uid to [session,talkturn]
            sstt2uid = {}  # [session][talkturn] to uid
            uid2segid = []  # uid to segment id
            segid2uids = defaultdict(list)  # segment id to utter id
            corpus_txt = []
            sesid_arr, segid_arr = [], []
            rows_to_keep = []

            prev_lab = None
            prev_sesid = -1

            # Does not count on the segments
            sid2labs_all = defaultdict(list)
            lid2lab_all = set()

            # Does count on the segments
            segid2lab = {}  # segment id to string label
            segid2lt = {}

            # Function that checks the valid label
            is_valid_lab = lambda x: True if len(x) > 0 else False  # Keep nan as one

            # Load the delimited file
            raw_df = pd.read_csv(corpus_file, sep=sep)
            raw_df = raw_df[raw_df.topicnumber > 0]
            raw_df = raw_df[raw_df.visitid > 0]

            print("  Cleaning the corpus (removing punctuations..)")
            for i in range(raw_df.shape[0]):
                row = raw_df.iloc[i]

                sesid = int(row['visitid'])
                tt = int(row['talkturn'])
                text_ = row['text']
                lab = str(int(row['topicnumber']))
                lab_letter = str(row['topicletter']).strip()

                # preprocess the sentence/doc
                if type(text_) == str:
                    text = preproc.remove_punc(text_, ignore_case, remove_numbers=remove_numbers, mhddata=True)

                    if len(text.split()) >= min_wcnt:  # or maybe do len(text) < min_charcnt?

                        sesid_arr.append(sesid)
                        rows_to_keep.append(i)

                        # Finds a segment
                        if (prev_sesid != sesid or prev_lab != lab) and is_valid_lab(lab):
                            segidx += 1
                            segid2lab[segidx] = lab
                            segid2lt[segidx] = lab_letter
                            sid2labs_all[sesid].append(lab)
                            lid2lab_all.add(lab)
                        uid2sstt.append([sesid, tt])
                        uid2segid.append(segidx)
                        segid_arr.append(segidx)
                        if sstt2uid.get(sesid, None) is None:
                            sstt2uid[sesid] = {}
                        sstt2uid[sesid][tt] = uidx
                        segid2uids[segidx].append(uidx)
                        corpus_txt.append(text)
                        uidx += 1
                        prev_lab = lab
                        prev_sesid = sesid

            # Get tokenized & POS tagged corpus with the set of noun phrases
            vocab, stw_all, np2cnt, nps, corpus_pos \
                = preproc.define_vocabulary(corpus_txt,
                                            self.stopwords,
                                            token_pattern,
                                            self.ngram_range,
                                            self.min_dfreq,
                                            self.max_dfreq,
                                            min_np_len, max_np_len,
                                            ignore_case,
                                            parser, stemmer_type)

            cleaned_df = raw_df.iloc[rows_to_keep]
            cleaned_df = cleaned_df.assign(visitid=sesid_arr, text_cleaned=corpus_txt,
                                           text_postag=corpus_pos, segmentid=segid_arr)
            self.corpus_df = cleaned_df.reset_index(drop=True)
            uid2lab = map(str, map(int, list(self.corpus_df.topicnumber)))

            self._save_corpus_pkl(self.corpus_df, nps, np2cnt, uid2sstt, sstt2uid,
                                  uid2segid, segid2uids, sid2labs_all, segid2lab,
                                  segid2lt, uid2lab, corpus_pkl)

            self._save_vocab_pkl(nps, vocab, stw_all, vocab_pkl)

            self._save_lab_pkl(lid2lab_all, segid2lt, label_pkl)

        # Get the reverse vocab mapping
        voc_len = max(self.vocabulary.values()) + 1
        self.vocabulary_inv = [""] * voc_len
        for voc, i in self.vocabulary.iteritems():
            self.vocabulary_inv[i] = voc


    def load_labels(self, min_sess_freq=20, label_mappings=None):
        """
        Using this method to keep the parent class' format.
        This function mainly does label cleaning since the labels are loaded with corpus.
        """
        self.lab2ltr = {'1': 'A', '2': 'A', '3': 'A', '4': 'A', '5': 'A', '6': 'A', '7': 'A',
                        '8': 'A', '9': 'A', '10': 'B', '11': 'B', '12': 'B', '13': 'B', '14': 'B',
                        '15': 'B', '16': 'B', '17': 'B', '18': 'C', '19': 'C', '20': 'C', '21': 'C',
                        '22': 'C', '23': 'D', '24': 'D', '25': 'D', '26': 'D', '27': 'D', '28': 'D',
                        '29': 'D', '30': 'D', '31': 'D', '32': 'D', '33': 'D', '34': 'E', '35': 'E',
                        '36': 'F', '37': 'F', '38': 'G', '39': 'G'}

        self.n_labels = len(self.lid2lab_all)

        # Clean labels (ignore labels that appear less than N times)
        cleaned = self.clean_labels(self.lid2lab_all, self.lab2lid_all,
                                    self.sid2labs_all, self.segid2lab_all, self.segid2lt,
                                    self.uid2lab_all, min_sess_freq, label_mappings)

        self.lid2lab, self.lab2lid = cleaned[0]
        self.sid2labs, self.sid2lidarr = cleaned[1]
        self.segid2lab, self.segid2lid, self.segid2lidarr, self.segid2ltid = cleaned[2]
        self.uid2lab, self.uid2lid = cleaned[3]
        self.label_mappings = cleaned[-1]
        self.lid2shortname = [self.lab2shortname[self.lid2lab[i]] for i in range(len(self.lid2lab))]


    def clean_labels(self, lid2lab, lab2lid, sid2labs, segid2lab, segid2lt, uid2lab,
                     min_sess_freq=10, label_mappings=None, print_mappings=True):
        """
        This function is based on Garren Gaut's work (matlab file 'loadlabels.m')

        Parameters
        ----------
        lid2lab : list
        lab2lid : dict
        sid2labs : dict
        segid2lab : dict
        min_sess_freq : int
            Threshold for the minimum number of sessions a label needs to be associated with.
            If not met, label is deleted. Default = 5

        Returns
        -------
        lid2lab_new
        lab2lid_new
        sid2txtlabels_new
        sid2labels_new

        """

        def print_label_mappings(lab_map, labcnts):
            for lab in sorted(lab_map.keys()):
                print("  %s %s --> %s %s" % (lab, self.lab2shortname.get(lab, "nan"),
                                             lab_map[lab], self.lab2shortname.get(lab_map[lab], "nan")))

        print("Cleaning labels ..")
        # Can include more steps other than deleting less frequent labels

        # 1. Label mappings for merging labels (it does not create new labels)
        if label_mappings is None:
            label_mappings = {}

        # For deleting rare labels (that appears less than 'min_sess_freq' times)
        # Create a matrix with the subject labels (so that we can get the # sessions per each label)
        labmat = np.zeros((len(sid2labs), self.n_labels))  # n_sessions X n_labels
        for i, sid in enumerate(sid2labs):
            if self.sstt2uid.get(sid, None) is None:
                continue
            labarr = np.zeros(self.n_labels, dtype=np.int8)
            for lab in sid2labs[sid]:
                if lab in lid2lab:
                    # merge labels
                    if label_mappings.get(lab, None) is not None:
                        lab = label_mappings[lab]
                    lid = lab2lid[lab]
                    labarr[lid] = 1
            labmat[i] = labarr


        # avoid deleting 'other' code
        code_other = '37'
        labmat[:, self.lab2lid_all[code_other]] += min_sess_freq

        # Delete rare labels
        labids_to_keep = np.where(np.sum(labmat, axis=0) >= min_sess_freq)[0]
        labids_to_merge = np.where(np.sum(labmat, axis=0) < min_sess_freq)[0]
        if len(labids_to_merge) > 0 and len(label_mappings) == 0:
            # If mapping was not defined, map the rare labels to 'others'
            label_mappings = {self.lid2lab_all[lid]: code_other for lid in labids_to_merge}

        if print_mappings:
            print_label_mappings(label_mappings, np.sum(labmat, axis=0))

        lid2lab_new = [lid2lab[lid] for lid in labids_to_keep]
        lab2lid_new = {ll: i for i, ll in enumerate(lid2lab_new)}
        self.n_labels = len(lid2lab_new)

        # update session-level, segment-level, utterance-level label data
        sid_labs = self._update_session_labs(sid2labs, lab2lid_new,
                                             label_mappings, code_other)
        segid_labs = self._update_segment_labs(segid2lab, lab2lid_new, segid2lt,
                                               label_mappings, code_other)
        uid_labs = self._update_utter_labs(uid2lab, lab2lid_new, label_mappings, code_other)

        return (lid2lab_new, lab2lid_new), sid_labs, segid_labs, uid_labs, label_mappings


    def _save_cleaned_lab_pkl(self, lid2lab, lab2lid, lab_mappings,
                              cleaned_label_pkl='./label_cleaned.pkl'):
        lab_data_to_save = (lid2lab, lab2lid, lab_mappings)
        with open(cleaned_label_pkl, 'wb') as f:
            cp.dump(lab_data_to_save, f, protocol=cp.HIGHEST_PROTOCOL)


class MHDTestData(DialogData):

    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                 min_dfreq=0.0, max_dfreq=0.9, min_sfreq=10,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                 corpus_pkl='./corpus_test.pkl', tr_label_pkl="./label.pkl", tr_vocab_pkl="./vocab.pkl"):

        cl_lab_pkl = tr_label_pkl.split(".pkl")[0] + "_cleaned.pkl"
        if self.load_train_lab_pkl(tr_label_pkl, cl_lab_pkl) and self.load_train_vocab_pkl(tr_vocab_pkl):

            self.has_label = False

            DialogData.__init__(self, data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                     remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                     label_mappings=label_mappings, ngram_range=ngram_range,
                     max_np_len=max_np_len, min_wlen=min_wlen,
                     min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                     token_pattern=token_pattern,
                     corpus_pkl=corpus_pkl, label_pkl="", vocab_pkl="")

            self.n_utters = len(self.uid2sstt)
            self.n_vocab = len(self.vocabulary)
            self.n_labels = len(self.lid2lab)

            # self.get_valid_data()
            self.load_labels(min_sfreq, self.label_mappings)

            self.bow = None
            self.vectorizer = None


    def print_stats(self):
        print("Number of sessions: %d (ones that have text)" % len(self.sstt2uid))
        print("Number of sessions: %d (ones that have labels)" % len(self.sid2labs_all))
        print("Number of labels that originally had: %d (including the ones that appear in the sessions without text)" % len(self.lid2lab_all))
        print("Number of labels: %d (after cleaning the labels)" % self.n_labels)
        print("Vocabulary size: %d" % self.n_vocab)
        print("Number of user-defined stopwords: %d" % len(self.stopwords))
        print("Number of stopwords used in total: %d (including the words with low dfs and high dfs)" % len(self.stopwords_all))


    def assign_codenames(self):
        shortnames = ['BiomedHistorySymptom', 'MusSkePain', 'DizzyFallMemoryDentHearVision',
                      'GynGenitoUrinary', 'Prognosis', 'TestDiagnostics', 'TherapeuticIntervention',
                      'Medication', 'PreventiveCare', 'Diet',
                      'Weight', 'Exercise', 'Alcohol', 'Sex', 'Cigarette',
                      'OtherAddictions', 'Sleep', 'Death', 'Bereavement', 'PainSuffering',
                      'Depression', 'GeneralAnxieties', 'HealthCareSystem', 'ActivityDailyLiving',
                      'WorkLeisureActivity', 'Unemployment', 'MoneyBenefits', 'Caregiver', 'HomeEnv', 'Family',
                      'Religion', 'Age', 'LivingWillAdvanceCarePlanning', 'MDLife', 'MDPT-Relationship',
                      'SmallTalk', 'Other', 'VisitFlowManagement', 'PhysicalExam']
        self.lab2shortname = {str(i + 1): shortnames[i] for i in range(len(shortnames))}


    def load_corpus(self, corpus_file, sep, min_wcnt=1, min_np_len=2, max_np_len=3,
                    token_pattern=r"(?u)\b\w[A-Za-z']*\b",
                    ignore_case=True, remove_numbers=False, sub_numbers=True, parser=None, stemmer_type=None,
                    corpus_pkl='./corpus.pkl', label_pkl='./labels.pkl', vocab_pkl='./vocab.pkl'):

        print('Loading and preprocessing the corpus with labels')
        if not self._load_corpus_pkl(corpus_pkl):
            uidx = 0
            segidx = -1
            uid2sstt = []  # uid to [session,talkturn]
            sstt2uid = {}  # [session][talkturn] to uid
            uid2segid = []  # uid to segment id
            segid2uids = defaultdict(list)  # segment id to utter id
            corpus_txt = []
            sesid_arr, segid_arr = [], []
            rows_to_keep = []

            prev_lab = None
            prev_sesid = -1

            # Does not count on the segments
            sid2labs_all = defaultdict(list)
            # lid2lab_all = set()

            # Does count on the segments
            segid2lab = {}  # segment id to string label
            segid2lt = {}

            # Function that checks the valid label
            is_valid_lab = lambda x: True if len(str(x)) > 0 else False  # Keep nan as one

            # Load the delimited file
            raw_df = pd.read_csv(corpus_file, sep=sep)
            if 'topicnumber' in raw_df.columns:
                raw_df = raw_df[raw_df.topicnumber > 0]
            raw_df = raw_df[raw_df.visitid > 0]

            print("  Cleaning the corpus (removing punctuations..)")
            for i in range(raw_df.shape[0]):
                row = raw_df.iloc[i]

                sesid = int(row['visitid'])
                tt = int(row['talkturn'])
                text_ = row['text']
                lab = row.get('topicnumber', '')
                if is_valid_lab(lab):
                    lab = str(int(lab))
                lab_letter = row.get('topicletter', '')
                if is_valid_lab(lab_letter):
                    lab_letter = str(lab_letter).strip()

                # preprocess the sentence/doc
                if type(text_) == str:
                    text = preproc.remove_punc(text_, ignore_case, remove_numbers=remove_numbers, mhddata=True)

                    if len(text.split()) >= min_wcnt:  # or maybe do len(text) < min_charcnt?

                        sesid_arr.append(sesid)
                        rows_to_keep.append(i)

                        # Finds a segment
                        if (prev_sesid != sesid or prev_lab != lab) and is_valid_lab(lab):
                            segidx += 1
                            segid2lab[segidx] = lab
                            segid2lt[segidx] = lab_letter
                            sid2labs_all[sesid].append(lab)
                        uid2sstt.append([sesid, tt])
                        uid2segid.append(segidx)
                        segid_arr.append(segidx)
                        if sstt2uid.get(sesid, None) is None:
                            sstt2uid[sesid] = {}
                        sstt2uid[sesid][tt] = uidx
                        segid2uids[segidx].append(uidx)
                        corpus_txt.append(text)
                        uidx += 1
                        prev_lab = lab
                        prev_sesid = sesid

            cleaned_df = raw_df.iloc[rows_to_keep]
            cleaned_df = cleaned_df.assign(visitid=sesid_arr, text_cleaned=corpus_txt,
                                           segmentid=segid_arr)
            self.corpus_df = cleaned_df.reset_index(drop=True)
            if 'topicnumber' in self.corpus_df.columns:
                uid2lab = map(str, map(int, list(self.corpus_df.topicnumber)))
            else:
                uid2lab = []

            self._save_corpus_pkl(self.corpus_df, None, None, uid2sstt, sstt2uid,
                                  uid2segid, segid2uids, sid2labs_all, segid2lab,
                                  segid2lt, uid2lab, corpus_pkl)

        if 'topicnumber' in self.corpus_df.columns:
            self.has_label = True

    def load_train_vocab_pkl(self, tr_vocab_pkl):
        if self._load_vocab_pkl(tr_vocab_pkl):
            # Get the reverse vocab mapping
            voc_len = max(self.vocabulary.values()) + 1
            self.vocabulary_inv = [""] * voc_len
            for voc, i in self.vocabulary.iteritems():
                self.vocabulary_inv[i] = voc
            return True
        else:
            return False

    def _load_cleaned_lab_pkl(self, cleaned_lab_pkl):
        if os.path.exists(cleaned_lab_pkl):
            print("Loading cleaned labels file from " + cleaned_lab_pkl)

            lab_data_list = cp.load(open(cleaned_lab_pkl, 'rb'))
            if len(lab_data_list) == 3:
                self.lid2lab, self.lab2lid, self.label_mappings = lab_data_list
                return True
            else:
                print("ERROR: Cannot load the label file." + cleaned_lab_pkl+ " has different format!")
                return False
        else:
            return False

    def load_train_lab_pkl(self, tr_label_pkl, cl_tr_label_pkl):
        self.lid2lab = []
        return self._load_lab_pkl(tr_label_pkl) and self._load_cleaned_lab_pkl(cl_tr_label_pkl)


    def load_labels(self, min_sess_freq=20, label_mappings=None):
        """
        Using this method to keep the parent class' format.
        This function mainly does label cleaning since the labels are loaded with corpus.
        label_mappings argument is just a placeholder since it is using traindata's mappings.
        """
        self.lab2ltr = {'1': 'A', '2': 'A', '3': 'A', '4': 'A', '5': 'A', '6': 'A', '7': 'A',
                        '8': 'A', '9': 'A', '10': 'B', '11': 'B', '12': 'B', '13': 'B', '14': 'B',
                        '15': 'B', '16': 'B', '17': 'B', '18': 'C', '19': 'C', '20': 'C', '21': 'C',
                        '22': 'C', '23': 'D', '24': 'D', '25': 'D', '26': 'D', '27': 'D', '28': 'D',
                        '29': 'D', '30': 'D', '31': 'D', '32': 'D', '33': 'D', '34': 'E', '35': 'E',
                        '36': 'F', '37': 'F', '38': 'G', '39': 'G'}

        # Clean labels (ignore labels that appear less than N times)
        if self.has_label:
            if label_mappings is None:
                label_mappings = self.label_mappings

            cleaned = self.clean_labels(self.sid2labs_all, self.segid2lab_all, self.segid2lt,
                                        self.uid2lab_all, min_sess_freq, label_mappings)

            self.sid2labs, self.sid2lidarr = cleaned[0]
            self.segid2lab, self.segid2lid, self.segid2lidarr, self.segid2ltid = cleaned[1]
            self.uid2lab, self.uid2lid = cleaned[2]

        self.lid2shortname = [self.lab2shortname[self.lid2lab[i]] for i in range(len(self.lid2lab))]


    def clean_labels(self, sid2labs, segid2lab, segid2lt, uid2lab,
                     min_sess_freq=10, label_mappings=None, print_mappings=True):
        """
        Skips the step where it defines the label mappings (by removing the rare topics.)
        Instead it uses the label mappings from the train data (that were loaded by pkl file).
        This function is only performed when the test data has labels.

        Parameters
        ----------
        sid2labs
        segid2lab
        segid2lt
        uid2lab
        min_sess_freq
        label_mappings
        print_mappings

        Returns
        -------

        """
        def print_label_mappings(lab_map):
            for lab in sorted(lab_map.keys()):
                print("  %s %s --> %s %s" % (lab, self.lab2shortname.get(lab, "nan"),
                                             lab_map[lab], self.lab2shortname.get(lab_map[lab], "nan")))

        print("Cleaning labels ..")

        if print_mappings:
            print_label_mappings(label_mappings)

        code_other = '37'

        # update session-level, segment-level, utterance-level label data
        sid_labs = self._update_session_labs(sid2labs, self.lab2lid,
                                             label_mappings, code_other)
        segid_labs = self._update_segment_labs(segid2lab, self.lab2lid, segid2lt,
                                               label_mappings, code_other)
        uid_labs = self._update_utter_labs(uid2lab, self.lab2lid, label_mappings, code_other)

        return sid_labs, segid_labs, uid_labs
