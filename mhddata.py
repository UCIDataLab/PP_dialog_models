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
    """
    Base class for MHD Data

    Vocabulary Related Variables
    ----------------------------
    n_vocab : int
        Size of vocabulary
        (including noun phrases and N-grams, depending on the parameters)
    vocabulary : set[str]
    stopwords_all : set[str]
        A set of stopwords, including the user-defined ones and the ones that are
        very rare or common (those are taken out by min_df and max_df parameters)
    stopwords : set[str]
        A set of user-defined stopwords
    nps : set[str]
        A set of noun phrases
    np2cnt : dict[str, int]
        A dictionary that has noun phrases to its counts


    Label Related Variables
    -----------------------
    n_labels : int
        Number of labels after cleaning/merging

    lid2lab_all : list[str]
        A list of labels. Indices are the label IDs.
        '_all' means that these are the list of labels 'before' cleaning.
        Cleaned (merged) version of this label is 'lid2lab'.
        Labels are topic code, which ranges from '1' ~ '39'.
    lab2lid_all : dict[str,int]
        Reverse map of lid2lab_all
    lid2lab : list[str]
        The same type of information as lid2lab_all 'after' cleaning/merging the data.
        Mapping between label ID/index to label can be different from that of lid2lab_all.
    lab2lid : dict[str,int]
        Reverse map of lid2lab
    lab2name : dict[str, str]
        Label to shortnames. 'lab' refers to 'topic code', which ranges from '1'~'39'.
    lid2name : dict[int, str]
        Label ID/IDX (after cleaning labels) to its shortname mapping
    lab2ltr : dict[str, str]
        Label (topic code) to topic letters. Topic letters range from 'A' ~ 'G'
    ltid2lt : list
        A list of Topic letters. Where indices are the IDs of those topic letters
    lt2ltid : dict[str, int]
        Reverse mapping of ltid2lt


    Corpus related variables
    ------------------------
    * uid : unique identifier of an utterance starts from 0
    * sid : unique identifier of a session. 5 digits.
    * tt/talkturn : index of each talkturn/utterance within each session.
                    This field exists in the original data, and it starts from 0 at
                    the beginning of each session.
    * segid : unique identifier of a segment. A segment is a consecutive list of
              utterances that share the same label.

    n_utters : int
        Number of utterances that we are using.

    uid2sstt : dict[int, dict[int, int]]
        Dictionary that maps utterance ID to its [sessionID, TalkturnID]
    sstt2uid : dict[dict[int, int], int]
        Dictionary that maps [sessionID, talkturnID] to the utterance IDs
    uid2segid : dict[int, int]
        Maps to utterance ID to segment ID
    segid2uids : list[int]
        List that maps segment ID to the list of utterance IDs who belongs to the segment.

    corpus_df : pd.DataFrame
        Pandas dataframe that has all the data. (except for the cleaned version of labels)
    corpus_txt : list[str]
        List of utterances (flattened), where each index is the utterance ID
        It is a mapping from uid to cleaned text of that utterance.

    segid2lab_all : dict[int, str]
        Mapping between the segment id to the label (topic code), before label cleaning
    segid2lab : dict[int, str]
        Same as above, after label cleaning.
    uid2lab_all : list[str]
        Mapping between the utterance id to the label (topic code), before label cleaning
    uid2lab : list[str]
        same as above, after label cleaning
    uid2lid : list[int]
        Mapping between the utterance ID to the label ID


    Others
    ------
    spkrid2spkr : list[str]
        A List that maps from speaker ID/index to speaker
    spkr2spkrid : dict[str,int]
        Reverse map of .spkrid2spkr
    valid_sids, valid_segids, valid_uids : list[int]
        List of session, segment, and utterance IDs that are valid (have labels)

    """

    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True,
                 stopwords_dir="./stopwordlists", proper_nouns_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,2), max_np_len=3, min_wlen=1,
                 min_dfreq=0.0001, max_dfreq=0.9, min_sfreq=5,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+", verbose=1,
                 corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):
        """

        Parameters
        ----------
        data_file : str
            File path to the data
        nouns_only : bool
            Only include nouns
        ignore_case : bool
            True if ignoring cases (transformed to all lower case, default)
        remove_numbers : bool
            Remove all digits if True, default is False
        sub_numbers : bool
            Rather than removing all digits, substitute them to symbol -num-. default True
        stopwords_dir : str
            Path to the stopwords directory
        proper_nouns_dir : str
            Path to the proper nouns directory
        label_mappings : dict or None
            Label mappings that can be use for label merging/cleaning.
            If None (default), rare labels are mapped to the 'Other' label (topic code: '37')
        ngram_range : tuple
            The lower and upper boundary of the range of n-values for different n-grams
            to be extracted. (Maximum length of noun phrase can be different from this.)
        max_np_len : int
            Maximum length of noun phrases to be considered.
        min_wlen : int
            Minimum word length of an utterance to be considered.
            If an utterance has word length less than 'min_wlen', the utterance is skipped.
        min_dfreq : float
            float in range [0.0, 1.0]
            When building the vocabulary ignore terms that have
            a document frequency strictly lower than the given threshold.
            The parameter represents a proportion of documents, integer absolute counts.
            NOTE: Here, each utterance is treated as a document, therefore very small value would
            still reduce the size of vocabulary. Because the utterances are very short therefore most of the words have very
            small document frequencies.
        max_dfreq : float
            float in range [0.0, 1.0]
            When building the vocabulary ignore terms that have a document frequency strictly higher than
            the given threshold (corpus-specific stop words).
            The parameter represents a proportion of documents, integer absolute counts.
        min_sfreq : int
            If there are labels that appear less than this number of sessions,
            the labels are merged to the 'other' label (topic code: '37')
        token_pattern : str
            Regular expression denoting what constitutes a 'token'.
        verbose : int in range [0,3]
            The level of verbosity. Larger value means more verbosity.
        corpus_pkl : str
            Path to the pickle file that saves corpus related data.
        label_pkl : str
            Path to the pickle file that saves label related data.
        vocab_pkl : str
            Path to the pickle file that saves vocab related data.
        """
        self.verbose = verbose

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
                         sub_numbers=sub_numbers, proper_nouns_dir=proper_nouns_dir,
                         corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        self.assign_labnames()

        self.assign_spkr_codes()

    def assign_labnames(self):
        """
        Assigns short names for each topic code.
        Original topic code ranges from 1~39.
        Topic codes are called as 'lab', and they are treated as strings.
        """
        shortnames = ['BiomedHistorySymptom', 'MusSkePain', 'DizzyFallMemoryDentHearVision',
                      'GynGenitoUrinary', 'Prognosis', 'TestDiagnostics', 'TherapeuticIntervention',
                      'Medication', 'PreventiveCare', 'Diet',
                      'Weight', 'Exercise', 'Alcohol', 'Sex', 'Cigarette',
                      'OtherAddictions', 'Sleep', 'Death', 'Bereavement', 'PainSuffering',
                      'Depression', 'GeneralAnxieties', 'HealthCareSystem', 'ActivityDailyLiving',
                      'WorkLeisureActivity', 'Unemployment', 'MoneyBenefits', 'Caregiver', 'HomeEnv', 'Family',
                      'Religion', 'Age', 'LivingWillAdvanceCarePlanning', 'MDLife', 'MDPT-Relationship',
                      'SmallTalk', 'Other', 'VisitFlowManagement', 'PhysicalExam']
        self.lab2name = {str(i + 1): shortnames[i] for i in range(len(shortnames))}
        self.lab2ltr = {'1': 'A', '2': 'A', '3': 'A', '4': 'A', '5': 'A', '6': 'A', '7': 'A',
                        '8': 'A', '9': 'A', '10': 'B', '11': 'B', '12': 'B', '13': 'B', '14': 'B',
                        '15': 'B', '16': 'B', '17': 'B', '18': 'C', '19': 'C', '20': 'C', '21': 'C',
                        '22': 'C', '23': 'D', '24': 'D', '25': 'D', '26': 'D', '27': 'D', '28': 'D',
                        '29': 'D', '30': 'D', '31': 'D', '32': 'D', '33': 'D', '34': 'E', '35': 'E',
                        '36': 'F', '37': 'F', '38': 'G', '39': 'G'}

    def assign_spkr_codes(self):
        """
        Assigns speaker codes.
        """
        self.spkrid2spkr = ["MD", "PT", "OTHER", "RA", "RN"]
        self.spkr2spkrid = {sp: i for i, sp in enumerate(self.spkrid2spkr)}

    def load_corpus(self, corpus_file, sep, min_wcnt=1, min_np_len=2, max_np_len=3,
                    token_pattern=r"(?u)\b\w[A-Za-z']*\b",
                    ignore_case=True, remove_numbers=False, sub_numbers=True, parser=None, stemmer_type=None,
                    proper_nouns_dir="./stopwordlists",
                    corpus_pkl='./corpus.pkl', label_pkl='./labels.pkl', vocab_pkl='./vocab.pkl'):
        """
        Train and test will have slightly different process.
        """
        pass

    def _save_lab_pkl(self, lid2lab_all, segid2lt, label_pkl):
        """
        Saves label related data to a pickle file. (before cleaning)
        Also creates ltid2lt, lt2ltid, lab2lid_all, lid2lab_all

        Parameters
        ----------
        lid2lab_all : list
            Label id to label mapping before cleaning
        segid2lt : list
            Segment id to the topic letters
        label_pkl : str
            File path to the pickle file which the label data will be saved to
        """
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
        """
        Saves vocab related data to a pickle file.
        also assigns self.vocabulary and self.stopwords_all

        Parameters
        ----------
        nps : set
            Set of noun phrases
        vocab : set
            set of vocabularies
        stw_all : set
            Set of all the stopwords (including user-defined vocabs + vocabs by doc freq cutoffs)
        vocab_pkl : str
            File path to the pickle file which the vocab data will be saved to
        """
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
        """
        Saves corpus related data to a pickle file.
        Also assigns bunch of variables as member variables.

        For parameter information, please refer to the top of this code,
        description of MHDData class.

        Parameters
        ----------
        corpus_df : pd.DataFrame
        nps : set[str]
        np2cnt : dict[str, int]
        uid2sstt : list[int, dict[int, int]]
        sstt2uid : dict[dict[int, int], int]
        uid2segid : list[int]
        segid2uids : dict[int]
        sid2labs_all : dict[int, str]
        segid2lab_all : dict[str] or defaultdict(str)
        segid2lt : dict[str] or defaultdict(str)
        uid2lab_all : list[str]
        corpus_pkl : str
            File path to the pickle file which the corpus related data will be saved to

        """
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
            if self.verbose > 0:
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
            if self.verbose > 0:
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
            if self.verbose > 0:
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

    def clean_labels(self, min_sess_freq=20, label_mappings=None):
        pass

    def _update_session_labs(self, sid2labs, lab2lid_new, label_mappings, code_other):
        """
        Used inside of clean_labels.
        Update the old set of session-level label information
        to the new using 'label_mappings'
        """
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
        """
        Used inside of clean_labels.
        Update the old set of segment-level label information
        to the new using 'label_mappings'
        """
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
        """
        Used inside of clean_labels.
        Update the old set of utterance-level label information
        to the new using 'label_mappings'
        """
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
        """
        Given a string, returns an ID of the speaker.
        String that describes speaker can be a bit noisy in the raw data,
        this function cleans and assigns the ID.

        Parameters
        ----------
        spkr : str
            Description of a speaker, or a speaker type in string.
            (e.g. PT, patient, clinician, RA, etc.)
        include_ra : bool
            True if you want to treat RA as a separate speaker
            Default is False. If False, the RA will have the code for OTHERS.
        include_nurse : bool
            True if you want to treat RN as a separate speaker
            Default is False. If False, the RN will have the code for OTHERS.

        Returns
        -------
        int
            ID of a speaker

        """
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
        """
        Given a list of utterance IDs, and
        the function 'get_spkr_code' with appropriate parameters
        to get the list of speaker IDs of those utterances.

        Parameters
        ----------
        uids : list[int] or list[list[int]]
            list of utterances.
            It can be nested or flattened.
        get_spkr_category : method
            self.get_spkr_code method with appropriate arguments
        nested : bool
            True if nested.

        Returns
        -------
        list[int]
            List of speaker IDs.
        """
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
        """
        Fit bag of words or TF-IDF

        Parameters
        ----------
        train_doc
        tfidf
        vocabulary
        stop_words
        token_pattern
        max_wlen

        Returns
        -------

        """

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
        Get nested data using a list of session IDs

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
        Given a list of session or segment IDs, convert them to a list of
        corresponding segment IDs or utterance IDs.

        Parameters
        ----------
        ids_list : list[int]
            List of session IDs or segment IDs
        from_level : str
            Either 'session' or 'segment'
        to_level : str
            Either 'segment' or 'utterance'
        insert_st_end : bool
            If True, -1 and -2 will be inserted for start and end of the session or segment
            (depending on the 'from_level').
            (Not used currently. default=False)

        Returns
        -------
        list[int]
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
        Given a list of utterance IDs, return the corresponding text and the label data.
        (If the utterance ID has been changed due to chunk sizes and overlaps.)

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
                labid = -1  # If there's no label, assign -1.
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


class MHDTrainData(MHDData):

    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                 min_dfreq=0.0, max_dfreq=0.9, min_sfreq=20,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+", verbose=1,
                 corpus_pkl='./corpus.pkl', label_pkl='./label.pkl', vocab_pkl='./vocab.pkl'):

        MHDData.__init__(self, data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                         remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                         label_mappings=label_mappings, ngram_range=ngram_range,
                         max_np_len=max_np_len, min_wlen=min_wlen,
                         min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                         token_pattern=token_pattern, verbose=verbose,
                         corpus_pkl=corpus_pkl, label_pkl=label_pkl, vocab_pkl=vocab_pkl)

        cl_lab_pkl = label_pkl.split(".pkl")[0] + "_cleaned.pkl"

        # Cleans and save cleaned data so that the TestData can load and use
        self.clean_labels(min_sess_freq=min_sfreq, label_mappings=label_mappings)
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
        """
        Get valid session, segment, utterance IDs that has labels
        """
        # Get session IDs and utterance IDs that both have the text and the labels
        if self.verbose > 0:
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

    def load_corpus(self, corpus_file, sep, min_wcnt=1, min_np_len=2, max_np_len=3,
                    token_pattern=r"(?u)\b\w[A-Za-z']*\b",
                    ignore_case=True, remove_numbers=False, sub_numbers=True, parser=None, stemmer_type=None,
                    proper_nouns_dir="./stopwordlists",
                    corpus_pkl='./corpus.pkl', label_pkl='./labels.pkl', vocab_pkl='./vocab.pkl'):
        """
        Read corpus from 'corpus_file',
        cleans the text, (cleaning is mostly done in preprocess.py)
        finds segments and assigns segment IDs,
        saves all the data.

        For parameters, see the parameter descriptions of the MHDData and MHDTrainData class.

        Parameters
        ----------
        corpus_file
        sep
        min_wcnt
        min_np_len
        max_np_len
        token_pattern
        ignore_case
        remove_numbers
        sub_numbers
        parser
        stemmer_type
        proper_nouns_dir
        corpus_pkl
        label_pkl
        vocab_pkl
        """

        if self.verbose > 0:
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

            # Load the list of proper nouns
            names_to_sub = preproc.read_words(os.path.join(proper_nouns_dir, "names.txt"), ignore_case)
            locs_to_sub = preproc.read_words(os.path.join(proper_nouns_dir, "locations.txt"), ignore_case)

            if self.verbose > 1:
                print("  Cleaning the corpus (removing punctuations..)")

            for i in range(raw_df.shape[0]):
                if self.verbose > 2 and i % 5000 == 0:
                    print("   %10d utterances" % i)
                row = raw_df.iloc[i]
                sesid = int(row['visitid'])
                tt = int(row['talkturn'])
                text_ = row['text']
                lab = str(int(row['topicnumber']))
                lab_letter = str(row['topicletter']).strip()

                # preprocess the sentence/doc
                if type(text_) == str:
                    text = preproc.remove_punc(text_, ignore_case, remove_numbers=remove_numbers,
                                               names_list=names_to_sub, locations_list=locs_to_sub,
                                               is_mhddata=True)

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
                                            parser, stemmer_type, self.verbose)

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

    def clean_labels(self, min_sess_freq=20, label_mappings=None):
        """
        Clean labels. If there is a 'label_mapping' given, it does the label merging/updates
        using the 'label_mapping'.
        If not, rare labels that appears less than 'miss_sess_freq' sessions
        will be merged to 'Others' label.

        Parameters
        ----------
        min_sess_freq : int
            Ignore labels that appear less than min_sess_freq times
        label_mappings : dict or None
            Label mappings can be given manually.
        """
        if self.verbose > 0:
            print("Cleaning labels ..")
        self.n_labels = len(self.lid2lab_all)

        # Clean labels (ignore labels that appear less than N times)
        cleaned = self._clean_labels_inner(self.lid2lab_all, self.lab2lid_all,
                                           self.sid2labs_all, self.segid2lab_all, self.segid2lt,
                                           self.uid2lab_all, min_sess_freq, label_mappings)

        self.lid2lab, self.lab2lid = cleaned[0]
        self.sid2labs, self.sid2lidarr = cleaned[1]
        self.segid2lab, self.segid2lid, self.segid2lidarr, self.segid2ltid = cleaned[2]
        self.uid2lab, self.uid2lid = cleaned[3]
        self.label_mappings = cleaned[-1]
        self.lid2name = [self.lab2name[self.lid2lab[i]] for i in range(len(self.lid2lab))]


    def _clean_labels_inner(self, lid2lab, lab2lid, sid2labs, segid2lab, segid2lt, uid2lab,
                            min_sess_freq=10, label_mappings=None):
        """
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
                print("  %s %s --> %s %s" % (lab, self.lab2name.get(lab, "nan"),
                                             lab_map[lab], self.lab2name.get(lab_map[lab], "nan")))
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

        # Avoid deleting 'other' code
        code_other = '37'
        labmat[:, self.lab2lid_all[code_other]] += min_sess_freq

        # Delete rare labels
        labids_to_keep = np.where(np.sum(labmat, axis=0) >= min_sess_freq)[0]
        labids_to_merge = np.where(np.sum(labmat, axis=0) < min_sess_freq)[0]
        if len(labids_to_merge) > 0 and len(label_mappings) == 0:
            # If mapping was not defined, map the rare labels to 'others'
            label_mappings = {self.lid2lab_all[lid]: code_other for lid in labids_to_merge}

        if self.verbose > 1:
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


class MHDTestData(MHDData):
    """
    Additional parameters in this class


    """
    def __init__(self, data_file, nouns_only=False, ignore_case=True,
                 remove_numbers=False, sub_numbers=True, stopwords_dir="./stopwordlists",
                 label_mappings=None, ngram_range=(1,1), max_np_len=2, min_wlen=1,
                 min_dfreq=0.0, max_dfreq=0.9, min_sfreq=10,
                 token_pattern=r"(?u)[A-Za-z\?\!\-\.']+", verbose=1,
                 corpus_pkl='./corpus_test.pkl', tr_label_pkl="./label.pkl", tr_vocab_pkl="./vocab.pkl"):

        self.verbose = verbose
        cl_lab_pkl = tr_label_pkl.split(".pkl")[0] + "_cleaned.pkl"
        if self.load_train_lab_pkl(tr_label_pkl, cl_lab_pkl) and self.load_train_vocab_pkl(tr_vocab_pkl):

            self.has_label = False

            MHDData.__init__(self, data_file, nouns_only=nouns_only, ignore_case=ignore_case,
                             remove_numbers=remove_numbers, sub_numbers=sub_numbers, stopwords_dir=stopwords_dir,
                             label_mappings=label_mappings, ngram_range=ngram_range,
                             max_np_len=max_np_len, min_wlen=min_wlen,
                             min_dfreq=min_dfreq, max_dfreq=max_dfreq, min_sfreq=min_sfreq,
                             token_pattern=token_pattern, verbose=verbose,
                             corpus_pkl=corpus_pkl, label_pkl="", vocab_pkl="")

            self.n_utters = len(self.uid2sstt)
            self.n_vocab = len(self.vocabulary)
            self.n_labels = len(self.lid2lab)
            self.clean_labels(min_sfreq, self.label_mappings)

    def print_stats(self):
        print("Number of sessions: %d (ones that have text)" % len(self.sstt2uid))
        print("Number of sessions: %d (ones that have labels)" % len(self.sid2labs_all))
        print("Number of labels that originally had: %d (including the ones that appear in the sessions without text)" % len(self.lid2lab_all))
        print("Number of labels: %d (after cleaning the labels)" % self.n_labels)
        print("Vocabulary size: %d" % self.n_vocab)
        print("Number of user-defined stopwords: %d" % len(self.stopwords))
        print("Number of stopwords used in total: %d (including the words with low dfs and high dfs)" % len(self.stopwords_all))

    def load_corpus(self, corpus_file, sep, min_wcnt=1, min_np_len=2, max_np_len=3,
                    token_pattern=r"(?u)\b\w[A-Za-z']*\b",
                    ignore_case=True, remove_numbers=False, sub_numbers=True, parser=None, stemmer_type=None,
                    proper_nouns_dir="./stopwordlists",
                    corpus_pkl='./corpus.pkl', label_pkl='./labels.pkl', vocab_pkl='./vocab.pkl'):
        """
        Read corpus from 'corpus_file', which is the test file.
        cleans the text, (cleaning is mostly done in preprocess.py)
        finds segments and assigns segment IDs,
        saves all the data.

        Parameters
        ----------
        corpus_file
        sep
        min_wcnt
        min_np_len
        max_np_len
        token_pattern
        ignore_case
        remove_numbers
        sub_numbers
        parser
        stemmer_type
        proper_nouns_dir
        corpus_pkl
        label_pkl
        vocab_pkl

        Returns
        -------

        """
        if self.verbose > 0:
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

            names_to_sub = preproc.read_words(os.path.join(proper_nouns_dir, "names.txt"), ignore_case)
            locs_to_sub = preproc.read_words(os.path.join(proper_nouns_dir, "locations.txt"), ignore_case)

            if self.verbose > 1:
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
                    text = preproc.remove_punc(text_, ignore_case, remove_numbers=remove_numbers,
                                               names_list=names_to_sub, locations_list=locs_to_sub, is_mhddata=True)

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

        # Set the extra variable 'has_label' when the data has the column 'topicnumber'
        if 'topicnumber' in self.corpus_df.columns:
            self.has_label = True

    def load_train_vocab_pkl(self, tr_vocab_pkl):
        """
        Loads vocab data from the pickle file.
        """
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
        """
        Loads the cleaned (merged) label data from the pickle file.
        (The ones without '_all')
        """
        if os.path.exists(cleaned_lab_pkl):
            if self.verbose > 0:
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
        """
        Loads all the label data from the training step.
        (Both the cleaned version and the version before cleaning.)
        """
        self.lid2lab = []
        return self._load_lab_pkl(tr_label_pkl) and self._load_cleaned_lab_pkl(cl_tr_label_pkl)

    def clean_labels(self, min_sess_freq=20, label_mappings=None):
        """
        For test data, all the parameters are redundant since it will use
        the same label cleaning process that training data used.
        """
        if self.has_label:
            if label_mappings is None:
                label_mappings = self.label_mappings

            if self.verbose > 0:
                print("Cleaning labels ..")
            cleaned = self._clean_labels_inner(self.sid2labs_all, self.segid2lab_all, self.segid2lt,
                                               self.uid2lab_all, label_mappings)

            self.sid2labs, self.sid2lidarr = cleaned[0]
            self.segid2lab, self.segid2lid, self.segid2lidarr, self.segid2ltid = cleaned[1]
            self.uid2lab, self.uid2lid = cleaned[2]

        self.lid2name = [self.lab2name[self.lid2lab[i]] for i in range(len(self.lid2lab))]

    def _clean_labels_inner(self, sid2labs, segid2lab, segid2lt, uid2lab,
                            label_mappings=None):
        """
        Skips the step where it defines the label mappings (by removing the rare topics.)
        Instead it uses the label mappings from the train data (that were loaded by pkl file).
        This function is only performed when the test data has labels.
        """

        def print_label_mappings(lab_map):
            for lab in sorted(lab_map.keys()):
                print("  %s %s --> %s %s" % (lab, self.lab2name.get(lab, "nan"),
                                             lab_map[lab], self.lab2name.get(lab_map[lab], "nan")))

        if self.verbose > 1:
            print_label_mappings(label_mappings)

        code_other = '37'
        # update session-level, segment-level, utterance-level label data
        sid_labs = self._update_session_labs(sid2labs, self.lab2lid,
                                             label_mappings, code_other)
        segid_labs = self._update_segment_labs(segid2lab, self.lab2lid, segid2lt,
                                               label_mappings, code_other)
        uid_labs = self._update_utter_labs(uid2lab, self.lab2lid, label_mappings, code_other)

        return sid_labs, segid_labs, uid_labs
