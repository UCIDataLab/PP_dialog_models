# coding=utf-8
__author__ = 'Jihyun Park'
__email__ = 'jihyunp@uci.edu'

import nltk
import re
import os
import csv
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

def remove_punc(rawtext, ignore_case=False, remove_numbers=False, sub_numbers=True, mhddata=False):
    """

    Parameters
    ----------
    rawtext : str
    ignore_case: bool
    remove_numbers: bool
        Removes all numbers if True
    sub_numbers: bool
        Substitutes all numbers to a token -num-.
        if 'remove_numbers' is True, it will have no effect.

    Returns
    -------
    str

    """
    # from NLTK

    #ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), " '' "),
        (re.compile(r'(\S)(\'\')'), r'\1 \2 '),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                     re.compile(r"(?i)\b(d)('ye)\b"),
                     re.compile(r"(?i)\b(gim)(me)\b"),
                     re.compile(r"(?i)\b(gon)(na)\b"),
                     re.compile(r"(?i)\b(got)(ta)\b"),
                     re.compile(r"(?i)\b(lem)(me)\b"),
                     re.compile(r"(?i)\b(mor)('n)\b"),
                     re.compile(r"(?i)\b(wan)(na) ")]
    CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                     re.compile(r"(?i) ('t)(was)\b")]


    # TODO : Might have to modify the regular expressions below
    # For names
    txt = re.sub(r'[\[\{\(\<]patient name[\]\}\)\>]', ' -name- ', rawtext)

    # First remove all parentheses and brackets
    txt = re.sub(r'[\[\{\(\<][^\[\{\(\<]*[\]\}\)\>]', ' ', txt)

    if remove_numbers:
        # remove everything except for alpha characters and ', -, ?, !, .
        txt = re.sub(r'[^\.,A-Za-z\'\-\?\! ]', ' ', txt)
    else:
        # remove everything except for alphanumeric characters and ', -, ?, !, .
        txt = re.sub(r'[^\.,A-Za-z0-9\'\-\?\! ]', ' ', txt)
        # split numbers
        txt = re.sub(r"([0-9]+)", r" \1 ", txt)

        if sub_numbers:
            txt = re.sub(r'[0-9]+', '-num-', txt)

    if mhddata:
        # Replace non-ascii characters (for now just replace one)
        txt = re.sub("\x92", "'", txt)
        txt = re.sub("í", "'", txt)
        # Remove all NAs
        txt = re.sub(r"\bNA\b", "", txt)

    if ignore_case:
        txt = txt.lower()

    # space out the periods and commas
    txt = re.sub(r"\.", " . ", txt)
    txt = re.sub(r",", " , ", txt)
    # split ! and ?
    txt = re.sub(r'\!', ' ! ', txt)
    txt = re.sub(r'\?', ' ? ', txt)

    #add extra space to make things easier
    txt = " " + txt + " "

    # remove dashes that are used alone -- but this might be useful later
    # for now we're just removing them
    txt = re.sub(r' \-\-* ', ' ', txt)
    # remove -- these (consecutive dashes)
    txt = re.sub(r'\-\-+', ' ', txt)

    #add extra space to make things easier
    txt = " " + txt + " "

    # Added these two to find a match with the pre-trained words
    # txt = re.sub(r'[A-Za-z]\-$', '<PARTIALWORD>', txt) # add this? probably not at the moment.
    # txt = re.sub(r'\-', ' ', txt)
    txt = re.sub(r"''", "'", txt)
    txt = re.sub(r"\s+'\s+", " ", txt)

    for regexp, substitution in ENDING_QUOTES:
        txt = regexp.sub(substitution, txt)

    for regexp in CONTRACTIONS2:
        txt = regexp.sub(r' \1 \2 ', txt)
    for regexp in CONTRACTIONS3:
        txt = regexp.sub(r' \1 \2 ', txt)

    txt = re.sub(r'\s+', ' ', txt) # make multiple white spaces into 1

    return txt.strip()


def define_vocabulary(doc, stopwords=None, token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                            ngram_range=(1,2), min_dfreq=0.0001, max_dfreq=0.9,
                            min_np_len=2, max_np_len=3, ignore_case=False,
                            parser=None, stemmer_type=None):

    vocab, stw_all, tokenizer = define_vocabulary_inner(doc, stopwords, token_pattern,
                                                        ngram_range, min_dfreq, max_dfreq)
    if min_np_len > 1:
        print("  Extracting noun phrases..")
        corpus_pos, np2cnt = get_noun_phrase_counts(doc, tokenizer,
                                                    min_np_len, max_np_len, stw_all,
                                                    ignore_case, parser, stemmer_type)
        # vocabulary with noun phrases (min count = 5)
        min_dcnt = max(5, int(min_dfreq * len(doc)))
        vocab, nps = add_noun_phrases_to_vocab(vocab, np2cnt, min_dcnt)
    else:
        nps = set()
        np2cnt = dict()
        corpus_pos = []

    return vocab, stw_all, np2cnt, nps, corpus_pos


def define_vocabulary_inner(doc, stopwords=None, token_pattern=r"(?u)[A-Za-z\?\!\-\.']+",
                      ngram_range=(1,2), min_dfreq=1e-5, max_dfreq=0.9):
    """
    Tokenize the document and attach POS tag.
    Also return the noun phrases from the document.

    Parameters
    ----------
    doc : list[str]
    stemmer_type : str

    Returns
    -------
    list[list[tup[str]]], set, defaultdict

    """
    # This countvectorizer is used only for creating vocabulary
    cntvec = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords,
                             min_df=min_dfreq, max_df=max_dfreq, token_pattern=token_pattern)
    cntvec.fit(doc)
    vocabulary = cntvec.vocabulary_
    stopwords_all = stopwords.union(cntvec.stop_words_)
    tokenizer = cntvec.build_tokenizer()

    return vocabulary, stopwords_all, tokenizer


def add_noun_phrases_to_vocab(vocabulary, np2cnt, min_dcnt):
    nps = set()
    npsinorder = sorted(np2cnt.keys(), key= lambda x: np2cnt[x], reverse=True)
    # Save noun phrases that appeared more than min_dcnt docs
    # Also add those noun phrases to the vocabulary
    i = len(vocabulary)
    for np in npsinorder:
        # stop saving the ones that appeared less than min_dcnt documents
        if np2cnt[np] < min_dcnt:
            break
        nps.add(np)
        vocabulary[np] = i
        i += 1
    return vocabulary, nps


def get_noun_phrase_counts(doc, tokenizer, min_np_len, max_np_len, stopwords_list,
                           ignore_case, parser, stemmer_type):
    is_stopword = defaultdict(int)
    for stw in stopwords_list:
        is_stopword[stw] = 1

    corpus_postag = []
    np2cnt = defaultdict(int) # noun phrase to document count
    for text in doc:
        # much faster when using regex & lambda (cntvec's tokenizer) than using nltk's word tokenizer
        # toktxt = nltk.pos_tag(nltk.word_tokenize(text))
        toktxt = nltk.pos_tag(tokenizer(text.decode('utf-8')))
        corpus_postag.append(toktxt)
        # update the noun phrase list
        if len(toktxt) > 0:
            np_subset = get_noun_phrases_inner(toktxt, min_np_len, max_np_len, is_stopword,
                                          ignore_case=ignore_case, parser=parser, stemmer_type=stemmer_type)
            for np in np_subset:
                np2cnt[np] += 1
    return corpus_postag, np2cnt

def get_noun_phrases_inner(tagged_text, min_np_len=2, max_np_len=3, stopwords_map=None, ignore_case=True,
                         parser=None, stemmer_type=None):
    """
    Return a set of noun phrases that are extracted from the pos-tagged sentence/text.

    Parameters
    ----------
    tagged_text : Tree or list[tuple[str,str]]
        POS tagged tokenized text
    max_np_len : int
        Maximum word length of noun phrases
    stopwords_map : defaultdict(int)
        A dictionary that indicates whether the key (string) is a stopword (value=1) or not (value=0)
    ignore_case : bool
        True (default)
    parser : nltk.RegexpParser or None
        If None, it will use the RegexpParser with "NP: {<VBN|JJ|NN>*<NN|NNS>}")
    stemmer_type : str or None ---- currently not supported!
        If None, stmmer is not used.
        'porter' : porter stemmer
        'snowball' : snowball stemmer

    Returns
    -------
    set
        A set of noun phrases
    """

    if parser is None:
        parser = nltk.RegexpParser(r"NP: {<VBN|JJ|NN>*<NN|NNS>}")

    # Currently not used!
    if stemmer_type is not None:
        if stemmer_type == 'snowball':
            stemmer = nltk.stem.SnowballStemmer('english')
        else:
            stemmer = nltk.stem.PorterStemmer()

    nps = set()
    parsed_txt = parser.parse(tagged_text)
    for tok in parsed_txt:
        if not isinstance(tok[0], basestring): # tok[0] will be a tree (not str) if they are parsed as NPs.
            if len(tok) < min_np_len or len(tok) > max_np_len:
                continue
            words = map(lambda k: k[0], tok)
            if ignore_case:
                add = True
                for i in xrange(len(words)):
                    if not words[i].isupper(): # if all-caps words, don't change it to lower case
                        words[i] = words[i].lower()
                    if stopwords_map[words[i]] == 1:
                        add = False
                        break
                    if len(words[i]) < 2:
                        add = False
                        break
                    ## Stemmer is not supported at the moment
                    # if stemmer_type is not None:
                    #     words[-1] = stemmer.stem(words[-1])
                if add:
                    nps.add(" ".join(words))
    return nps


def get_stopwords(stopwords_dir='./stopwordlists'):
    # This list of English stop words is taken from the "Glasgow Information
    # Retrieval Group". The original list can be found at
    # http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
    # This is the one that scikit learn uses.
    # Also some stopwords are from Garren Gaut.

    # stopwords_unicode = {s.decode('utf-8') if isinstance(s, basestring) else s for s in ENGLISH_STOP_WORDS}

    stopwords = set()
    if os.path.isdir(stopwords_dir):
        for fname in os.listdir(stopwords_dir):
            if fname.split(".")[-1] == "txt":
                fpath = os.path.join(stopwords_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        reader = csv.reader(f, delimiter="\t")
                        for line in reader:
                            stopwords.add(' '.join(line).decode('utf-8'))

    return stopwords

