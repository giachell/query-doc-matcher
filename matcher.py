#
# QueryDocMatcher
#
# Author: Fabio Giachelle <giachelle.fabio@gmail.com>
# URL: <https://github.com/giachell/query-doc-matcher>
# License: MIT

"""
QueryDocMatcher

This module provides several functions to obtain the matching words between a topic (query) and a document.
The matching words are computed by taking into account also stopwords removal, stemming and lemmatization.
The matching words are ranked by tf-idf scores.

There is also a demo function: `matcher.demo()`.

"""

from operator import itemgetter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import preprocess_string
from logger import IcLogger
import pandas as pd


class QueryDocMatcher:
    def __init__(self, topic, doc, corpus, language="english", df_tfidf=None, k=10, logging=False):
        self.topic = topic
        self.doc = doc
        self.corpus = corpus
        self.language = language
        self.k = k
        self.df_tfidf = df_tfidf
        self.ic_logger = IcLogger(print_status=logging)
        self.bow_topic = set()
        self.bow_doc = set()
        self.bow_doc_stemmed = set()
        self.matching_bow_stemmed = set()
        self.matching_bow = set()
        self.map_stemmed_bow_doc_to_not_stemmed = None

    def get_bag_of_words(self, text, stopwords_removal=True, stemming=True, lemming=False):
        preprocessed_text = preprocess_string(text)

        self.ic_logger.log(preprocessed_text)

        # init stopwords, stemmer and lemmatizer tools
        stopwords_nltk = stopwords.words(self.language)
        snow_stemmer = SnowballStemmer(language=self.language)
        lemmatizer = WordNetLemmatizer()
        bows = []
        stem_bows = None
        lemm_bows = None

        if stopwords_removal:
            # remove (filter) every stop word contained in stopwords_nltk for the specified language (default: "english")
            bows = [word for word in preprocessed_text if word.lower() not in stopwords_nltk]

        if stemming:
            # stem's of each word
            stem_bows = [snow_stemmer.stem(word) for word in bows]

            bows = [stem_bow for stem_bow in stem_bows]

        if lemming:
            # get the corresponding lemma of each word
            lemm_bows = [lemmatizer.lemmatize(word) for word in bows]
            # merge "bows" list with "lemm_bows"
            bows = bows + lemm_bows

        self.ic_logger.log(bows)

        return bows

    def custom_tokenizer(self, text):
        bow = self.get_bag_of_words(text)

        return bow

    def gen_tfidf_map(self):
        # init sklearn TfidfVectorizer
        tfidfvectorizer = TfidfVectorizer(analyzer='word', tokenizer=self.custom_tokenizer, stop_words=self.language)
        # tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words=self.language)

        tfidf_wm = tfidfvectorizer.fit_transform([doc['text'] for doc in self.corpus])
        self.ic_logger.log(tfidf_wm)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()
        # create pandas DataFrame from tfidf matrix
        df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), index=[doc['docno'] for doc in self.corpus],
                                    columns=tfidf_tokens)
        self.ic_logger.log(tfidf_tokens)
        self.ic_logger.log(df_tfidfvect)

        self.df_tfidf = df_tfidfvect

        return df_tfidfvect

    def topic_doc_matching_words(self):

        matching_words = set()

        for t_i in self.bow_topic:
            for dt_i in self.bow_doc:
                self.ic_logger.log(t_i, dt_i)
                if dt_i.startswith(t_i):
                    matching_words.add(t_i)

        return matching_words

    def get_corresponding_word_in_doc(self, word, doc_bow):

        corresponding_word = None

        for doc_wi in doc_bow:
            if doc_wi == word:
                corresponding_word = doc_wi
                break

        return corresponding_word

    def get_top_k_matching_words(self, docno):

        # list of the top-k matching words
        top_k_matching_words = []

        for w in self.matching_bow:
            c_w = self.get_corresponding_word_in_doc(w, self.df_tfidf.columns)
            if c_w is not None:
                c_w_not_stemmed = self.map_stemmed_bow_doc_to_not_stemmed[c_w]
                top_k_matching_words.append((c_w_not_stemmed, self.df_tfidf._get_value(docno, c_w)))

        self.ic_logger.log(top_k_matching_words)

        # sort list with key
        top_k_matching_words.sort(key=itemgetter(1), reverse=True)

        self.ic_logger.log(top_k_matching_words)

        # get top-k matching words (keeps only the top-k ones)

        return top_k_matching_words



    def gen_map_bow(self):
        dict_stemmed_not_stemmed = {}
        for bow_i_stemmed in self.bow_doc_stemmed:
            for bow_i_not_stemmed in self.bow_doc:
                if preprocess_string(bow_i_not_stemmed) == bow_i_stemmed:
                    dict_stemmed_not_stemmed[bow_i_stemmed] = bow_i_not_stemmed

        self.map_stemmed_bow_doc_to_not_stemmed = dict_stemmed_not_stemmed

    def get_words_to_highlight(self):

        # topic data
        title = self.topic["title"]
        desc = self.topic["description"]
        topic_joint_text = ' '.join([title, desc])

        # document data
        docno = self.doc['docno']
        doc_text = self.doc['text']

        # compute bow for topic and document (for the document compute bow in both cases: (not) stemmed)
        self.bow_topic = self.get_bag_of_words(topic_joint_text)
        self.bow_doc = self.get_bag_of_words(doc_text, stemming=False)
        self.bow_doc_stemmed = self.get_bag_of_words(doc_text, stemming=True)

        # get matching words stemmed
        self.matching_bow_stemmed = self.topic_doc_matching_words()
        self.matching_bow = self.matching_bow_stemmed

        # print "matching_words" in case verbose=True
        self.ic_logger.log(self.matching_bow_stemmed)

        self.gen_tfidf_map()
        # compute top-k matching words sorted (descending) according to tfidf score
        top_k_matching_words = self.get_top_k_matching_words(docno)

        return top_k_matching_words

    @staticmethod
    def demo():
        """
        This function provides a demonstration of the QueryDocMatcher module.

        After invoking this function, the top-k matching words between a toy query (topic) and document are computed.
        Finally, the list of top-k matching words sorted by tf-idf score is printed.

        """

        topic = {
            "title": "Cities the First Lady visited on official business.",
            "description": "What cities other than Washington D.C. has the First Lady visited on official business (i.e., accompanying the President or addressing audiences/attending events)?"
        }

        # Log the topic
        IcLogger.print_always(topic)

        corpus = [{'docno': 'DOC1', 'text': 'The sky is blue, actually very blue.'},
                  {'docno': 'DOC2',
                   'text': 'The sun is bright and blue in Washington D.C., New York city and other cities. New York citizens are over eight million.'}]

        # Pick a document from the corpus
        document = corpus[1]

        # Log the document
        IcLogger.print_always(document)

        tfidf_matcher = QueryDocMatcher(topic, document, corpus, logging=True)

        top_k_matching_words = tfidf_matcher.get_words_to_highlight()

        # Log the top-k matching_words
        IcLogger.print_always(top_k_matching_words)
