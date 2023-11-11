# package preprocessing_ml

"""
    A class to preprocess data for classifiers provided by scikit learn.
# """
import glob
import numpy as np
import scipy
# import spacy
# from spacy_langdetect import LanguageDetector
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
# from spacy.lang.xx import MultiLanguage

from featureExtraction.featureExtractor import FeatureExtractor
from resources.corpus import Corpus


class PreprocessorExtensions:
    def __init__(self):
        """
        Null Constructor
        """
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.X = None
        self.all_data = Corpus()
        self.all_tags = []
        self.feature_list = []

    def get_data(self, filenames):
        """
        Load available data (training and data)
        :param filenames: A string filenames
        """
        gold_files = (
            filenames
        )  # "src/data/*.txt"
        predicted_files = glob.glob(gold_files)
        predicted_files_sorted = sorted(predicted_files)
        for file in predicted_files_sorted:
            self.all_data.read_gold_file(file)

    def get_features(self, corpus):
        """
        Extract feature(s) combinations of a given corpus
        :param corpus: A corpus of type corpus
        """
        # Uncomment to extract a new stopwords list
        # Make sure to uncomment the necessary import statements as well (spacy, spacy-langdetect)
        # dict_stop_words = {}
        # threshold = 10
        # for restaurant in c.restaurants:
        #     self.extract_features_bow(restaurant)
        #     for k, v in restaurant.features.items():
        #         if v >= threshold:
        #             if k not in dict_stop_words:
        #                 dict_stop_words[k] = v
        #             else:
        #                 dict_stop_words[k] = dict_stop_words[k] + v
        #
        # # restaurant.features = {}
        # stopwords_sorted = sorted(dict_stop_words.items(), key=lambda x: x[1], reverse=True)
        # stopwords_sorted_top100 = stopwords_sorted[:100]
        # stopwords_top100 = [i[0] for i in stopwords_sorted_top100]
        # f = open(
        #     "src/featureExtraction/stopwords_handcrafted_new_stopword_list.txt","a+")
        # for s in stopwords_top100:
        #     f.write(s + "\r\n")
        # f.close()
        stopwords = [
            line.rstrip("\n")
            for line in open(
                "src/featureExtraction/stopwords_handcrafted_train.txt"
            )
        ]
        adjectives_positive = [
            line.rstrip("\n")
            for line in open(
                "src/featureExtraction/adjective_positive_sentiment.txt"
            )
        ]
        adjectives_sensory = [
            line.rstrip("\n")
            for line in open(
                "src/featureExtraction/adjective_sensory.txt"
            )
        ]
        # load precomputed detected languages on dish level from files into one dictionary
        with open("src/featureExtraction/lang_extracted_dev.txt") as file:
            rows = (line.rstrip().split('\t') for line in file)
            language_dict_dev = {(str(row[0]) + "_dev"): row[1:] for row in rows}
        with open("src/featureExtraction/lang_extracted_train.txt") as file:
            rows = (line.rstrip().split('\t') for line in file)
            language_dict_train = {(str(row[0]) + "_train"): row[1:] for row in rows}
        language_dict = {**language_dict_dev, **language_dict_train}

        feat = FeatureExtractor()  # (un)comment whichever feature combination wished
        for rest in corpus.restaurants:
            feat.extract_features_bow(rest)  # 1
            # feat.extract_features_BOW_with_stop_word_list(rest)  # 2
            # feat.extract_features_BOW_with_stop_word_list_handcrafted(rest, stopwords) #3
            feat.extract_features_word_length(rest)  # 4
            feat.extract_features_average_dish_length(rest)  # 5
            # feat.extract_features_menu_length(rest)  # 6
            # feat.extract_features_bigrams(rest)  # 7
            # feat.extract_adjectives_sensory(rest, adjectives_sensory)  # 8
            # feat.extract_adjectives_positive_sentiment(
            #    rest, adjectives_positive  # 9
            # )
            # nlp = spacy.load('xx_ent_wiki_sm')
            # sentencizer = nlp.create_pipe('sentencizer')
            # nlp.add_pipe(sentencizer)
            # nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
            if corpus.restaurants.index(rest) < 607:
                feat.extract_language(rest, language_dict[str(rest.restaurant_id) + "_dev"])
            if corpus.restaurants.index(rest) >= 607:
                feat.extract_language(rest, language_dict[(str(rest.restaurant_id) + "_train")])

    def convert_tags_and_data(self, corpus):
        """
        Convert tags and data into necessary format (ordinal numbers, training and development corpus)
        :param corpus: A corpus of type corpus
        """
        for restaurant in corpus.restaurants:
            self.feature_list.append(restaurant.features)
            # convert price tags into real values ($ - 1.0, etc.)
            if restaurant.gold == "$":
                self.all_tags.append(1.0)
            elif restaurant.gold == "$$":
                self.all_tags.append(2.0)
            elif restaurant.gold == "$$$":
                self.all_tags.append(3.0)
            elif restaurant.gold == "$$$$":
                self.all_tags.append(4.0)

        # fit and transform extracted features
        vec = DictVectorizer()
        self.X = vec.fit_transform(self.feature_list).toarray()

        # get training and development data as sparse matrices
        self.X_train = scipy.sparse.csr_matrix(self.X[607:])
        self.X_dev = scipy.sparse.csr_matrix(self.X[:607])
        self.y_train = np.asarray(self.all_tags[607:])
        self.y_dev = np.asarray(self.all_tags[:607])
