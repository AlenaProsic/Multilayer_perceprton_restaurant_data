# package preprocessing_ml

"""
    A class to preprocess data for the baseline implementation
"""

from featureExtraction.featureExtractor import FeatureExtractor
from resources.corpus import Corpus


class PreprocessorBaseline:
    def __init__(self):
        """
        Null Constructor
        """
        self.train_corpus = None
        self.dev_corpus = None

    def get_train_corpus(self, filename):
        """
        Load available data (training only)
        :param filename: A String filename
        """
        self.train_corpus = Corpus()
        self.train_corpus.read_gold_file(filename)

    def get_dev_corpus(self, filename):
        """
        Load available data (dev only)
        :param filename:
        """
        self.dev_corpus = Corpus()
        self.dev_corpus.read_gold_file(filename)

    def get_features(self, corpus):
        """
        Extract BOW feature from a given corpus of type corpus
        :param corpus: A corpus corpus
        """
        feat = FeatureExtractor()
        for rest in corpus.restaurants:
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
                feat.extract_features_menu_length(rest)  # 6
                # feat.extract_features_bigrams(rest)  # 7
                # feat.extract_adjectives_sensory(rest, adjectives_sensory)  # 8
                # feat.extract_adjectives_positive_sentiment(
                #    rest, adjectives_positive  # 9
                # )
                # nlp = spacy.load('xx_ent_wiki_sm')
                # sentencizer = nlp.create_pipe('sentencizer')
                # nlp.add_pipe(sentencizer)
                # nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
                # if corpus.restaurants.index(rest) < 607:
                #    feat.extract_language(rest, language_dict[str(rest.restaurant_id) + "_dev"])
                # if corpus.restaurants.index(rest) >= 607:
                #    feat.extract_language(rest, language_dict[(str(rest.restaurant_id) + "_train")])
