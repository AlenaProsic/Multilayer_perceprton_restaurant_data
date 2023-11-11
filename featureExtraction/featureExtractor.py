# package featureExtraction

"""
    A class to extract various features from restaurants
"""

from collections import OrderedDict
import nltk
from nltk.corpus import stopwords
# import spacy
# from spacy_langdetect import LanguageDetector
# from spacy.lang.xx import MultiLanguage

from resources.corpus import Corpus


class FeatureExtractor(object):
    def __init__(self):
        """
        Null constructor
        """
        pass

    def extract_features_bow(self, restaurant):
        """
        The function extract Bag-of-Words features from restaurant.tokens
        :param restaur  ant: A restaurant of type restaurant
        """
        features = {"bias": 1.0}
        # features = {}  # uncomment to use w/o bias

        for token in restaurant.tokens:
            if token not in features:
                features[token] = 1.0
            else:
                features[token] = features[token] + 1.0
        restaurant.features = features
        # print(restaurant.features)

    ## FEATURE EXPLORATION ##

    def extract_features_word_length(self, restaurant):
        """
        Extracts word length by counting the number of characters of each token in a given restaurant's tokens.
        The underlying assumption here is one the one hand that the longer a word, the more complex it is and on the
        other hand that the more complex words a restaurant's menu contains the more likely it is to be expensive and vice versa.
        :param restaurant: A given restaurant of type restaurant.
        """
        # extract word length from restaurant.tokens
        for token in restaurant.tokens:
            word_length = float(len(token))
            # print(word_length)

            # add word length counts to restaurant_features
            if (token + "_word_length") not in restaurant.features:
                restaurant.features[(token + "_word_length")] = word_length

    def extract_features_average_dish_length(self, restaurant):
        """
        The feature extracts the average dish length from a restaurants dishes.
        The underlying assumption is that the longer the dish is the more likely the restaurant is going to be cheap
        (concepts of "plenty" and consumer choice), and vice versa.
        :param restaurant: A given restaurant of type restaurant.
        """
        # extracting dish length from restaurant.menu_text
        sum = 0
        num_dishes = 0  # len(restaurant.menu_text)
        average_dish_length = 0
        for dish in restaurant.menu_text:
            dish_length = float(len(dish))
            sum += dish_length
            num_dishes += 1

        # calculating the average dish length
        if num_dishes == len(restaurant.menu_text):
            average_dish_length = sum / len(restaurant.menu_text)

        # adding average dish length to restaurant.features
        # uncomment to add the same average dish lenght feature 20 times
        # for i in range (0, 20):
        #     if (restaurant.name + " " + restaurant.location + "_dish_length" + str(i)) not in restaurant.features:
        #         restaurant.features[
        #             (restaurant.name + " " + restaurant.location + "_dish_length" + str(i))] = average_dish_length
        if (restaurant.name + " " + restaurant.location + "_dish_length") not in restaurant.features:
            restaurant.features[(restaurant.name + " " + restaurant.location + "_dish_length")] = average_dish_length

    def extract_features_menu_length(self, restaurant):
        """
        The feature extracts the menu length, using restaurants.menu_text.
        The underlying assumption is that the longer menu, the more likely it is to be cheap (the concept of plenty
        and abundance of consumer choice), and vice versa.
        :param A restaurant of type restaurant
        """
        menu_length = 0
        for dish in restaurant.menu_text:
            dish_length = float(len(dish))
            menu_length += dish_length

        if ("menu_length_w_wspace") not in restaurant.features:
            restaurant.features[("menu_length_w_wspace")] = menu_length

    def extract_features_BOW_with_stop_word_list(self, restaurant):
        """
        The function extracts bag-of-words features coupled with a list of predefined stop words adopted from NLTK,
        using a restaurant.features BOW for most often occurring words. The stop word list includes function words,
        like e.g. prepositions, conjunctions ("and"), articles, auxilary verbs,  etc. The underlying concept is that
        function words won't contribute a lot to classification due to their high frequency.
        :param restaurant: A restaurant of type restaurant
        """
        stop_words = set(stopwords.words("english"))
        if "us" in stop_words:
            print("yes")
        word_tokens = restaurant.tokens
        filtered = [w for w in word_tokens if w not in stop_words]

        for token in filtered:
            if token not in restaurant.features:
                restaurant.features[token] = 1.0
            else:
                restaurant.features[token] = restaurant.features[token] + 1.0

    def extract_features_BOW_with_stop_word_list_handcrafted(
            self, restaurant, stopwords
    ):
        """
        The function extracts bag-of-word feature in combination with the handcrafted stop words list, which takes the
        top 100 most frequent words from a given training corpus. The underlying idea is to compare the
        performances with feature with the predefined stopwords list from NLTK as the given data are "only" dishes
        and no "normal" text in the sense of standard written English, as e.g. in newspaper, etc.
        :param restaurant: A restaurant of type restaurant
        :param stopwords: A list of strings of stopwords extracted from a given training corpus
        """
        filtered = [
            w for w in restaurant.tokens if w not in stopwords
        ]  # stopwords list: see preprocessorExtensions
        for token in filtered:
            if token not in restaurant.features:
                restaurant.features[token] = 1.0
            else:
                restaurant.features[token] = restaurant.features[token] + 1.0

    def extract_features_bigrams(self, restaurant):
        """
        The function extracts bigrams from restaurant.tokens.
        The underlying concept is that the frequently occuring bigrams might help to boost the performance of a
        certain class/classes.
        :param restaurant: A restaurant of type restaurant
        """
        # extract bigrams from restaurant.tokens in a list using nltk
        bigram_scikit = []
        text = restaurant.tokens
        bigrams = list(nltk.bigrams(restaurant.tokens))

        for bigram in bigrams:
            bigram_scikit.append("(" + bigram[0] + "," + bigram[1] + ")")

        # add extracted bigrams as features to restaurant.features
        for bigram in bigram_scikit:
            if bigram not in restaurant.features:
                restaurant.features[bigram] = 1.0
            else:
                restaurant.features[bigram] = restaurant.features[bigram] + 1.0

        # d = OrderedDict(restaurant.features)
        # print(d)

    def extract_adjectives_positive_sentiment(
            self, restaurant, adj_positive_list
    ):
        """
        Extracts adjectives accounting for positive sentiment from a given restaurant's tokens.
        The list of adjectives used was downloaded from: DOI: 10.31235/osf.io/j9tga.
        :param restaurant: A given restaurant of type restaurant.
        :param adj_positive_list: A list of strings
        """
        for adj_pos_sentiment in adj_positive_list:
            if "adj_pos_sentiment_" + adj_pos_sentiment not in restaurant.features:
                restaurant.features[
                    "adj_pos_sentiment_" + adj_pos_sentiment
                    ] = 1.0
            else:
                restaurant.features[
                    "adj_pos_sentiment_" + adj_pos_sentiment
                    ] = (
                        restaurant.features[
                            "adj_pos_sentiment_" + adj_pos_sentiment
                            ]
                        + 1.0
                )

    def extract_adjectives_sensory(self, restaurant, adj_sensory_list):
        """
        Extracts sensory adjectives from a given restaurant's tokens.
        The list of adjectives used was downloaded from: DOI: 10.31235/osf.io/j9tga.
        :param restaurant: A given restaurant of type restaurant
        :param adj_sensory_list: A list of strings
        """
        for adj_sens in adj_sensory_list:
            if "adj_sens_" + adj_sens not in restaurant.features:
                restaurant.features["adj_sens_" + adj_sens] = 1.0
            else:
                restaurant.features["adj_sens_" + adj_sens] = (restaurant.features["adj_sens_" + adj_sens] + 1.0)

    def extract_language(self, restaurant, lang_extracted):
        """
        Extracts the language of a given dish using the precomputed language_extracted_dev.txt and
        language_extracted_train.txt for time efficiency (retrieved with spacy-langdetect, see uncommented
        code and preprocessorExtensions)

        :param restaurant: A restaurant of type restaurant
        :param lang_extracted: A list of strings indicating the language of the dishes on a given restaurant's menu
        """
        # f = open(
        #    "/homePycharmProjects/Lab_V2/src/featureExtraction/language_extracted.txt",
        #    "a+",
        # )
        # f.write(str(restaurant.restaurant_id) + "\t")
        # for dish in restaurant.menu_text:
        #    language = ""
        #    text = dish #restaurant.menu_text_as_whole
        #    doc = nlp(text)
        #    for i, sent in enumerate(doc.sents):
        #        language = sent._.language['language']
        #    f.write(language + "\t")
        for language in lang_extracted:
            if language not in restaurant.features:
                restaurant.features[language] = 1.0
            else:
                restaurant.features[language] = restaurant.features[language] + 1.0
        # f.write("\n")
        # f.close()

# feat = FeatureExtractor()
# c = Corpus()
# c.read_gold_file("/home/PycharmProjects/Lab_V2/src/data/menu_dev.txt")
# for restaurant in c.restaurants:
# feat.extract_language(restaurant, nlp)
# feat.extract_features_bow(restaurant)
# feat.extract_features_word_length(restaurant)
# feat.extract_features_menu_length(restaurant)
# feat.extract_features_average_dish_length(restaurant)
# feat.extract_features_stop_word_list(restaurant)
# feat.extract_features_BOW_with_stop_word_list_handcrafted(restaurant)
# feat.extract_features_bigrams(restaurant)
# print(restaurant.features)
