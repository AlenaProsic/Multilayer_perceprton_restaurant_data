# package machineLearning

"""
    A class to predict the price tag of restaurants
"""
import random

from evaluation.evaluation_process import EvaluationProcess
from resources.corpus import Corpus
from machineLearning.perceptron import Perceptron


class MultiClassPerceptron:
    def __init__(self):
        """
        Constructor
        """
        self.perceptrons = {}  # a dict { price_tag : perceptron }
        for price_tag in Corpus.price_tag_set:
            self.perceptrons[price_tag] = Perceptron()

    def train(self, corpus, epochs):
        """
        Train on a corpus with an even number of epochs (iterations)
        :param corpus: A corpus of type corpus
        :param epochs: A given number of iterations
        """
        eval_process = EvaluationProcess()
        eval_results = ""
        restaurants = corpus.restaurants
        random.seed(0)
        random.shuffle(restaurants)

        # initialize the weight vectors for every binary perceptron before training
        for price_tag in self.perceptrons:
            for restaurant in restaurants:
                perceptron = self.perceptrons.get(price_tag)
                perceptron.initialize_weights(restaurant)

        for i in range(0, epochs):
            # go over every restaurant
            for rest in restaurants:
                self.predict(rest)
                self.update_perceptron(rest)

            # evaluate after current iteration
            for price_tag in Corpus.price_tag_set:
                eval_results += (
                        price_tag
                        + ": "
                        + format(
                    (
                        eval_process.compute_fscore(
                            corpus.restaurants, price_tag
                        )
                    ),
                    ".5f",
                )
                        + "\n"
                )
            eval_results += (
                    "Macro F-Score: "
                    + format(
                (eval_process.compute_fscore_macro(corpus.restaurants)),
                ".5f",
            )
                    + "\n"
            )
            eval_results += (
                    "Micro F-Score: "
                    + format(
                (eval_process.compute_fscore_micro(corpus.restaurants)),
                ".5f",
            )
                    + "\n\n"
            )
        print(eval_results)

    def predict(self, restaurant):
        """
        Price tag prediction for one restaurant
        :param restaurant: An object of type restaurant
        """
        max_score = 0.0
        predicted = "NONE"

        for price_tag in self.perceptrons.keys():
            perceptron = self.perceptrons.get(price_tag)
            score = perceptron.compute_score(restaurant.features)

            # look for argMax for given price tag and set restaurant.predicted to corresponding tag
            if score > max_score:
                max_score = score
                predicted = price_tag

        restaurant.predicted = predicted

    def update_perceptron(self, restaurant):
        """
        Update feature vectors of perceptrons for predicted and gold price tags
        :param restaurant: A restaurant of type restaurant
        """
        if restaurant.predicted != restaurant.gold:
            # decrease wrongly classifying features
            if restaurant.predicted != "NONE":
                self.perceptrons.get(
                    restaurant.predicted
                ).decrease_feat_vector(restaurant.features)

            # increase correctly classifying features
            self.perceptrons.get(restaurant.gold).increase_feat_vector(
                restaurant.features
            )

# c = Corpus()
# c.read_gold_file("/home/PycharmProjects/Lab_V2/src/data/menu_train.txt")
# feat = FeatureExtractor()
# for rest in c.restaurants:
#     feat.extract_features_bow(rest)
# m = MultiClassPerceptron()
# m.train(c, 100)
