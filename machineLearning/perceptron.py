# package machineLearning

"""
    A class to represent a perceptron classifier for a given class price tag
"""
import random


class Perceptron:
    def __init__(self):
        """
        Constructor
        """
        self.weights = {}  # dict { feature : weight }

    def initialize_weights(self, restaurant):
        """
        Initialize self.weights with random values, all zero or all 1
        :param restaurant: A restaurant of type restaurant
        """
        for k in restaurant.features.keys():
            if k != "bias":
                self.weights[k] = random.uniform(
                    0, 1
                )  # initialize with random numbers
                # perceptron.weights[k] = 0 --> uncomment to initialize with 0.0
                # perceptron.weights[k] = 11 --> uncomment to initialize with 1.0
            if (
                    k == "bias"
            ):  # if no bias is given, this will just never be reached
                self.weights[k] = restaurant.features.get(k)

    def get_weight(self, feature):
        """
        Get the weight of a feature; if the feature is not in the dict, 0 is returned
        :param feature: A single feature
        :return: A float weight
        """
        if feature in self.weights.keys():
            return self.weights[feature]
        else:
            return 0.0

    def compute_score(self, features):
        """
        Get the score for a given feature set
        :param features: A dictionary feature set ( string feature : float value )
        :return: A float score
        """
        score = 0.0
        for feature in features.keys():
            score += features.get(feature) * self.get_weight(feature)
        return score

    def increase_feat_vector(self, features):
        """
        Increase feature vector for a given feature set
        :param features: A dictionary feature set ( string feature : float value )
        """
        for feature in features.keys():
            if feature in self.weights.keys():
                self.weights[feature] = self.weights.get(
                    feature
                ) + features.get(feature)
            else:
                self.weights[feature] = features.get(feature)

    def decrease_feat_vector(self, features):
        """
        Decrease feature vector for a given feature set
        :param features:  A dictionary feature set ( string feature : float value )
        """
        for feature in features.keys():
            if feature in self.weights.keys():
                self.weights[feature] = self.weights.get(
                    feature
                ) - features.get(feature)
            else:
                self.weights[feature] = features.get(feature)

# c = Corpus()
# c.read_gold_file("/home/PycharmProjects/Lab_V2/src/data/menu_dev.txt")
# feat = FeatureExtractor()
# for restaurant in c.restaurants:
#     feat.extract_features_bow(restaurant)
# p = Perceptron()
# for r in c.restaurants:
#     fs = r.features
#     for f in r.features:
#         p.get_weight(f, fs)
#     print(p.compute_score(fs))
