# package evaluation

"""
    Class for computing TP, FP, FN, precision, recall, a balanced F-Score for a given price tag as well as micro-
    and macro-average f-scores
    evaluation
"""
import collections

from evaluation.eval_prep import EvalPrep
from resources.corpus import Corpus


class EvaluationProcess:
    def __init__(self):
        """
        Constructor
        """
        self.num_experiments = (
            1
        )  # evaluate at least one file with predicted values
        self.eval_metrics = collections.OrderedDict()

    @staticmethod
    def tag_metrics(corpus, price_tag):
        """
        Compute TP, FP, FN for a given price tag
        :param corpus: A valid corpus of type corpus
        :param price_tag: A string price tag int range of $ - $$$$
        :return: An object of type eval_prep
        """
        true_pos = list(
            filter(
                lambda r: r.gold == price_tag and r.predicted == price_tag,
                corpus,
            )
        )
        false_pos = list(
            filter(
                lambda r: r.gold != price_tag and r.predicted == price_tag,
                corpus,
            )
        )
        false_neg = list(
            filter(
                lambda r: r.gold == price_tag and r.predicted != price_tag,
                corpus,
            )
        )

        return EvalPrep(len(true_pos), len(false_pos), len(false_neg))

    def compute_precision(self, corpus, price_tag):
        """
        Compute precision for a given tag
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :param price_tag: A price tag
        :return: A float precision
        """
        eval_prep = self.tag_metrics(corpus, price_tag)
        denominator = eval_prep.true_positive + eval_prep.false_positive
        if denominator == 0:
            return 0
        return float(eval_prep.true_positive / denominator)

    def compute_recall(self, corpus, price_tag):
        """
        Compute recall for a given tag
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :param price_tag: A price tag
        :return: A float recall
        """
        eval_prep = self.tag_metrics(corpus, price_tag)
        denominator = eval_prep.true_positive + eval_prep.false_negative
        if denominator == 0:
            return 0
        return float(eval_prep.true_positive / denominator)

    def compute_fscore(self, corpus, price_tag):
        """
        Compute balanced F-Score for a given tag
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :param price_tag: A price tag
        :return: A float f_score
        """
        precision = self.compute_precision(corpus, price_tag)
        recall = self.compute_recall(corpus, price_tag)
        if precision == 0.0 or recall == 0.0:
            return 0.0
        return float((2 * (precision * recall)) / (precision + recall))

    def compute_precision_micro(self, corpus):
        """
        Compute overall micro precision
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :return: A float micro_precision
        """
        numerator = 0
        denominator = 0
        for price_tag in Corpus.price_tag_set:
            eval_prep = self.tag_metrics(corpus, price_tag)
            numerator += eval_prep.true_positive
            denominator += float(
                eval_prep.true_positive + eval_prep.false_positive
            )
        if denominator == 0:
            return 0
        return float(numerator / denominator)

    def compute_recall_micro(self, corpus):
        """
        Compute overall micro recall
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :return: A float micro_recall
        """
        numerator = 0
        denominator = 0
        for price_tag in Corpus.price_tag_set:
            eval_prep = self.tag_metrics(corpus, price_tag)
            numerator += eval_prep.true_positive
            denominator += float(
                eval_prep.true_positive + eval_prep.false_negative
            )
        if denominator == 0:
            return 0
        return float(numerator / denominator)

    def compute_fscore_micro(self, corpus):
        """
        Compute micro F-Score
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :return: A float micro f_score
        """
        precision_micro = self.compute_precision_micro(corpus)
        recall_micro = self.compute_recall_micro(corpus)
        numerator = 2 * (precision_micro * recall_micro)
        denominator = precision_micro + recall_micro
        if denominator == 0:
            return 0
        return float(numerator / denominator)

    def compute_precision_macro(self, corpus):
        """
        Compute overall macro precision
        :param corpus:A corpus of type corpus with an attribute restaurants containing restaurant objects
        :return: A float macro_precision
        """
        numerator = 0
        for price_tag in Corpus.price_tag_set:
            numerator += self.compute_precision(corpus, price_tag)
        return float(numerator / len(Corpus.price_tag_set))

    def compute_recall_macro(self, corpus):
        """
        Compute overall macro recall
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :return: A float macro_recall
        """
        numerator = 0
        for price_tag in Corpus.price_tag_set:
            numerator += self.compute_recall(corpus, price_tag)
        return float(numerator / len(Corpus.price_tag_set))

    def compute_fscore_macro(self, corpus):
        """
        Compute macro f-score
        :param corpus: A corpus of type corpus with an attribute restaurants containing restaurant objects
        :return: A float macro f_score
        """
        precision_macro = self.compute_precision_macro(corpus)
        recall_macro = self.compute_recall_macro(corpus)
        numerator = 2 * (precision_macro * recall_macro)
        denominator = precision_macro + recall_macro
        if denominator == 0:
            return 0
        return float(numerator / denominator)

# eval_process = EvaluationProcess()
# c = Corpus()
# c.read_gold_file(
#     "src/data/menu_dev.txt"
# )
# c.read_predicted_file("src/data/predicted_data/menu_dev-predicted.txt")
# print(c)
# eval_process.compute_fscore(c, '$')
