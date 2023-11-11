# package evaluation

"""
    Preparation class for evaluation to store TP, FP, and FN for a given restaurant
"""


class EvalPrep:
    def __init__(self, true_positive=0, false_positive=0, false_negative=0):
        """
        Constructor
        :param true_positive: An int true positive
        :param false_positive: An int false positive
        :param false_negative: An int false negative
        """
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.false_negative = false_negative

    def __str__(self):
        """
        String representation
        :return: A String representation of an instance of this class
        """
        return (
                "EvalPrep [truePositive = "
                + str(self.true_positive)
                + ", falsePositive = "
                + str(self.false_positive)
                + ", falseNegative = "
                + str(self.false_negative)
                + "]"
        )
