# package src

"""
    A main to run and evaluate a classifier with a chosen set of features.
    The default set feature set is a combination of a simple BOW, a menu length and a average dish length feature.
    The classifier reports partial computations which can be changed setting the classifier parameter "verbose" to
    False.
"""

import glob
import pickle

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from evaluation.evaluation_process import EvaluationProcess
from machineLearning.multiclassPerceptron import MultiClassPerceptron
from preprocessing_ml.preprocessorBaseline import PreprocessorBaseline
from preprocessing_ml.preprocessorExtensions import PreprocessorExtensions

if __name__ == "__main__":
    ### Uncomment to use baseline classifier ###

    # To chose a feature combination, go to src/preprocessing_ml/preprocessorBaseline and (un)comment the features
    # wished
    # p_bl = PreprocessorBaseline()
    # p_bl.get_train_corpus(
    #     "src/data/menu_train.txt"
    # )
    # load training and development corpora for baseline results
    # p_bl.get_dev_corpus(
    #     "src/data/menu_test.txt"
    # )
    #
    # # # extract features from training corpus
    # # # to choose feature combinations go to module preprocessorBaseline
    # p_bl.get_features(p_bl.train_corpus)
    # #
    # # train multiclass perceptron (baseline) on training corpus
    # ml = MultiClassPerceptron()
    # ml.train(p_bl.train_corpus, 25)
    #
    # # uncomment to save the trained perceptron (trained on baseline (f1 only))
    # # pickle_out = open("perceptron_pickle_1", "wb")
    # # pickle.dump(ml, pickle_out)
    # # pickle_in = open("perceptron_pickle_1", "rb")
    # # m_loaded = pickle.load(pickle_in)
    #
    # # # extract features from development corpus (same feature combination as for training corpus)
    # p_bl.get_features(p_bl.dev_corpus)
    # #
    # # apply the trained classifier to restaurants in p_bl.dev_corpus
    # # change filename accordingly to feature combination used e.g. f1_f2_f3
    # f = open(
    #     "src/data/predicted_data/bl.txt",
    #     "w",
    # )
    # f.close()
    # f = open(
    #     "src/data/predicted_data/bl.txt",
    #     "a+",
    # )
    #
    # for rest in p_bl.dev_corpus.restaurants:
    #     ml.predict(rest)
    #     # store predicted tags in a file
    #     f.write(rest.predicted + "\r\n")
    # f.close()

    ### BEST ADVANCED CLASSIFIERS: 1. MLP, 2.SVM, 3.RFC

    # Uncomment for every advanced classifier #

    # To chose a feature combination, go to src/preprocessing_ml/preprocessorExtensions
    # and (un)comment the features wished
    pe = PreprocessorExtensions()
    pe.get_data("src/data/*.txt")
    pe.get_features(pe.all_data)
    pe.convert_tags_and_data(pe.all_data)

    # set up model
    mlp = MLPClassifier(
        solver='adam',
        hidden_layer_sizes=(50, 50, 50),
        max_iter=150,
        verbose=True,
        random_state=0,
    )
    mlp.fit(pe.X_train, pe.y_train)

    # evaluate after last training iteration
    y_pred_train = mlp.predict(pe.X_train)
    print("Evaluation after training:\n")
    print(classification_report(pe.y_train, y_pred_train))

    # predict price tag
    y_pred = mlp.predict(pe.X_dev)
    print("Confusion matrix:\n")
    print(confusion_matrix(pe.y_dev, y_pred))
    print("Evaluation after prediction:\n")
    print(
        classification_report(y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4])
    )
    print(
        metrics.precision_recall_fscore_support(
            y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4], average="micro"
        )
    )
    print(
        metrics.precision_recall_fscore_support(
            y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4], average="macro"
        )
    )
    # # enable evaluation via main
    f = open(
        "src/data/predicted_data/menu_dev_mlp.txt",
        "w",
    )
    f.close()
    f = open(
        "src/data/predicted_data/menu_dev_mlp.txt",
        "a+",
    )
    for pred in y_pred_train:
        if pred < 1.5:
            f.write("$" + "\r\n")
        elif 1.5 <= pred <= 2.5:
            f.write("$$" + "\r\n")
        elif 2.5 <= pred <= 3.5:
            f.write("$$$" + "\r\n")
        elif pred >= 3.5:
            f.write("$$$$" + "\r\n")
    f.close()

    ### uncomment to use RF Classifier ###
    # clf = RandomForestClassifier(250, random_state=115)
    # clf.fit(pe.X_train, pe.y_train)
    # y_pred = clf.predict(pe.X_dev)
    #
    # print(confusion_matrix(pe.y_dev, y_pred))
    # print(
    #     classification_report(y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4])
    # )
    # print(
    #     metrics.precision_recall_fscore_support(
    #         y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4], average="micro"
    #     )
    # )
    # print(
    #     metrics.precision_recall_fscore_support(
    #         y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4], average="macro"
    #     )
    # )
    # f = open(
    #     "src/data/predicted_data/menu_dev_rfc.txt",
    #     "w",
    # )
    # f.close()
    # f = open(
    #     "src/data/predicted_data/menu_dev_rfc.txt",
    #     "a+",
    # )
    # # convert real-valued tags back into String price tags
    # for pred in y_pred:
    #     if pred < 1.5:
    #         f.write("$" + "\r\n")
    #     elif 1.5 <= pred <= 2.5:
    #         f.write("$$" + "\r\n")
    #     elif 2.5 <= pred <= 3.5:
    #         f.write("$$$" + "\r\n")
    #     elif pred >= 3.5:
    #         f.write("$$$$" + "\r\n")
    # f.close()

    ### Uncomment to use multinomial NB classifier ###
    # set up classifier and predict price tags
    # mnb = MultinomialNB()
    # mnb.fit(pe.X_train, pe.y_train)
    #
    # # evaluate after last training iteration
    # y_pred_train = mnb.predict(pe.X_train)
    # print("Evaluation after training:\n")
    # print(classification_report(pe.y_train, y_pred_train))
    #
    # y_pred = mnb.predict(pe.X_dev)
    #
    # print(confusion_matrix(pe.y_dev, y_pred))
    # print(
    #     classification_report(y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4])
    # )
    # print(
    #     metrics.precision_recall_fscore_support(
    #         y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4], average="micro"
    #     )
    # )
    # print(
    #     metrics.precision_recall_fscore_support(
    #         y_true=pe.y_dev, y_pred=y_pred, labels=[1, 2, 3, 4], average="macro"
    #     )
    # )
    # print("Evaluation after prediction:\n")
    # print(confusion_matrix(pe.y_dev, y_pred))
    # print(classification_report(pe.y_dev, y_pred))
    #
    # f = open(
    #     "src/data/predicted_data/menu_dev_mnb.txt",
    #     "w",
    # )
    # f.close()
    # f = open(
    #     "src/data/predicted_data/menu_dev_mnb.txt",
    #     "a+",
    # )
    # # convert real-valued tags back into String price tags
    # for pred in y_pred:
    #     if pred < 1.5:
    #         f.write("$" + "\r\n")
    #     elif 1.5 <= pred <= 2.5:
    #         f.write("$$" + "\r\n")
    #     elif 2.5 <= pred <= 3.5:
    #         f.write("$$$" + "\r\n")
    #     elif pred >= 3.5:
    #         f.write("$$$$" + "\r\n")
    # f.close()

    ### Uncomment to use baseline classifier ###
    # read in dev predicted file(s)
    # predicted_data_path = (
    #    "src/data/predicted_data/*.txt"
    # )
    # predicted_files_unsorted = glob.glob(predicted_data_path)
    # predicted_files = sorted(predicted_files_unsorted, key=str.lower)
    #
    # # do evaluation for every predicted file in predicted_files
    # eval_process = EvaluationProcess()
    # for file in predicted_files:
    #
    #     p_bl.dev_corpus.read_predicted_file(file)
    #     eval_process.eval_metrics[eval_process.num_experiments] = [
    #         eval_process.compute_fscore_micro(p_bl.dev_corpus.restaurants),
    #         eval_process.compute_fscore_macro(p_bl.dev_corpus.restaurants),
    #     ]
    #
    #     # uncomment to get more detailed results
    #
    #     ### overall micro precision and recall ###
    #     # print(eval_process.compute_precision_macro(dev))
    #     # print(eval_process.compute_recall_macro(dev))
    #     # print(eval_process.compute_precision_micro(dev))
    #     # print(eval_process.compute_recall_micro(dev))
    #
    #     ### precision and recall price tag wise ###
    #     print("$:\t" + str(eval_process.compute_fscore(p_bl.dev_corpus.restaurants, '$')))
    #     print("$$:\t" + str(eval_process.compute_fscore(p_bl.dev_corpus.restaurants, '$$')))
    #     print("$$$:\t" + str(eval_process.compute_fscore(p_bl.dev_corpus.restaurants, '$$$$')))
    #     print("$$$$:\t" + str(eval_process.compute_fscore(p_bl.dev_corpus.restaurants, '$$$$')))
    #     # print("$:\t" + str(eval_process.compute_recall(p_bl.dev_corpus.restaurants, '$')))
    #     # print("$$:\t" + str(eval_process.compute_recall(p_bl.dev_corpus.restaurants, '$$')))
    #     # print("$$$:\t" + str(eval_process.compute_recall(p_bl.dev_corpus.restaurants, '$$$$')))
    #     # print("$$$$:\t" + str(eval_process.compute_recall(p_bl.dev_corpus.restaurants, '$$$$')))
    #
    #     # print results
    #     #if eval_process.num_experiments == len(predicted_files):
    #     for k, v in eval_process.eval_metrics.items():
    #             print(
    #                 "Experiment No "
    #                 + str(k)
    #                 + ":\tF-Score Micro: "
    #                 + format(v[0], ".5f")
    #                 + "\tF-Score Macro: "
    #                 + format(v[1], ".5f")
    #             )
    #     eval_process.num_experiments += 1
