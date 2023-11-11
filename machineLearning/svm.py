# package machineLearning

"""
    A Python module to predict a restaurant's price tag with a SVM Classifier
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing_ml.preprocessorExtensions import PreprocessorExtensions

# get and preprocess data
pe = PreprocessorExtensions()
pe.get_data("/home/PycharmProjects/Lab_V2/src/data/*.txt")
pe.get_features(pe.all_data)
pe.convert_tags_and_data(pe.all_data)

# scaler = StandardScaler()
# scaler.fit_transform(pe.X_train)
# X_dev = scaler.transform(pe.X_dev)

# set up classifier and predict price tag on dev set
# Hyperparameter optimization:
# SGD is empirically found to converge after ovserving approx. 10^6 training samples
# (https://scikit-learn.org/stable/modules/sgd.html, 2019-07-26)
svm = SVC(gamma="auto", random_state=0, max_iter=np.ceil(10 ** 6 / 4860))
svm.fit(pe.X_train, pe.y_train)

# evaluate after last training iteration
y_pred_train = svm.predict(pe.X_train)
print("Evaluation after training:\n")
print(classification_report(pe.y_train, y_pred_train))

y_pred = svm.predict(pe.X_dev)

print("Evaluation after prediction:\n")
print(confusion_matrix(pe.y_dev, y_pred))
print(classification_report(pe.y_dev, y_pred))

# enable evaluation via main
f = open(
    "/home/PycharmProjects/Lab_V2/src/data/predicted_data/svm.txt",
    "w",
)
f.close()
f = open(
    "/home/PycharmProjects/Lab_V2/src/data/predicted_data/svm.txt",
    "a+",
)
for pred in y_pred:
    if pred < 1.5:
        f.write("$" + "\r\n")
    elif 1.5 <= pred <= 2.5:
        f.write("$$" + "\r\n")
    elif 2.5 <= pred <= 3.5:
        f.write("$$$" + "\r\n")
    elif pred >= 3.5:
        f.write("$$$$" + "\r\n")
f.close()
