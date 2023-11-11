# package machineLearning

"""
    A Python module to predict a restaurant's price tag with a Linear Regressor

"""

import numpy
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# get data
from sklearn.metrics import confusion_matrix, classification_report

from src.preprocessing_ml.preprocessorExtensions import PreprocessorExtensions

pe = PreprocessorExtensions()
pe.get_data("/home/PycharmProjects/Lab_V2/src/data/*.txt")
pe.get_features(pe.all_data)
pe.convert_tags_and_data(pe.all_data)

# set up model and predict price tags
lr = LinearRegression()
lr.fit(pe.X_train, pe.y_train)

y_pred = lr.predict(pe.X_dev)

# print("Evaluation after prediction:\n")
# print(confusion_matrix(pe.y_dev, y_pred))
# print(classification_report(pe.y_dev, y_pred))

# print("Mean Absolute Error:", metrics.mean_absolute_error(pe.y_dev, y_pred))
# print("Mean Squared Error:", metrics.mean_squared_error(pe.y_dev, y_pred))
# print(
#    "Root Mean Squared Error:",
#    numpy.sqrt(metrics.mean_squared_error(pe.y_dev, y_pred)),
# )

# print predictions to predicated data files to enable evaluation via evaluation tool and main
f = open(
    "/home/PycharmProjects/Lab_V2/src/data/predicted_data/menu_dev_lr.txt",
    "w",
)
f.close()
f = open(
    "/home/PycharmProjects/Lab_V2/src/data/predicted_data/menu_dev_lr.txt",
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
