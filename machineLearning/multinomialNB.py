# package machineLearning

"""
    A Python module to predict a restaurant's price tag with a Multinomial NB Classifier
"""

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

from preprocessing_ml.preprocessorExtensions import PreprocessorExtensions

# get and preprocess data
pe = PreprocessorExtensions()
pe.get_data("/home/PycharmProjects/Lab_V2/src/data/*.txt")
pe.get_features(pe.all_data)
pe.convert_tags_and_data(pe.all_data)

# set up classifier and predict price tags
mnb = MultinomialNB()
mnb.fit(pe.X_train, pe.y_train)

# evaluate after last training iteration
y_pred_train = mnb.predict(pe.X_train)
print("Evaluation after training:\n")
print(classification_report(pe.y_train, y_pred_train))

y_pred = mnb.predict(pe.X_dev)

print("Evaluation after prediction:\n")
print(confusion_matrix(pe.y_dev, y_pred))
print(classification_report(pe.y_dev, y_pred))

# enable evaluation via main
f = open(
    "/home/PycharmProjects/Lab_V2/src/data/predicted_data/menu_dev_mnb.txt",
    "w",
)
f.close()
f = open(
    "/home/PycharmProjects/Lab_V2/src/data/predicted_data/menu_dev_mnb.txt",
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
