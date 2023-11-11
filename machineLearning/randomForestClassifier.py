# package machineLearning
"""
    A Python module to predict a restaurant's price tag with a Random Forest Classifier

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing_ml.preprocessorExtensions import PreprocessorExtensions

# get and preprocess data
pe = PreprocessorExtensions()
pe.get_data("/home/PycharmProjects/Lab_V2/src/data/*.txt")
pe.get_features(pe.all_data)
pe.convert_tags_and_data(pe.all_data)

## Hyperparamater Tuning ##
# Pick best random state: 115
# for i in range (0,500):
#     clf = RandomForestClassifier(250, random_state=i)
#     clf.fit(X_train, y_train)
#     #print(clf.feature_importances_)
#     y_pred = clf.predict(X_dev)

# train and predict
clf = RandomForestClassifier(250, random_state=115)
clf.fit(pe.X_train, pe.y_train)

# evaluate after last training iteration
y_pred_train = clf.predict(pe.X_train)
print("Evaluation after training:\n")
print(classification_report(pe.y_train, y_pred_train))

y_pred = clf.predict(pe.X_dev)

print("Evaluation after prediction:\n")
print(confusion_matrix(pe.y_dev, y_pred))
print(classification_report(pe.y_dev, y_pred))

# print predictions to predicted data files to enable an evaluation via evaluation tool and main

# uncomment to test various random states in different ranges
# f = open(
#     "src/data/predicted_data/menu_dev_rfc" + str(i) + ".txt",
#     "w",
# )
# f.close()
# f = open(
#     "src/data/predicted_data/menu_dev_rfc" + str(i) + ".txt",
#     "a+",
# )

f = open("/home/PycharmProjects/Lab_V2/src/data/predicted_data/menu_dev_rfc.txt", "w")
f.close()
f = open("/home/PycharmProjects/Lab_V2/src/data/predicted_data/menu_dev_rfc.txt", "a+")
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
