# package machineLearning

"""
    A Python module to predict a restaurant's price tag with a Multilayered Perceptron Classifier
"""
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

from src.preprocessing_ml.preprocessorExtensions import PreprocessorExtensions

pe = PreprocessorExtensions()
pe.get_data("src/data/*.txt")
pe.get_features(pe.all_data)
pe.convert_tags_and_data(pe.all_data)

# scaler = StandardScaler()
# scaler.fit_transform(X_train)
# X_dev = scaler.transform(X_dev)

# mm_scaler = MiniMaxScaler()
# X_train = mm_scaler.fit_transform(X_train)
# X_dev = mm_scaler.transform(X_dev)

# set up model
mlp = MLPClassifier(
    solver='adam',
    hidden_layer_sizes=(50, 50, 50),
    max_iter=150,
    verbose=True,
    random_state=0,
)

## Hyperparameter Tuning ##
# parameter_space = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100,100)],
#     'solver': ['sgd', 'adam'],
#     'random_state' : [0,50,100]
# }
# clf = GridSearchCV(mlp, parameter_space, cv=3)
# clf.fit(pe.X_train, pe.y_train)
#
# #Best parameter set
# print('Best parameters found:\n', clf.best_params_)
# {'solver': 'adam', 'random_state': 0, 'hidden_layer_sizes': (50, 50, 50)}

mlp.fit(pe.X_train, pe.y_train)

# evaluate after last training iteration
y_pred_train = mlp.predict(pe.X_train)
print("Evaluation after training:\n")
print(classification_report(pe.y_train, y_pred_train))

# enable evaluation via main
# f = open(
#     "src/data/predicted_data/menu_dev_mlp_train.txt",
#     "w",
# )
# f.close()
# f = open(
#     "src/data/predicted_data/menu_dev_mlp_train.txt",
#     "a+",
# )
# for pred in y_pred_train:
#     if pred < 1.5:
#         f.write("$" + "\r\n")
#     elif 1.5 <= pred <= 2.5:
#         f.write("$$" + "\r\n")
#     elif 2.5 <= pred <= 3.5:
#         f.write("$$$" + "\r\n")
#     elif pred >= 3.5:
#         f.write("$$$$" + "\r\n")
# f.close()

# predict price tag
y_pred = mlp.predict(pe.X_dev)

print("Evaluation after prediction:\n")
print(confusion_matrix(pe.y_dev, y_pred))
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
