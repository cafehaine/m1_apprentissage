import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

def read_data(path: str, y_label: str, remove_columns: list[str]=[], encode_columns: list[str]=[]):
    """Return X and y."""
    data = pd.read_csv(path)

    for colname in remove_columns:
        data.pop(colname)

    for colname in encode_columns:
        encode_column(data, colname)

    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()

    X = data
    y = data.pop(y_label)
    return X, y

def encode_column(data, colname):
    le = LabelEncoder()
    unique_values = list(data[colname].unique())
    le_fitted = le.fit(unique_values)
    values = list(data[colname].values)
    values_transformed = le.transform(values)
    data[colname] = values_transformed

def build_classifier(X, y, **kwargs) -> MLPClassifier:
    classifier = MLPClassifier(**kwargs)
    classifier.fit(X, y)

    return classifier

#def build_regressor(X, y, **kwargs) -> tree.DecisionTreeRegressor:
#    regressor = tree.DecisionTreeRegressor(**kwargs)
#    regressor.fit(X, y)
#
#    return regressor

def graph_tree(model, path: str, features: list[str]):
    tree.export_graphviz(model, out_file=path, feature_names=features)


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    print("Accuracy train:", classifier.score(X_train, y_train))
    print("Accuracy test:", classifier.score(X_test, y_test))
    y_pred = classifier.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_regressor(regressor, X_train, y_train, X_test, y_test):
    print("Score train:", regressor.score(X_train, y_train))
    print("Score test:", regressor.score(X_test, y_test))
    y_pred = regressor.predict(X_test)
    print("Mean squared error:", mean_squared_error(y_test, y_pred))
    print("Mean absolute error:", mean_absolute_error(y_test, y_pred))
    print("r2 score:", r2_score(y_test, y_pred))

def human_resources():
    X, y = read_data('human_resources.csv', 'left', encode_columns=["sales", "salary"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #classifier = build_classifier(X_train, y_train)
    classifier = build_classifier(X_train, y_train)

    # graph_tree(classifier, 'tree.dot', ['Meteo','Amis','Vent','Jour'])

    evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

def main():
    human_resources()

if __name__ == '__main__':
    main()
