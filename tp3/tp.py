import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

def read_data(path: str, y_label: str, remove_columns: list[str]=[]):
    """Return X and y."""
    data = pd.read_csv(path)

    for colname in remove_columns:
        data.pop(colname)

    X = data
    y = data.pop(y_label)
    return X, y

def build_classifier(X, y, criterion='entropy', **kwargs) -> tree.DecisionTreeClassifier:
    classifier = tree.DecisionTreeClassifier(criterion=criterion, **kwargs)
    classifier.fit(X, y)

    return classifier

def build_regressor(X, y, **kwargs) -> tree.DecisionTreeRegressor:
    regressor = tree.DecisionTreeRegressor(**kwargs)
    regressor.fit(X, y)

    return regressor

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

def barbecue():
    X_train, y_train = read_data('barbecue.csv', 'barbecue')
    classifier = build_classifier(X_train, y_train)

    graph_tree(classifier, 'tree.dot', ['Meteo','Amis','Vent','Jour'])

def glass():
    X, y = read_data('glass.data', 'Type', ['Id'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    classifier = build_classifier(X_train, y_train, 'entropy', min_impurity_decrease=0.03)

    graph_tree(classifier, 'tree.dot', ['Indice réfraction', 'Na', 'Mg', 'Al', 'Si', 'K', 'C', 'Br', 'Fe'])
    evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

def redwine():
    X, y = read_data('winequality-red.csv', 'quality')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    regressor = build_regressor(X_train, y_train, min_impurity_decrease=0.03)

    graph_tree(regressor, 'tree.dot', ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
    evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def whitewine():
    X, y = read_data('winequality-white.csv', 'quality')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    regressor = build_regressor(X_train, y_train, min_impurity_decrease=0.03)

    graph_tree(regressor, 'tree.dot', ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])
    evaluate_regressor(regressor, X_train, y_train, X_test, y_test)


def main():
    #barbecue()
    #glass()
    #redwine()
    whitewine()

if __name__ == '__main__':
    main()
