import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

def read_data(path: str, y_label: str, remove_columns: list[str]=[], encode_columns: list[str]=[], show_correlation=False):
    """Return X and y."""
    data = pd.read_csv(path)

    for colname in remove_columns:
        data.pop(colname)

    for colname in encode_columns:
        encode_column(data, colname)

    if show_correlation:
        f = plt.figure(figsize=(19, 15))
        plt.matshow(data.corr(), fignum=f.number)
        plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
        plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16);
        plt.show()

    #analyze_good_employees(data)

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
    print("training...")
    classifier.fit(X, y)
    print("done.")

    return classifier

def build_regressor(X, y, **kwargs) -> MLPRegressor:
    regressor = MLPRegressor(**kwargs)
    print("training...")
    regressor.fit(X, y)
    print("done.")

    return regressor

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

def analyze_good_employees(data):
    averages = data.mean()
    average_last_evaluation = averages['last_evaluation']
    average_project = averages['number_project']
    average_montly_hours = averages['average_montly_hours']
    average_time_spend = averages['time_spend_company']

    good_employees = data[data['last_evaluation'] > average_last_evaluation]
    good_employees = good_employees[good_employees['number_project'] > average_project]
    good_employees = good_employees[good_employees['average_montly_hours'] > average_montly_hours]
    good_employees = good_employees[good_employees['time_spend_company'] > average_time_spend]

    sns.set()
    plt.figure(figsize=(15, 8))
    plt.hist(data['left'])
    print(good_employees.shape)
    sns.heatmap(good_employees.corr(), vmax=0.5, cmap="PiYG")
    plt.title('Correlation matrix')
    plt.show()

def human_resources():
    X, y = read_data('human_resources.csv', 'left', encode_columns=["sales", "salary"])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #classifier = build_classifier(X_train, y_train)
    #classifier = build_classifier(X_train, y_train)

    # graph_tree(classifier, 'tree.dot', ['Meteo','Amis','Vent','Jour'])

    #evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

def bikes_hour():
    #X, y = read_data('./hour.csv', 'cnt', remove_columns=['instant', 'dteday', 'registered'], show_correlation=True)
    X, y = read_data('./hour.csv', 'cnt', remove_columns=['instant', 'dteday'], show_correlation=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    regressor = build_regressor(X_train, y_train)

    evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def main():
    #human_resources()
    bikes_hour()

if __name__ == '__main__':
    main()
