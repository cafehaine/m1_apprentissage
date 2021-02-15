import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def load_dataframe(path: str, *, to_pop=()) -> pandas.DataFrame:
    dataframe = pandas.read_csv(path)
    for col in to_pop:
        dataframe.pop(col)
    return dataframe

def plot_attributes(dataframe: pandas.DataFrame) -> None:
    sns.set_style("whitegrid")
    sns.pairplot(dataframe, hue="Species")
    plt.show()

def get_knn_classifier(n, X, y):
    classifier = KNeighborsClassifier(n)
    classifier.fit(X, y)
    return classifier

def get_knn_regressor(n, X, y):
    regressor = KNeighborsRegressor(n)
    regressor.fit(X, y)
    return regressor

def evaluate_classifier(classifier,X_train, y_train, X_test, y_test):
    print("Score train:", classifier.score(X_train, y_train))
    print("Score test:", classifier.score(X_test, y_test))
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

def iris():
    X = load_dataframe("./iris.csv", to_pop=("Id",))
    #plot_attributes(dataframe)
    y = X.pop("Species")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    for n in range(1,11):
        print(f"==== {n} ====")
        classifier = get_knn_classifier(n, X_train, y_train)
        evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

def normalize(X, y):
    """Return normalized X."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)
    return X

def mpg():
    X = load_dataframe("./auto-mpg.data", to_pop=("name",))
    y = X.pop("mpg")
    X = normalize(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    for n in range(1,11):
        print(f"==== {n} ====")
        regressor = get_knn_regressor(n, X_train, y_train)
        evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def main():
    #iris()
    mpg()

if __name__ == "__main__":
    main()
