import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def load_dataframe(path: str, *, to_pop=()) -> pandas.DataFrame:
    dataframe = pandas.read_csv(path)
    for col in to_pop:
        dataframe.pop(col)
    return dataframe


def get_knn_classifier(n, X, y):
    classifier = KNeighborsClassifier(n)
    classifier.fit(X, y)
    return classifier


def get_tree_classifier(X, y, criterion='entropy', **kwargs) -> DecisionTreeClassifier:
    classifier = DecisionTreeClassifier(criterion=criterion, **kwargs)
    classifier.fit(X, y)
    return classifier

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)
    print(f"Train score: {train_score}, Test score: {test_score}")
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))


def normalize(X, y):
    """Return normalized X."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)
    return X


def analyzeData(data):
    # le nombre d'exemples et de caractéristiques
    exemples, caracteristiques = data.shape
    print(f"Caractéristiques : {caracteristiques}, Exemples : {exemples}")
    # les différentes statistiques ?

    # le nombre d'exemples de chaque classe
    classes = data['Z'].unique()
    classes.sort()
    print("Population classes:")
    for class_ in classes:
        pop = len(data[data['Z'] == class_])
        print(f"    {class_}: {pop}")

    # la matrice de corrélation
    sns.set()
    plt.figure(figsize=(15, 8))
    sns.heatmap(data.corr(), vmin=-0.75, vmax=0.75, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True))
    plt.title('Matrice de corrélation')
    plt.show()


def learning_knn(X, y, *, n):
    print(f"Learning with KNN ({n})")
    X = normalize(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    classifier = get_knn_classifier(n, X_train, y_train)
    evaluate_classifier(classifier, X_train, y_train, X_test, y_test)


def learning_tree(X, y):
    print(f"Learning with decision tree")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    # Les paramètres par défaut donnent un assez bon résultat
    classifier = get_tree_classifier(X_train, y_train)
    evaluate_classifier(classifier, X_train, y_train, X_test, y_test)


def main():
    X = load_dataframe(
        "exam.csv",
        to_pop=(
            "A",  # Constant
            "B",  # Constant
            "C",  # Seems to be a unique ID
            "H", "J", "R", "T", "U", "V", "W", "X", "Y", # Corrélation ~= 0
        ),
    )
    analyzeData(X)
    y = X.pop("Z")

    # 3 est un bon compromis, la population minimum étant de 4
    learning_knn(X, y, n=3)
    learning_tree(X, y)


if __name__ == "__main__":
    main()
