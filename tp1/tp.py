import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1337)

def gen_data(nb: int):
    x = np.random.uniform(-3, 10, nb)
    x.sort()
    y = np.random.normal(size=nb) + np.sin(x) * 10 / x
    return x, y


def setup_plot():
    plt.xlim(-4, 12)
    plt.ylim(-5, 15)
    plt.ylabel("f(x) = 10*sin(x)/x + gaussian")
    plt.xlabel("x")
    plt.suptitle('Plot of our dataset')
    plt.grid(b=True)


def plot_data(X, y, *, fmt=None, label=None):
    if fmt is None:
        plt.plot(X, y, label=label)
    else:
        plt.plot(X, y, fmt, label=label)


def show_plot():
    plt.legend()
    plt.show()


def polynomial_pipeline(degree, X, y):
    pipeline = make_pipeline(PolynomialFeatures(degree), Ridge())
    pipeline.fit(X.reshape(-1, 1), y)
    return pipeline


def rss(X, y, model) -> float:
    total = 0
    for y, f_x in zip(y, model.predict(X)):
        total += (y - f_x) ** 2
    return total


def main():
    X, y = gen_data(15)
    setup_plot()
    #plot_data(X, y, fmt="ko", label="Training points")
    X_test, y_test = gen_data(50)
    plot_data(X_test, y_test, fmt="ko", label="Test points")
    for degree in (1, 3, 6, 9, 12):
        pipeline = polynomial_pipeline(degree, X, y)
        fake_X = np.linspace(-3, 10, 100).reshape(-1, 1)
        prediction = pipeline.predict(fake_X)
        y_test_predicted = pipeline.predict(X_test.reshape(-1, 1))
        plot_data(fake_X, prediction, label=f"degree {degree}")
        print(f"Degree {degree}: R-squared: {r2_score(y_test.reshape(-1, 1), y_test_predicted.reshape(-1, 1))}")
        print(f"Degree {degree}: Residual sum of squares: {rss(X_test.reshape(-1, 1), y_test, pipeline)}")

    show_plot()
# d=0, ligne droite  horizontale
# d=1, modèle linéaire


if __name__ == "__main__":
    main()
