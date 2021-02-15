import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[6], [8], [10], [14], [18]])
y = [7, 9, 13, 17.5, 18]

import matplotlib.pyplot as plt

# EX 1
plt.plot(X, y, 'ko')
plt.ylabel('Prices in euros')
plt.ylim(0, 25)

plt.xlabel('Sizes in cms')
plt.xlim(0, 25)

plt.suptitle('Pizza prices plotted against sizes')
plt.grid(b=True)
#plt.show()

# EX 2
reg = LinearRegression()
reg.fit(X, y)
fake_values=[[0], [25]]
prediction=reg.predict(fake_values)
plt.plot(fake_values, prediction)
#plt.show()

# EX 3
def rss(X, y, model) -> float:
    total = 0
    for y, f_x in zip(y, model.predict(X)):
        total += (y - f_x) ** 2
    return total

print("Residual sum of squares : {:.2f}".format(rss(X, y, reg)))
