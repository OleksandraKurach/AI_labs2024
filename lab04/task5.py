import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = np.sin(X).flatten() + np.random.uniform(-0.5, 0.5, m)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)

print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)

y_pred = lin_reg.predict(X_poly)

plt.scatter(X, y, color='green', label='Дані з шумом')
plt.plot(X, y_pred, color='red', linewidth=1, label='Прогноз')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
