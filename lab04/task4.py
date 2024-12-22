import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import numpy as np
import sklearn.metrics as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)
fig, ax = plt.subplots()

ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()

print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(ytest, ypred), 2))
print("Mean squared error =", round(sm.mean_squared_error(ytest, ypred), 2))
print("Median absolute error =", round(sm.median_absolute_error(ytest, ypred), 2))
print("Explain variance score =", round(sm.explained_variance_score(ytest, ypred), 2))
print("R2 score =", round(sm.r2_score(ytest, ypred), 2))