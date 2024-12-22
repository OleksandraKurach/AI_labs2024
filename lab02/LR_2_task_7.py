import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


data = []
labels = []
input_file = 'income_data.txt'

with open(input_file, 'r') as file:
    for line in file.readlines():
        if '?' in line:
            continue
        line_data = line.strip().split(', ')
        data.append(line_data[:-1])
        labels.append(line_data[-1])

data = pd.DataFrame(data)
labels = pd.Series(labels)

encoder = LabelEncoder()
encoded_data = data.apply(encoder.fit_transform)
encoded_labels = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(encoded_data, encoded_labels, test_size=0.2, random_state=42)

models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів класифікації')
plt.xlabel('Алгоритми')
plt.ylabel('Точність')
plt.grid()
plt.show()
