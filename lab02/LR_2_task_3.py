from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

print("\nОпис набору даних (перші 193 символи):")
print(iris_dataset['DESCR'][:193] + "\n...")

print("\nНазви відповідей (цільових класів): {}".format(iris_dataset['target_names']))

print("\nНазва ознак: \n{}".format(iris_dataset['feature_names']))

print("\nТип масиву 'data': {}".format(type(iris_dataset['data'])))
print("Форма масиву 'data': {}".format(iris_dataset['data'].shape))

print("\nТип масиву 'target': {}".format(type(iris_dataset['target'])))

print("\nВідповіді (цільові значення):\n{}".format(iris_dataset['target']))

