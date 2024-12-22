import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Classify data using Ensemble Learning techniques'
    )
    parser.add_argument(
        '--classifier-type',
        dest='classifier_type',
        required=False,
        default='rf',
        choices=['rf', 'erf'],
        help="Type of classifier to use; can be either 'rf' or 'erf'"
    )
    return parser


def load_data(input_file):
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y


def plot_input_data(X, y):
    classes = np.unique(y)
    markers = ['s', 'o', '^']
    plt.figure()
    for idx, cls in enumerate(classes):
        class_data = X[y == cls]
        plt.scatter(
            class_data[:, 0], class_data[:, 1],
            s=75, facecolors='white', edgecolors='black',
            linewidth=1, marker=markers[idx], label=f'Class {int(cls)}'
        )
    plt.title('Input Data')
    plt.legend()
    plt.show()


def get_classifier(classifier_type, params):
    if classifier_type == 'rf':
        return RandomForestClassifier(**params)
    elif classifier_type == 'erf':
        return ExtraTreesClassifier(**params)


def plot_decision_boundaries(classifier, X, y, title):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
    markers = ['s', 'o', '^']
    for idx, cls in enumerate(np.unique(y)):
        class_data = X[y == cls]
        plt.scatter(
            class_data[:, 0], class_data[:, 1],
            s=75, edgecolor='black', marker=markers[idx], label=f'Class {int(cls)}'
        )
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    input_file = 'data_random_forests.txt'
    X, y = load_data(input_file)
    plot_input_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5
    )
    classifier = get_classifier(classifier_type, params)
    classifier.fit(X_train, y_train)

    plot_decision_boundaries(classifier, X_train, y_train, "Training Set")
    plot_decision_boundaries(classifier, X_test, y_test, "Test Set")

    class_names = [f'Class-{int(cls)}' for cls in np.unique(y)]
    print("\n" + "#" * 40)
    print("Classifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("Classifier performance on test dataset\n")
    print(classification_report(y_test, classifier.predict(X_test), target_names=class_names))
    print("#" * 40 + "\n")

    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
    print("Confidence measure:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = f"Class-{np.argmax(probabilities)}"
        print('\nDatapoint:', datapoint)
        print('Predicted class:', predicted_class)

    plot_decision_boundaries(classifier, np.vstack((X_test, test_datapoints)),
                             np.hstack((y_test, [0] * len(test_datapoints))), "Test Points with Predictions")


if __name__ == '__main__':
    main()