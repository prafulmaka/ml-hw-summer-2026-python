import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class TrainingData:
    def __init__(self, n):
        self.points = np.zeros((n, 2), dtype=float)
        self.count = 0

    def insert_point(self, x, y):
        self.points[self.count, 0] = x
        self.points[self.count, 1] = y
        self.count += 1

    def get_features_and_labels(self):
        x_data = self.points[: self.count, 0].reshape(-1, 1)
        y_data = self.points[: self.count, 1].astype(int)
        return x_data, y_data


def read_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Value must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def read_real_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a real number.")


def read_non_negative_integer(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            print("Value must be a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter a non-negative integer.")


def read_dataset(n):
    data = TrainingData(n)
    for i in range(n):
        x = read_real_number(f"Enter x for point {i + 1}: ")
        y = read_non_negative_integer(f"Enter y for point {i + 1}: ")
        data.insert_point(x, y)
    return data


def find_best_knn(x_train, y_train, n):
    if n == 1:
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(x_train, y_train)
        return 1, model

    min_class_count = min(np.bincount(y_train))
    cv = min(5, n, min_class_count)
    min_train_size = n * (cv - 1) // cv
    max_k = min(10, n, min_train_size)
    param_grid = {"n_neighbors": list(range(1, max_k + 1))}

    if cv < 2:
        best_k = 1
        best_score = -1.0
        for k in param_grid["n_neighbors"]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(x_train, y_train)
            score = model.score(x_train, y_train)
            if score > best_score:
                best_score = score
                best_k = k
        model = KNeighborsClassifier(n_neighbors=best_k)
        model.fit(x_train, y_train)
        return best_k, model

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv)
    grid_search.fit(x_train, y_train)
    best_k = grid_search.best_params_["n_neighbors"]
    return best_k, grid_search.best_estimator_


def main():
    n = read_positive_integer("Enter N, a positive integer: ")
    train_data = read_dataset(n)
    x_train, y_train = train_data.get_features_and_labels()

    m = read_positive_integer("Enter M, a positive integer: ")
    test_data = read_dataset(m)
    x_test, y_test = test_data.get_features_and_labels()

    best_k, model = find_best_knn(x_train, y_train, n)
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(best_k)
    print(test_accuracy)


if __name__ == "__main__":
    main()


