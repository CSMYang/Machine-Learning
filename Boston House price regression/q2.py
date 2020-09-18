from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.xlabel(features[i])
        plt.ylabel('price')
        plt.grid(True)
        plt.plot(X[:, i], y, '.')
    plt.tight_layout()
    plt.show()


def min_max_normalize(x):
    """
    This function returns the normalized version of x, excluding the first
    column.
    """
    X = x.copy()
    for i in range(1, X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min())/(X[:, i].max() - X[:, i].min())
    return X


def fit_regression(X, Y):
    a = np.matmul(X.T, X)
    b = np.matmul(X.T, Y)
    w = np.linalg.solve(a, b)

    return w


def mse_calculator(y_hat, y):
    """
    This function takes the coefficients of linear regression function, one test
    set and its corresponding values, and then calculate the MSE of given data
    set.
    """
    return np.square(y_hat - y).mean()


def mae_calculator(y_hat, y):
    """
    This function takes the coefficients of linear regression function, one test
    set and its corresponding values, and then calculate the MAE of given data
    set.
    """
    return np.absolute(y_hat - y).mean()


def mape_calculator(y_hat, y):
    """
    This function takes the coefficients of linear regression function, one test
    set and its corresponding values, and then calculate the SSE of given data
    set.
    """
    percentages = 0
    errors = np.absolute(y_hat - y)
    for i in range(errors.shape[0]):
        percentages += errors[i]/y[i]
    return (percentages/errors.shape[0]) * 100


def tabulator(features, weights):
    """
    This function takes an array of features and an array of its corresponding
    weights, and prints a table with each feature associating with a weight.
    """
    feature_string = 'Bias '
    weight_string = ''
    for feature in features:
        feature_string += feature
        feature_string += ' '

    for weight in weights:
        weight_string += str(round(weight, 3))
        weight_string += ' '

    print(feature_string)
    print(weight_string)


def main():
    # Load the data
    X, y, features = load_data()
    print("This data set has {} data points, {} features,"
          " and the features are: {}".format(X.shape[0], X.shape[1], features))
    # print("The targets are {}".format(y))
    # Visualize the features
    visualize(X, y, features)

    bias_col = np.ones((X.shape[0], 1))
    X = np.concatenate((bias_col, X), axis=1)
    X = min_max_normalize(X)
    x_train, x_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=737)

    # Fit regression model
    w = fit_regression(x_train, y_train)
    # Compute fitted values, MSE, etc.
    tabulator(features, w)
    fitted_value = []
    for i in range(x_test.shape[0]):
        fitted_value.append(np.dot(w, x_test[i, :]))
    fitted_value = np.array(fitted_value)
    mse = mse_calculator(fitted_value, y_test)
    print("MSE is: {}".format(mse))
    print("MAPE is: {}%".format(mape_calculator(fitted_value, y_test)))
    print("MAE is: {}".format(mae_calculator(fitted_value, y_test)))


if __name__ == "__main__":
    main()
