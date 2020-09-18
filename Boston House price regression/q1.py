from matplotlib import pyplot as plt
import numpy as np

def plot(mean, variance, size, lams):
    """
    This function takes the mean, variance and size of one optimization problem
    and plots the bias-variance decomposition of that problem.
    """
    bias = ((lams * mean) / (lams + 1)) ** 2
    var = variance/(size * ((lams + 1) ** 2))
    expected_squared_error = bias + var
    plt.title('Bias-variance decomposition')
    plt.xlabel('lambda')
    plt.grid(True)
    plt.plot(lams, bias, color='r', label='Bias')
    plt.plot(lams, var, color='g', label='Variance')
    plt.plot(lams, expected_squared_error, color='b',
             label='Expected squared error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lams = np.linspace(0, 10, 1000)
    plot(1, 9, 10, lams)