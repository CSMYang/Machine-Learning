'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from numpy.linalg import inv, det


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        corresponding_digits = data.get_digits_by_label(train_data,
                                                        train_labels, i)
        mean = np.mean(corresponding_digits, 0)
        means[i, :] = mean
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    all_means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        corresponding_digits = data.get_digits_by_label(train_data,
                                                        train_labels, i)
        np.fill_diagonal(covariances[i], 0.01)
        for pixel_1 in range(64):
            for pixel_2 in range(64):
                cov = np.mean((corresponding_digits[:, pixel_1] -
                               all_means[i, pixel_1]) *
                              (corresponding_digits[:, pixel_2] -
                               all_means[i, pixel_2]), 0)
                covariances[i, pixel_1, pixel_2] += cov
    return covariances


def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        cov.append(cov_diag.reshape(8, 8))
    cov = np.log(np.concatenate(cov, 1))
    plt.imshow(cov, cmap='gray')
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    n = digits.shape[0]
    to_ret = np.zeros((n, 10))
    determinant = det(covariances)
    inverse = inv(covariances)
    print("Start computing generative likelihood:")
    for i in range(n):
        for j in range(10):
            first_part = -5 * np.log(2 * np.pi)
            second_part = -0.5 * np.log(determinant[j])
            third_part = -0.5 * np.dot(np.dot((digits[i] - means[j]).T,
                                              inverse[j]),
                                       (digits[i] - means[j]))
            to_ret[i, j] = first_part + second_part + third_part
            # print("The value in to_ret[{}, {}] is: "
            #       "{}".format(i, j, to_ret[i, j]))
    print("Generative likelihood computed!")
    return to_ret


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    print('Start computing conditional likelihood:')
    gen_probs = generative_likelihood(digits, means, covariances)
    prob_y_k = 0.1
    top = gen_probs + np.log(prob_y_k)
    bottom = np.log(np.sum((prob_y_k * np.exp(gen_probs)), 1).reshape(-1, 1))
    print('Conditional likelihood computed!')
    return top - bottom


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    print('Start computing average conditional likelihood:')
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    n = cond_likelihood.shape[0]
    to_ret = np.zeros(cond_likelihood.shape)
    for i in range(n):
        to_ret[i, int(labels[i])] += 1
    print('Average conditional likelihood computed!')
    return np.sum(to_ret * cond_likelihood) / n


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    to_ret = []
    for prob in cond_likelihood:
        to_ret.append(np.argmax(prob))
    return to_ret


def accuracy_calculator(classified_labels, true_labels):
    """
    This functions takes the classified_labels and true_labels and computes
    the accuracy.
    """
    acc = 0
    for i in range(len(classified_labels)):
        if classified_labels[i] == true_labels[i]:
            acc += 1
    return acc/len(classified_labels)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)
    # Evaluation
    test_cllh = avg_conditional_likelihood(test_data, test_labels,
                                           means, covariances)
    print()
    print('The average conditional log-likelihood on '
          'test set is: {}'.format(test_cllh))
    print()
    train_cllh = avg_conditional_likelihood(train_data, train_labels,
                                            means, covariances)
    print()
    print('The average conditional log-likelihood on '
          'train set is: {}'.format(train_cllh))
    print()
    train_classified_labels = classify_data(train_data, means, covariances)
    test_classified_labels = classify_data(test_data, means, covariances)
    print("The train accuracy is: {}".format(
        accuracy_calculator(train_classified_labels, train_labels)))
    print('The test accuracy is: {}'.format(
        accuracy_calculator(test_classified_labels, test_labels)))


if __name__ == '__main__':
    main()
