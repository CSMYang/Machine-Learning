'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.ones((10, 64))
    for i in range(train_data.shape[0]):
        corresponding_label = int(train_labels[i])
        eta[corresponding_label, :] += train_data[i, :]
    return eta / (0.1 * train_data.shape[0] + 2)


def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    images = []
    for i in range(10):
        img_i = class_images[i]
        images.append(img_i.reshape(8, 8))
    images = np.concatenate(images, 1)
    plt.imshow(images, cmap='gray')
    plt.show()



def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    random_sample = np.random.uniform(size=(10, 64))
    for i in range(10):
        for j in range(64):
            if random_sample[i, j] < eta[i, j]:
                generated_data[i, j] += 1
    plot_images(generated_data)


def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    to_ret = np.zeros((bin_digits.shape[0], 10))
    for i in range(10):
        for j in range(bin_digits.shape[0]):
            to_ret[j, i] = np.prod(np.power(eta[i], bin_digits[j])
                                   * np.power(1 - eta[i], 1 - bin_digits[j]))
    return np.log(to_ret)


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_probs = generative_likelihood(bin_digits, eta)
    prob_y_k = 0.1
    top = gen_probs + np.log(prob_y_k)
    bottom = np.log(np.sum((prob_y_k * np.exp(gen_probs)), 1).reshape(-1, 1))
    return top - bottom


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return

    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    n = cond_likelihood.shape[0]
    to_ret = np.zeros(cond_likelihood.shape)
    for i in range(n):
        to_ret[i, int(labels[i])] += 1

    return np.sum(to_ret * cond_likelihood) / n


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
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
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)
    test_cllh = avg_conditional_likelihood(test_data, test_labels, eta)
    print('The average conditional log-likelihood on '
          'test set is: {}'.format(test_cllh))
    train_cllh = avg_conditional_likelihood(train_data, train_labels, eta)
    print('The average conditional log-likelihood on '
          'train set is: {}'.format(train_cllh))
    train_classified_labels = classify_data(train_data, eta)
    test_classified_labels = classify_data(test_data, eta)
    print("The train accuracy is: {}".format(
        accuracy_calculator(train_classified_labels, train_labels)))
    print('The test accuracy is: {}'.format(
        accuracy_calculator(test_classified_labels, test_labels)))


if __name__ == '__main__':
    main()
