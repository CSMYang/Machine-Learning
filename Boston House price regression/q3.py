from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
from sklearn.utils import shuffle

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x),
                   axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    """
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    top = -l2(test_datum.T, x_train)/(2 * (tau ** 2))
    bottom = np.array([logsumexp(-l2(test_datum.T, x_train)/(2 * (tau ** 2)))])
    local_weights = np.diag(np.exp((top - bottom)).flatten())
    a = np.matmul(x_train.T, np.matmul(local_weights, x_train))
    b = np.matmul(x_train.T, np.matmul(local_weights, y_train))
    w = np.linalg.solve(a, b)
    return np.matmul(w.T, test_datum)


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array(
            [LRLS(x_test[i, :].reshape(d, 1), x_train, y_train, tau) \
             for i in range(N_test)])
        losses[j] = ((predictions.flatten() - y_test.flatten()) ** 2).mean()
    return losses


# to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    x_shuffled, y_shuffled = shuffle(x, y)
    partition = round(x_shuffled.shape[0]/k)
    k_fold_losses = []
    for num_part in range(k):
        x_test = x_shuffled[(num_part * partition):((num_part + 1) * partition)]
        y_test = y_shuffled[(num_part * partition):((num_part + 1) * partition)]
        x_train = np.concatenate((x_shuffled[:num_part * partition],
                                  x_shuffled[num_part * partition:]))
        y_train = np.concatenate((y_shuffled[:num_part * partition],
                                  y_shuffled[num_part * partition:]))
        k_fold_losses.append(run_on_fold(x_test,
                                         y_test, x_train, y_train, taus))
    k_fold_losses = np.array(k_fold_losses)
    print("k_fold_losses has {} rows and {} columns".format(
        k_fold_losses.shape[0], k_fold_losses.shape[1]))

    return k_fold_losses


def min_max_normalize(x):
    """
    This function returns the normalized version of x, excluding the first
    column.
    """
    X = x.copy()
    for i in range(1, X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min())/(X[:, i].max() - X[:, i].min())
    return X


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    x = min_max_normalize(x)
    losses = run_k_fold(x, y, taus, k=5)
    plt.title('Loss VS Tau')
    plt.xlabel('Tau')
    plt.ylabel('Loss')
    for i in range(5):
        plt.plot(taus, losses[i], label="The {}-th cross validation".format(i))
    plt.legend()
    plt.show()
    print("min loss = {}".format(losses.min()))
