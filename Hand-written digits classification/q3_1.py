'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''
import torch
import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, \
    confusion_matrix, accuracy_score, recall_score, precision_score
from torch.nn.functional import one_hot


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data ** 2).sum(axis=1).reshape(-1, 1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        """
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array Output: dist is a numpy array
        containing the distances between the test point and each training point
        """
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point ** 2).sum(axis=1).reshape(1, -1)
        dist = self.train_norm + test_norm - 2 * self.train_data.dot(
            test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        """
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        """
        distances = np.array([self.l2_distance(test_point)])
        labels = np.array([self.train_labels])
        distances_with_labels = np.concatenate((distances.T, labels.T),
                                               axis=1)
        sorted_distances = distances_with_labels[np.argsort(
            distances_with_labels[:, 0])]
        label_and_occurrences = dict()
        for i in range(k):
            if sorted_distances[i, 1] not in label_and_occurrences:
                label_and_occurrences[sorted_distances[i, 1]] = \
                    [1, [sorted_distances[i, 0]]]
            else:
                label_and_occurrences[sorted_distances[i, 1]][0] += 1
                label_and_occurrences[sorted_distances[i, 1]][1].append(
                    sorted_distances[i, 0])
        highest, second_highest = None, None
        for key in label_and_occurrences:
            if not (highest and second_highest):
                highest, second_highest = key, key
            elif label_and_occurrences[key][0] > \
                    label_and_occurrences[highest][0]:
                highest = key
            elif label_and_occurrences[key][0] >= \
                    label_and_occurrences[second_highest][0]:
                second_highest = key
        if label_and_occurrences[highest] > \
                label_and_occurrences[second_highest]:
            return highest
        highest_mean_distance = 0
        for dist in label_and_occurrences[highest][1]:
            highest_mean_distance += dist
        highest_mean_distance = highest_mean_distance / \
                    len(label_and_occurrences[highest][1])
        second_mean_distance = 0
        for dist in label_and_occurrences[second_highest][1]:
            second_mean_distance += dist
        second_mean_distance = second_mean_distance / \
                    len(label_and_occurrences[second_highest][1])
        if highest_mean_distance <= second_mean_distance:
            return highest
        return second_highest


def cross_validation(train_data, train_labels, k_range=np.arange(1, 16)):
    """
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of
    train_data,train_labels. The intention was for students to take the
    training data from the knn object - this should be clearer from the new
    function signature.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=1746)

    for k in k_range:
        test_acc = []
        for train_index, test_index in kf.split(train_data):
            x_train, x_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            knn = KNearestNeighbor(x_train, y_train)
            test = classification_accuracy(knn, k, x_test, y_test)
            test_acc.append(test)
        accumulated_acc = 0
        for acc in test_acc:
            accumulated_acc += acc
        print('The average accuracy for k = {} using 10-Fold-Cross-Validation'
              ' is: {}'.format(k, accumulated_acc/len(test_acc)))


def classification_accuracy(knn, k, eval_data, eval_labels):
    """
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    """
    num_correct = 0
    for i in range(len(eval_data)):
        label = knn.query_knn(eval_data[i], k)
        if label == eval_labels[i]:
            num_correct += 1
    return num_correct/len(eval_labels)


def draw_roc(test_labels, outputs):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for KNN')
    plt.legend(loc="lower right")
    plt.show()


def draw_conf(y_labels, y_pred):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cm = confusion_matrix(y_labels, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')
    knn = KNearestNeighbor(train_data, train_labels)

    cross_validation(train_data, train_labels)
    train_acc_1 = classification_accuracy(knn, 1, train_data, train_labels)
    test_acc_1 = classification_accuracy(knn, 1, test_data, test_labels)
    train_acc_15 = classification_accuracy(knn, 15, train_data, train_labels)
    test_acc_15 = classification_accuracy(knn, 15, test_data, test_labels)
    print('For k = 1, the train accuracy is {} '
          'and the test accuracy is {}'.format(train_acc_1, test_acc_1))
    print('For k = 15, the train accuracy is {} '
          'and the test accuracy is {}'.format(train_acc_15, test_acc_15))

    y_pred = []
    for i in range(len(test_data)):
        label = knn.query_knn(test_data[i], 2)
        y_pred.append(label)

    pred_array = []
    for pred in y_pred:
        to_insert = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        to_insert[int(pred)] = 1
        pred_array.append(to_insert)

    pred_array = np.array(pred_array)
    y_pred = np.array(y_pred)

    draw_conf(test_labels, y_pred)
    acc_score = accuracy_score(test_labels, y_pred)
    print('The accuracy score for KNN is: {}'.format(acc_score))
    prec_score = precision_score(test_labels, y_pred, average='macro')
    print('The precision score for KNN is: {}'.format(prec_score))
    rec_score = recall_score(test_labels, y_pred, average='macro')
    print('The recall score for KNN is: {}'.format(rec_score))

    te_labels = []
    for y in test_labels:
        te_labels.append(int(y))
    te_labels = one_hot(torch.tensor(te_labels))

    draw_roc(te_labels, pred_array)


if __name__ == '__main__':
    main()
