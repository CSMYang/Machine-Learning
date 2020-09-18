import data
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix, \
    accuracy_score, precision_score, recall_score
from torch.nn.functional import one_hot


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
    plt.title('Roc for Ada')
    plt.legend(loc="lower right")
    plt.show()


def draw_conf(classifier, X_test, y_test):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title('Confusion matrix for Ada')
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')
    tuned_parameters = [{'n_estimators': [10, 25, 50, 100],
                         'learning_rate': [0.1, 0.5, 0.75, 1, 2]}]

    ada = GridSearchCV(AdaBoostClassifier(), tuned_parameters)
    ada.fit(train_data, train_labels)
    print('The best parameters are:')
    print(ada.best_params_)
    print('The test score for AdaBoost is {}'.format(ada.score(test_data,
                                                               test_labels)))

    y_pred = ada.predict(test_data)
    acc_score = accuracy_score(test_labels, y_pred)
    print('The accuracy score for Ada is: {}'.format(acc_score))
    prec_score = precision_score(test_labels, y_pred, average='macro')
    print('The precision score for Ada is: {}'.format(prec_score))
    rec_score = recall_score(test_labels, y_pred, average='macro')
    print('The recall score for Ada is: {}'.format(rec_score))

    draw_conf(ada, test_data, test_labels)
    te_labels = []
    for y in test_labels:
        te_labels.append(int(y))
    test_labels = one_hot(torch.tensor(te_labels))
    y_score = ada.decision_function(test_data)
    draw_roc(test_labels, y_score)


if __name__ == '__main__':
    main()
