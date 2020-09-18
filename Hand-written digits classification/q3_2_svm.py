import data, torch
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from torch.nn.functional import one_hot
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix,\
    accuracy_score, precision_score, recall_score


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
    plt.title('Roc for SVM')
    plt.legend(loc="lower right")
    plt.show()


def draw_conf(classifier, X_test, y_test):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title('Confusion matrix for SVM')
    plt.show()


def svm_classifier(x_test, y_test, x_train, y_train):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto'],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']
    for metric in scores:
        print('Now tuning hyper-parameter for {}:'.format(metric))
        svm = GridSearchCV(SVC(), tuned_parameters,
                           scoring='{}_macro'.format(metric))
        svm.fit(x_train, y_train)
        print('Test score using {} as metric is: '
              '{}'.format(metric, svm.score(x_test, y_test)))
        print(svm.best_params_)
    draw_conf(svm, x_test, y_test)

    y_pred = svm.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)
    print('The accuracy score for SVM is: {}'.format(acc_score))
    prec_score = precision_score(y_test, y_pred, average='macro')
    print('The precision score for SVM is: {}'.format(prec_score))
    rec_score = recall_score(y_test, y_pred, average='macro')
    print('The recall score for SVM is: {}'.format(rec_score))

    te_labels = []
    for y in y_test:
        te_labels.append(int(y))
    y_test = one_hot(torch.tensor(te_labels))
    y_score = svm.decision_function(test_data)
    draw_roc(y_test, y_score)


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')
    svm_classifier(test_data, test_labels, train_data, train_labels)
