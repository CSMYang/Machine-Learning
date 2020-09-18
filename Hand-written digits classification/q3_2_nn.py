import data
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, \
    confusion_matrix, accuracy_score, recall_score, precision_score
from torch.nn.functional import one_hot


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 10, bias=True)
        self.fc4 = nn.Linear(10, 10, bias=True)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        x = torch.softmax(self.fc4(x), dim=1)
        return x


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
    plt.title('ROC for Neural Net')
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


if __name__ == '__main__':
    # Loading data and convert the data into tensors
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')
    train_data = torch.from_numpy(np.array(train_data))
    train_labels = torch.from_numpy(np.array(train_labels))
    test_data = torch.from_numpy(np.array(test_data))
    test_label_for_cm = np.copy(test_labels)
    # Create one-hot test labels for measuring accuracy.
    te_labels = []
    for y in test_labels:
        te_labels.append(int(y))
    test_labels = one_hot(torch.tensor(te_labels))

    # Set up the neural net and starts training.
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):
        for i, data in enumerate(train_data):
            optimizer.zero_grad()
            output = net(data.unsqueeze(0).float())
            loss = criterion(output, torch.tensor([train_labels[i]]).long())
            loss.backward()
            optimizer.step()

    # Computes the classification accuracy.
    hit = 0
    total = 0
    outputs = []
    out_values = []
    with torch.no_grad():
        for x, y in enumerate(test_data):
            output = net(y.unsqueeze(0).float())
            outputs.append(output[0])
            out_values.append(torch.argmax(output))
            if test_labels[x][torch.argmax(output)] == 1:
                hit += 1
            total += 1
    print("The hit rate for Neural Net is: {}".format(hit/total))

    outputs = torch.stack(outputs)
    draw_roc(test_labels, outputs)
    draw_conf(test_label_for_cm, out_values)
    acc_score = accuracy_score(test_label_for_cm, out_values)
    print('The accuracy score for Neural Net is: {}'.format(acc_score))
    prec_score = precision_score(test_label_for_cm, out_values, average='macro')
    print('The precision score for Neural Net is: {}'.format(prec_score))
    rec_score = recall_score(test_label_for_cm, out_values, average='macro')
    print('The recall score for Neural Net is: {}'.format(rec_score))
