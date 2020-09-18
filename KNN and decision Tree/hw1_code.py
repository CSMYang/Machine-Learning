import math
import graphviz
import os
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def load_data(real, fake):
    """
    This function loads data from file and return the vectorized version
    of file.
    """
    real_file = open(real, 'r')
    fake_file = open(fake, 'r')
    real = real_file.readlines()
    fake = fake_file.readlines()
    labels = []

    for _ in real:
        labels.append(True)

    for _ in fake:
        labels.append(False)

    all_news = real + fake
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(all_news)
    all_features = vectorizer.get_feature_names()
    x_train, x_test, y_train, y_test = \
        train_test_split(data, labels, test_size=0.3, random_state=4)
    x_test, x_validation, y_test, y_validation = \
        train_test_split(x_test, y_test, test_size=0.5, random_state=4)
    return all_features, x_train, x_test, x_validation, y_train, \
        y_test, y_validation


def select_tree_model(x_train, x_validation, y_train,
                      y_validation, sensible_depth):
    """
    This function selects and trains a decision tree model
    """
    criterion = ['entropy', 'gini']
    results = []
    current_best_tree = None
    current_best_tree_stats = None
    for depth in sensible_depth:
        for criteria in criterion:
            decision_tree = DecisionTreeClassifier(criteria, max_depth=depth,
                                                   random_state=4)
            decision_tree.fit(x_train, y_train)
            predictions = decision_tree.predict(x_validation)
            results.append((depth, criteria, accuracy_score(y_validation,
                                                            predictions)))
            if (not current_best_tree) or \
                    current_best_tree_stats[2] <= accuracy_score(y_validation,
                                                                 predictions):
                current_best_tree = decision_tree
                current_best_tree_stats = ((depth, criteria,
                                            accuracy_score(y_validation,
                                                           predictions)))

    print('Decision Tree:')
    for result in results:
        print('Current criteria is: {}, max depth is: {} '
              'and the accuracy is: {}'.format(result[1], result[0], result[2]))

    return current_best_tree


def decision_tree_visualizer(tree, output_name, graph_depth=None):
    """
    This function takes a decision tree and visualizes it.
    """
    try:
        to_be_visualized = export_graphviz(tree, filled=True,
                                           max_depth=graph_depth)
        graph = graphviz.Source(to_be_visualized)
        graph.render(output_name)
        print('Graph of tree generated and named as {}.pdf'.format(output_name))
        print('')
    except Exception:
        print('PATH for graphviz is only accurate for local machine. Please'
              'change PATH.')


def get_node_info(decision_tree, node_id):
    """
    Gets internal information from a certain node from the tree.
    """
    num_samples = decision_tree.tree_.n_node_samples[node_id]
    num_left_sample = decision_tree.tree_.n_node_samples[
        decision_tree.tree_.children_left[node_id]]
    num_right_sample = decision_tree.tree_.n_node_samples[
        decision_tree.tree_.children_right[node_id]]
    sample_value = decision_tree.tree_.value[node_id]
    return decision_tree.tree_.feature[node_id], num_samples, \
        num_left_sample, num_right_sample, sample_value


def compute_information_gain(decision_tree, nodes, features):
    """
    This function takes a list of node_id of the tree and returns the
    information gain from that split.
    """
    to_ret = dict()
    for node_id in nodes:
        # Get the internal information from the node and calculate its entropy
        keyword_id, total_num, left_num, right_num, value = \
            get_node_info(decision_tree, node_id)
        try:
            entropy = - (value[0][0] / total_num) * \
                math.log2(value[0][0] / total_num) - \
                (value[0][1] / total_num) * math.log2(value[0][1] / total_num)
        except ValueError:
            print('The node {} is a leaf'.format(node_id))
            to_ret[node_id] = 0
            continue

        # Calculates the entropy of the left child of the node
        left_node_id = decision_tree.tree_.children_left[node_id]
        left_keyword, total_left, left_left_num, left_right_num, left_value = \
            get_node_info(decision_tree, left_node_id)
        try:
            left_child_entropy = - (left_value[0][0] / total_left) * \
                math.log2(left_value[0][0] / total_left) \
                - (left_value[0][1] / total_left) * math.log2(
                left_value[0][1] / total_left)
        except ValueError:
            print('The left child of node {} is a leaf'.format(node_id))
            left_child_entropy = 0

        # Calculates the entropy of the right child of the node
        right_node_id = decision_tree.tree_.children_right[node_id]
        right_keyword, total_right, right_left_num, right_right_num, \
            right_value = get_node_info(decision_tree, right_node_id)
        try:
            right_child_entropy = - (right_value[0][0] / total_right) * \
                math.log2(right_value[0][0] / total_right) \
                - (right_value[0][1] / total_right) * math.log2(
                right_value[0][1] / total_right)
        except ValueError:
            print('The right child of node {} is a leaf'.format(node_id))
            right_child_entropy = 0

        # Computes the information gain and store it into to_ret
        information_gain = entropy - (left_num / total_num) * \
            left_child_entropy - (right_num / total_num) * right_child_entropy
        to_ret[node_id] = (features[keyword_id], information_gain)

    return to_ret


def knn_visualizer(title, xlabel, ylabel, xticks, data_set1,
                   data_set2, data_label1, data_label2):
    """
    This function outputs a graph of errors of knn model.
    """
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.xticks(xticks)
    pyplot.plot(xticks, data_set1, label=data_label1)
    pyplot.plot(xticks, data_set2, label=data_label2)
    for a, b in zip(xticks, data_set1):
        pyplot.text(a, b, str(round(b, 2)))
    for a, b in zip(xticks, data_set2):
        pyplot.text(a, b, str(round(b, 2)))
    pyplot.legend()
    pyplot.show()


def select_knn_model(x_train, x_validation, y_train, y_validation, list_k,
                     visualize=False):
    """
    This function selects and trains a KNN model.
    """
    errors_validation = []
    errors_train = []
    print('K-nearest neighbors:')
    current_knn_model = None
    current_knn_error = None
    for k in list_k:
        knc = KNeighborsClassifier(k)
        knc.fit(x_train, y_train)
        errors_validation.append(1 - knc.score(x_validation, y_validation))
        errors_train.append(1 - knc.score(x_train, y_train))

        print('Current k is {}'.format(k))
        print('and the validation '
              'error is: {}'.format(errors_validation[k - 1]))
        print('and the train '
              'error is: {}'.format(errors_train[k - 1]))

        if (not current_knn_model) or \
                current_knn_error >= errors_validation[k - 1]:
            current_knn_model = knc
            current_knn_error = errors_validation[k - 1]
    print('')
    if visualize:
        knn_visualizer('KNN plot graph', 'k', 'Error', list_k,
                       errors_validation, errors_train, 'Validation', 'Train')

    return current_knn_model


if __name__ == '__main__':
    all_features, data_train, data_test, data_validation, labels_train, \
        labels_test, labels_validation = load_data('clean_real.txt',
                                                   'clean_fake.txt')
    sensible_depth = [3, 6, 9, 12, 15]
    best_tree = select_tree_model(data_train, data_validation,
                                  labels_train, labels_validation,
                                  sensible_depth)
    result = best_tree.predict(data_test)

    print('The accuracy of the best '
          'decision tree over test set'
          ' is: {}\n'.format(accuracy_score(labels_test, result)))
    decision_tree_visualizer(best_tree, 'decision_tree_graph', 2)

    nodes_of_interest = [0, best_tree.tree_.children_left[0],
                         best_tree.tree_.children_right[0]]

    ig = compute_information_gain(best_tree, nodes_of_interest, all_features)
    print('Keyword used to split the data, and the corresponding IG is:')
    for key in ig:
        print(ig[key])
    print('')

    best_knn = select_knn_model(data_train, data_validation, labels_train,
                                labels_validation, range(1, 21))
    print('The test accuracy of the best '
          'kNN model is {}'.format(best_knn.score(data_test, labels_test)))
