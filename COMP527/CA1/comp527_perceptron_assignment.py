import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class Perceptron:

    def __init__(self, learning_rate=1, n_iters=20):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights = self.weights + (update * x_i)
                self.bias = self.bias + update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def a_value(self, x):
        return np.dot(x, self.weights) + self.bias

    def _step_func(self, x):
        return np.where(x >= 0, 1, 0)


class PerceptronL2Regularized:

    def __init__(self, learning_rate=1, n_iters=20, r_c=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._step_func
        self.weights = None
        self.bias = None
        self.r_c = r_c

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights = (1 - 2 * self.r_c) * self.weights + (update * x_i)
                self.bias = self.bias + update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def a_value(self, x):
        return np.dot(x, self.weights) + self.bias

    def _step_func(self, x):
        return np.where(x >= 0, 1, 0)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def get_class_pairs(class_1, class_2, positive_class):
    np.random.seed(28)
    xy_train = np.concatenate((class_1, class_2), axis=0)
    np.random.shuffle(xy_train)

    y_train = [1 if i == positive_class else 0 for i in xy_train[:, 4:]]
    x_train = xy_train[:, :4]

    return x_train, y_train


def train_perceptron(x_train, y_train):
    percept = Perceptron(learning_rate=1, n_iters=20)
    percept.fit(x_train, y_train)
    return percept


def train_perceptron_regularized(x_train, y_train, r_coefficient):
    percept = PerceptronL2Regularized(learning_rate=1, n_iters=20, r_c=r_coefficient)
    percept.fit(x_train, y_train)
    return percept


def classify(x_test, perceptron):
    return perceptron.predict(x_test)


def multi_classify(x_test, p1, p2, p3, y_test):
    _sum = 0
    for i, j in enumerate(x_test):
        class_1 = p1.a_value(x_test[i])
        class_2 = p2.a_value(x_test[i])
        class_3 = p3.a_value(x_test[i])
        true = y_test[i][0]
        guess = ''
        if class_1 == max(class_1, class_2, class_3):
            guess = 'class-1'
        if class_2 == max(class_1, class_2, class_3):
            guess = 'class-2'
        if class_3 == max(class_1, class_2, class_3):
            guess = 'class-3'

        if true == guess:
            _sum = _sum + 1

    print('%.2f' % (_sum / len(y_test)))


def display_result(classes, _accuracy):
    print('Class ' + classes + ' data accuracy - ' + '%.2f' % _accuracy)


def get_multi_class(_data, positive_class):
    np.random.seed(28)
    np.random.shuffle(_data)
    y_train = [1 if i == positive_class else 0 for i in _data[:, 4:]]
    x_train = _data[:, :4]
    return x_train, y_train


if __name__ == '__main__':
    train_path = 'train.data'
    test_path = 'test.data'

    data = pd.read_csv(train_path, header=None).values
    test_data = pd.read_csv(test_path, header=None).values

    # train split
    class1_xy_train = data[:40, :]
    class2_xy_train = data[40:80, :]
    class3_xy_train = data[80:, :]

    # test split
    class1_xy_test = test_data[:10, :]
    class2_xy_test = test_data[10:20, :]
    class3_xy_test = test_data[20:, :]

    # %% BINARY CLASSIFICATION (1 v 1)
    # 1-2
    a, b = get_class_pairs(class1_xy_train, class2_xy_train, 'class-1')
    p = train_perceptron(a, b)
    display_result('1 v 2 train', accuracy(b, classify(a, p)))

    a, b = get_class_pairs(class1_xy_test, class2_xy_test, 'class-1')
    display_result('1 v 2 test', accuracy(b, classify(a, p)))

    # 1-3
    a, b = get_class_pairs(class1_xy_train, class3_xy_train, 'class-1')
    p = train_perceptron(a, b)
    display_result('1 v 3 train', accuracy(b, classify(a, p)))

    a, b = get_class_pairs(class1_xy_test, class3_xy_test, 'class-1')
    display_result('1 v 3 test', accuracy(b, classify(a, p)))

    # 2-3
    a, b = get_class_pairs(class2_xy_train, class3_xy_train, 'class-2')
    p = train_perceptron(a, b)
    display_result('2 v 3 train', accuracy(b, classify(a, p)))

    a, b = get_class_pairs(class2_xy_test, class3_xy_test, 'class-2')
    display_result('2 v 3 test', accuracy(b, classify(a, p)))
    print('\n')

    # %% MULTI-CLASS CLASSIFICATION (1 v REST)
    # 1-23
    atrain_1, btrain_1 = get_multi_class(data, 'class-1')
    p_1 = train_perceptron(atrain_1, btrain_1)

    # 2-13
    atrain_2, btrain_2 = get_multi_class(data, 'class-2')
    p_2 = train_perceptron(atrain_2, btrain_2)

    # 3-12
    atrain_3, btrain_3 = get_multi_class(data, 'class-3')
    p_3 = train_perceptron(atrain_3, btrain_3)

    print('Train set accuracy multiclass')
    multi_classify(data[:, :4], p_1, p_2, p_3, data[:, 4:])
    print('Test set accuracy multiclass')
    multi_classify(test_data[:, :4], p_1, p_2, p_3, test_data[:, 4:])

    print('\n')

    # %% MULTI-CLASS CLASSIFICATION WITH REGULARIZATION (1 v REST)
    for r in [0.01, 0.1, 1, 10, 100]:
        print('current lambda value - ' + str(r))
        # 1-23
        a, b = get_multi_class(data, 'class-1')
        pr_1 = train_perceptron_regularized(a, b, r)

        # 2-13
        a, b = get_multi_class(data, 'class-2')
        pr_2 = train_perceptron_regularized(a, b, r)

        # 3-12
        a, b = get_multi_class(data, 'class-3')
        pr_3 = train_perceptron_regularized(a, b, r)

        print('Train set accuracy')
        multi_classify(data[:, :4], pr_1, pr_2, pr_3, data[:, 4:])
        print('Test set accuracy')
        multi_classify(test_data[:, :4], pr_1, pr_2, pr_3, test_data[:, 4:])
        print('\n')
