import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    encoded = np.zeros((n_labels, n_unique_labels))
    encoded[np.arange(n_labels), labels] = 1
    return encoded


def read_dataframe():
    df = pd.read_csv("data/sonar-data.csv")
    X = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return X, Y


def multilayer_perceptron(x, weights, biases):

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    return tf.matmul(layer_4, weights['out']) + biases['out']


if __name__ == "__main__":
    X, Y = read_dataframe()
    X, Y = shuffle(X, Y, random_state=1)

    # 20% Testing data
    train_x, test_x, train_y , test_y = train_test_split(X, Y, test_size=0.2, random_state=415)

    learning_rate = 0.3
    epochs = 1000
    cost_history = np.empty(shape=[1], dtype=float)
    n_dim = X.shape[1]

    # 2 Classes: Mine or Rock
    n_class = 2
    model_path = "model"
    print(n_dim)

    # Number of hidden layers and number of neurons for each layer
    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60

    x = tf.placeholder(tf.float32, [None, n_dim])
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))
    y_ = tf.placeholder(tf.float32, [None, n_class])

    # Weights and biases for each layer

    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
    }

    biases = {
        'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_class])),

    }

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    y = multilayer_perceptron(x, weights, biases)

    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    mse_history = []
    accuracy_history = []
    with tf.Session() as session:
        session.run(init)
        for epoch in range(epochs):
            session.run(training_step, feed_dict={x: train_x, y_: train_y})
            cost = session.run(cost_function, feed_dict={x: train_x, y_: train_y})
            cost_history = np.append(cost_history, cost)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

            pred_y = session.run(y, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = session.run(mse)
            mse_history.append(mse_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_history.append(accuracy)

            print('Epoch: {} - Cost: {} - MSE: {} - ACC: {}'.format(epoch, cost, mse_, accuracy))

        save_path = saver.save(session, model_path)
        print("Saved model in {}".format(save_path))
        plt.plot(mse_history, 'r')
        plt.show()
        plt.plot(accuracy_history)
        plt.show()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Test Accuracy: ", session.run(accuracy, feed_dict={x: test_x, y_: test_y}))

        pred_y = session.run(y, feed_dict={x: test_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        print("MSE: {}".format(session.run(mse)))

