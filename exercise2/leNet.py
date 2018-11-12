import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class leNet(object):
    # Class that implements leNet-like CNN.
    def __init__(self,
                 learning_rate,
                 num_epochs,
                 num_filters,
                 batch_size,
                 filter_size,
                 max_pool_size,
                 model_path="./tmp/model.ckpt",
                 ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.max_pool_size = max_pool_size
        self.model_path = model_path

    def initialize(self, x_shape, y_shape):
        # Initialize the layers and weights
        self.W1 = tf.get_variable("W1", [self.filter_size, self.filter_size, 1, self.num_filters])
        self.W2 = tf.get_variable("W2", [self.filter_size, self.filter_size, self.num_filters, self.num_filters])
        self.b1 = tf.get_variable("b1", [self.num_filters], initializer=tf.constant_initializer(0.0))
        self.b2 = tf.get_variable("b2", [self.num_filters], initializer=tf.constant_initializer(0.0))

        # Initialize placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, *x_shape[1:]])
        self.y = tf.placeholder(tf.float32, shape=[None, *y_shape[1:]])

    def inference(self, X):
        # Define operations for forward propagation here
        ksize = [1, self.max_pool_size, self.max_pool_size, 1] # This should be [1, 2, 2, 1] usually
        stride_size = [1 for i in range(len(X.shape))] # This should be [1, 1, 1, 1] for mnist data
        padding = "SAME" # Set padding scheme here

        # First layer.
        conv1 = tf.nn.conv2d(X, self.W1, stride_size, padding)
        pre_activation1 = tf.nn.bias_add(conv1, self.b1)
        activation1 = tf.nn.relu(pre_activation1)
        max_pool1 = tf.nn.max_pool(activation1, ksize, stride_size, padding)

        # Second layer
        conv2 = tf.nn.conv2d(max_pool1, self.W2, stride_size, padding)
        pre_activation2 = tf.nn.bias_add(conv2, self.b2)
        activation2 = tf.nn.relu(pre_activation2)
        max_pool2 = tf.nn.max_pool(activation2, ksize, stride_size, padding)

        # Third layer (fully connected)
        flattened = tf.contrib.layers.flatten(max_pool2) # Flatten the array so it has shape (N, W*H*D, 1), right?
        activation3 = tf.contrib.layers.fully_connected(flattened, num_outputs=128)

        # Fourth layer
        logits = tf.contrib.layers.fully_connected(activation3, num_outputs=10, activation_fn=None)
        return logits


    def loss(self, logits):
        """
        Returns cross entropy loss.
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits) # loss per sample
        loss = tf.reduce_mean(cross_entropy) # Take the average of the loss per samples
        return loss


    def train(self, X_train, y_train, X_valid, y_valid):
        # Initialize parameters and placeholders
        self.initialize(X_train.shape, y_train.shape)
        # Build computational graph
        y_pred = self.inference(self.X)
        loss = self.loss(y_pred)
        # define Optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        # Initialize variables
        glob_vars = tf.initializers.global_variables()
        # define operation for computing accuracy
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # operation for cmoputing accuracy

        # Lists to store learning curve
        train_losses = []
        train_accuracies = []
        valid_accuracies = []

        # Start session
        with tf.Session() as sess:
            sess.run(glob_vars)
            for e in range(self.num_epochs):
                train_loss = 0
                for b in range(X_train.shape[0] // self.batch_size):
                    # extract batch
                    X_train_batch = X_train[self.batch_size * b : self.batch_size * (b + 1), :, :, :]
                    y_train_batch = y_train[self.batch_size * b : self.batch_size * (b + 1), :]
                    # optimize
                    _, batch_loss = sess.run([optimizer, loss], feed_dict={self.X:X_train_batch, self.y:y_train_batch})
                    train_loss += batch_loss / self.batch_size # Accumulate train loss for current epoch.
                train_losses.append(train_loss)

                # compute train and validation accuracies and store them
                train_accuracy = accuracy.eval(feed_dict={self.X:X_train, self.y:y_train})
                valid_accuracy = accuracy.eval(feed_dict={self.X:X_valid, self.y:y_valid})
                train_accuracies.append(train_accuracy)
                valid_accuracies.append(valid_accuracy)

                # print statistics
                print("Epoch %d: train_accuracy: %.4f, valid_accuracy: %.4f" % (e+1, train_accuracy, valid_accuracy))

            # Save model after done training.
            #saver = tf.train.Saver()
            #saver.save(sess, self.model_path)

        return train_losses, train_accuracies, valid_accuracies

    def eval(self, X_test, y_test):
        """
        Simple method for evaluating the trained model on the test data.
        """
        #y_pred = self.inference(self.X)
        #correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # operation for cmoputing accuracy
        #test_accuracy = accuracy.eval(feed_dict={self.X:X_test, self.y:y_test})
        pass


    def plot(self, train_losses, train_accuracies, valid_accuracies):
        """
        Simple function that plots the learning curve. The arguments are
        the outputs of self.train(). Use this for visualizing.
        """
        # Plot train loss.
        plt.figure(1)
        plt.plot(train_losses)
        plt.title("Train loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True)

        # plot train and validation accuracies.
        plt.figure(2)
        plt.plot(train_accuracies, label='train')
        plt.plot(valid_accuracies, label='validation')
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()


# Create dataset for testing.
from cnn_mnist import mnist
X_train, y_train, X_valid, y_valid, X_test, y_test = mnist()


# Testing
learning_rate=0.1
num_epochs=4
num_filters=16
batch_size=64
filter_size=3
max_pool_size=2


model = leNet(learning_rate,num_epochs,num_filters,batch_size,filter_size,max_pool_size)
model.train(X_train, y_train, X_valid, y_valid)
