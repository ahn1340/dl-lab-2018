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
                 model_path="/tmp/model.ckpt",
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
        flattened = tf.contrib.layers.flatten(max_pool2) # Flatten the array first.
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
        logits = self.inference(self.X)
        loss = self.loss(logits)
        #tf.add_to_collection('prediction', tf.argmax(logits, 1)) # need this for computing test accuracy later.

        # define Optimizer operation. note: this returns None, since this is an op, not a tensor.
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        # define operation for computing accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # initialize all variables
        init = tf.global_variables_initializer()

        # Lists to store learning curve
        train_losses = []
        train_accuracies = []
        valid_accuracies = []

        # Create saver object.
        saver = tf.train.Saver()

        # Start session
        with tf.Session() as sess:
            sess.run(init)
            for e in range(self.num_epochs):
                train_loss = 0
                train_accuracy = 0
                valid_accuracy = 0
                num_batches = X_train.shape[0] // self.batch_size # total number of training batches
                num_valid_batches = X_valid.shape[0] // self.batch_size # total number of validation batches
                # update setp
                for b in range(num_batches):
                    # extract batch
                    X_train_batch = X_train[self.batch_size * b : self.batch_size * (b + 1)]
                    y_train_batch = y_train[self.batch_size * b : self.batch_size * (b + 1)]
                    # optimize
                    _, batch_loss = sess.run([optimizer, loss], feed_dict={self.X:X_train_batch, self.y:y_train_batch})

                # train loss/accuracy computation step
                for b in range(num_batches):
                    # Compute loss and accuracy in a batch-by-batch manner to prevent memory overflow.
                    X_train_batch = X_train[self.batch_size * b : self.batch_size * (b + 1)]
                    y_train_batch = y_train[self.batch_size * b : self.batch_size * (b + 1)]
                    train_loss += loss.eval(feed_dict={self.X:X_train_batch, self.y:y_train_batch}) / num_batches
                    train_accuracy += accuracy.eval(feed_dict={self.X:X_train_batch, self.y:y_train_batch}) / num_batches

                # validation accuracy computation step
                for b in range(num_valid_batches):
                    X_valid_batch = X_valid[self.batch_size * b : self.batch_size * (b + 1)]
                    y_valid_batch = y_valid[self.batch_size * b : self.batch_size * (b + 1)]
                    valid_accuracy += accuracy.eval(feed_dict={self.X:X_valid_batch, self.y:y_valid_batch}) / num_valid_batches

                train_losses.append(float(train_loss))
                train_accuracies.append(float(train_accuracy))
                valid_accuracies.append(float(valid_accuracy))

                # print statistics
                print("Epoch %d: train_accuracy: %.4f, valid_accuracy: %.4f" % (e+1, train_accuracy, valid_accuracy))
                print("         training loss:", train_loss)

            # Save model after training.
            save_path = saver.save(sess, self.model_path)
            print("model saved in path:", save_path)

        return train_losses, train_accuracies, valid_accuracies

    def test_eval(self, X_test, y_test):
        """method which evaluates the model on the test dataset."""
        saver = tf.train.import_meta_graph(self.model_path + '.meta')
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))
            graph = tf.get_default_graph()
            accuracy = graph.get_tensor_by_name("accuracy:0")
            prediction = sess.run([accuracy], feed_dict={self.X:X_test, self.y:y_test})
            print("test accuracy: %.4f" % prediction[0])
            return float(prediction[0]) # change to float, because numpy.float32 is not JSON serializable.


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
