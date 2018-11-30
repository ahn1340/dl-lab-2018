import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors # for colormap
from FCN import *
from loss import *

flags = tf.app.flags

flags.DEFINE_string("dataset_dir", "./data/", "Dataset directory default is data/")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("logs_path", "./logs/", "Directory name to save the log files")
flags.DEFINE_float("beta1", 0.90, "Momentum for adam")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam")
flags.DEFINE_integer("batch_size", 1, "The size of the sample batch")
flags.DEFINE_integer("img_height", 300, "Image Height")
flags.DEFINE_integer("img_width", 300, "Image Width")
flags.DEFINE_float("dropout", 1.0, "Dropout")
flags.DEFINE_float("steps_per_epoch", 500, "Steps per epoch")
flags.DEFINE_integer("max_steps", 100000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("load_Model", False, "Load Model Flag")
flags.DEFINE_string("model_path", "", "Load model from a  previous checkpoint")
flags.DEFINE_string("dataset", "CamVid", "Choose dataset, options [Camvid, ...]")
flags.DEFINE_integer("numberClasses", 12, "Number of classes to be predicted")
flags.DEFINE_string("version_net", "FCN_Seg", "Version of the net")
flags.DEFINE_boolean("Test", False, "Flag for testing")
flags.DEFINE_integer("lower_iter", 1000, "initial iteration to be tested - default 1000")
flags.DEFINE_integer("higher_iter", 40000, "final iteration to be tested - default 40000")
flags.DEFINE_string("IoU_filename", "testIoU.txt", "Nane to save the IoU and Iterattion [default is testIoU.txt]")
flags.DEFINE_integer("configuration", 4, "Set of configurations decoder [default is 4 - full decoder], other options are [1,2,3]")

FLAGS = flags.FLAGS

# Loading Class
FCN = FCN_SS()
FCN.setup_inference(FLAGS.version_net, FLAGS.img_height, FLAGS.img_width, FLAGS.batch_size, FLAGS.Test, FLAGS.numberClasses, FLAGS.dataset, FLAGS.configuration)
saver = tf.train.Saver([var for var in tf.trainable_variables()])
variables_to_restore = tf.trainable_variables()
#variables_to_restore = slim.get_variables_to_restore()
#for v in variables_to_restore:
#print(v)

imgs, label = FCN.loadTest_set(FLAGS)
#print("Test Set DONE!")
#print(imgs.shape)
#print(label.shape)
#print(np.max(imgs[0]))
#print(np.min(imgs[0]))
#print(imgs[0])

num_pics = 3 # number of pictures to show
idx = 50 # Where to start plotting

# Show ground truth and prediction
fig = plt.figure(figsize=(8, 6))
# prepare trained model
with tf.Session() as sess:
    model_path = './checkpoints/model-2000'
    print("model to be restored: ", model_path)
    saver.restore(sess, model_path)
    for i in range(num_pics):
        results = FCN.inference(imgs[idx + i], sess)
        pred = results['Mask']
        pred = pred.argmax(axis=1).reshape(300, 300)
        truth = label[idx + i].argmax(axis=1).reshape(300, 300)
        fig.add_subplot(num_pics, 2, 2*i + 1)
        plt.imshow(pred, cmap=plt.get_cmap('viridis')) # Network prediction
        plt.title("Network prediction")
        fig.add_subplot(num_pics, 2, 2*i + 2)
        plt.imshow(truth) # ground truth
        plt.title("Ground truth")
    plt.show()



