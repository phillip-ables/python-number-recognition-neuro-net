  # https://www.youtube.com/watch?v=qVwm-9P609I&list=PLHy7B0Rk5zXLfY4Nz1sDiql4m2Kv0_B-W&index=19&t=15s
from __future__ import print_function  # use python 2 print function
import tensorflow as tf

  # import data
from tensorflow.examples.tutorials.mnist import input_data  #pull our images from the web stor in folder and format so we can use it in our code
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)  # one_hot is a way of representing data so that it is more machine readable

  # hyperparameters - tunning nobs
learning_rate = 0.001  # activation function at each level greater the learning rate the faster it trains lower the more accuracy
training_iters = 200000  # the more iteration the better our models going to train
batch_size = 128  # we have a hundred and 28 samples, the batch size is what we train
display_step = 10  # how ofter do we want to display what were training

  # network parameters
n_input = 784  # what is the size of our image, 28 by 28 is a 784
n_classes = 10  # number of classes means we have ten digits
dropout = 0.75  # technuiqe awsome method that prevents overtraining by randomly turning off neuro paths so training is more generalized by forcing them to find new paths

  # placeholder how were going to get the data in there
x = tf.placeholder(tf.float32, [None, n_input])  # this is our gateway one for the image and one for the lable, both fed in at the same time,
y = tf.placeholder(tf.float32, [None, n_classes])  
keep_prob = tf.placeholder(tf.float32)  # tf placeholder object represents gateways, its a gateway into our computational graph. so thats how the data flows into our graph


