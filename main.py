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

  # convelutional layers, taking that image and transforming it in some sort of way
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'SAME')  # strides are a list of integers, a tensors = data
    x = tf.nn.bias_add(x, b)  # tuning knob, makes our model more accurate
    return tf.nn.relu(x)  # rectified linear unit, activation function, 
  # image gets hieracharly more abstract as you go in your network

def maxpool2d(x, k=2):  # pooling takes small rect blocks from the confulusional layer, little samples, pools, max pool the average of the learned linear compulation 
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')  # 4d tensor there are four variables there
  # weve created our definitions now lets create our model

def conv_net(x, weights, biases, dropout):  # x is our input, rates are our connects or synapses between layers, biases affect each of our layers in some way
      # reshape our input data so it is formated for our computational graph that we are about to create
    x = tf.reshape(x, shape=[-1,28,28,1])  # our image is 28 by 28 pixel and our width and height im going to set to one

      #convolutional layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
      #max pooling layer
    conv1 = maxpool2d(conv1, weights['wc2'], biases['bc2'])  # boom boom boom


    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  # takes our previous layer as our input
    conv2 = maxpool2d(conv2, k=2)

      # weve got both our concolutional layers now we need to create a fully connected layer
      # a fully connected layer is a generic layer, every neuron in the fully connected layer is connected to the convolutional network, just a representation of image data
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))  # this is where the actual matrix multiply happens, all that data weve been transforming this is where that actually happens, the actual classification, all that data - htis si where we combine it together
    fc1 = tf.nn.relu(fc1)  # add our activation function relu
      # apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)

      # output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out'], biases['out']))
    return out  # this is our class classification

  # create weights, weights as a pictionary
weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),  # first set of weights, a 5 by 5 convolutional, with one input  and thirty two outputs, 5 by 5 width and height, one input an image, 32 output thats the bits
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),  # 32 inputs, their is thirty two different connections its going 32 different and its splitting it into 64 thats our synaptic connections
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),  # fully connected layer
    'out': tf.Variable(tf.random_normal([1024, n_classes]))  # this is where we predict our class
}

  # construct model
pred = conv_net(x, weights, biases, keep_prob)  # keep prob is our dropout
  # define optimizer and loss, loss is our cost.... this is measuring the probability error in a classification task 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # reduce mean is synonamous with reducing loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Atom reduces the loss over a gradient descent process 

# evaluate our model
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y,1))  # we have our test data and our predicted value and we want to see the difference in that
accuracy = tf.reduce_mean(tf.case(correct_pred, tf.float32))

#initialize the variables
init = tf.initialize_all_variables()

#launch the graph
with tf.Session as sess:  # a graph is encapsulated by a session
    sess.run(init)
    step = 1
    # keep training until max iterations
    while step * batch_size < training_iters:
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        print('iteration step')
