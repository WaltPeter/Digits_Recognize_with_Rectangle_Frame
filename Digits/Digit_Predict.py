from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def predict(img, output=False):
    if len(img.ravel()) > 800: 
        gray = np.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else: 
        gray = img
    test_x = np.array([gray.ravel()], dtype=int)
    test_y = np.zeros((1, 10), dtype=int)
    feed_dict = {x: test_x[0:1, :], y_true: test_y[0:1, :]}
    
    # Calculate the predicted class.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    
    if output: 
        print(cls_pred)
        plt.imshow(test_x[0].reshape(28, 28), cmap="binary")
    
    return cls_pred, gray

# Useful variable.
img_size = 28
img_size_flat = 784
img_shape = (28, 28)
num_classes = 10
num_channels = 1

# Convolutional Layer 1.
filter_size1 = 5
num_filters1 = 16

# Convolutional Layer 2.
filter_size2 = 5
num_filters2 = 36

# Fully-connected layer.
fc_size = 128

# Random parameters.
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# Create convolutional layer. 
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # size of each filter (int).
                   num_filters,        # Number of filters.
                   use_pooling=True):  # 2x2 max-pooling?

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    
    layer += biases

    if use_pooling:
        # 2x2 max-pooling.
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # ReLU. 
    layer = tf.nn.relu(layer)
    
    return layer, weights

# Flatten layer to 1-dim before pass to Fully-connected layer. 
def flatten_layer(layer):
    # Get input layer shape. 
    layer_shape = layer.get_shape() # [num_images, img_height, img_width, num_channels]

    num_features = layer_shape[1:4].num_elements() # img_height * img_width * num_channels
    
    # Reshape. 
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

# Fully-Connected Layer. 
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num of inputs from prev layer.
                 num_outputs,    # Num of outputs.
                 use_relu=True): # ReLU?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Matrix multiplication then add the biases.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# Placeholders.

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="x")
print(type(x), x.shape)

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) # -1 for auto dimension. 
print(type(x_image), x_image.shape)

# Output matrix
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true') # None for auto dimension. 

# Class
y_true_cls = tf.argmax(y_true, axis=1) # argmax get max 

# Convolutional Layer 1. 
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, 
                                            num_input_channels=num_channels, 
                                            filter_size=filter_size1, 
                                            num_filters=num_filters1,
                                            use_pooling=True)

# Convolutional Layer 2. 
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

# Flatten layer. 
layer_flat, num_features = flatten_layer(layer_conv2)

# Fully-Connected Layer 1. 
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

# Fully-Connected Layer 2 (Output). 
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

# Final Output. 
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost function. 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Adam Optimizer. 
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ### Saver

saver = tf.train.Saver()
save_dir = "Checkpoints/" ## WARNING: Relative path for main script. Digits/

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')


# ### Recover

session = tf.Session()
session.run(tf.global_variables_initializer())

saver.restore(sess=session, save_path=save_path)



test_batch_size = 256

def get_accuracy(input_images, 
                 input_labels, 
                 cls_true, 
                 show_example_errors=False,
                 show_confusion_matrix=False):

    num_test = len(input_labels)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)
        
        images = input_images[i:j, :]
        labels = input_labels[i:j, :]
        #print(images.shape, labels.shape)

        feed_dict = {x: images, y_true: labels}

        # Calculate the predicted class.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    # Correct list. 
    correct = np.zeros(num_test, dtype=np.int)
    for a in np.arange(0,num_test):
        if (cls_pred[a] == cls_true[a]): 
            correct[a] = 1 

    correct_sum = correct.sum()
    
    # Accurancy
    acc = float(correct_sum) / num_test
    
    return acc


# ### Real Stuff
predict(cv2.imread("D:\Deep Learning\My Dataset\MNIST\letter2.png"), output=False) 


# ### DON'T FORGET TO CLOSE SESSION!!

#session.close()

