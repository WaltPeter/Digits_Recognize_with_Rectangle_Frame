from __future__ import print_function

print(
"                    .,,,               \n",
"                .,,,,,,,,,,.           \n",
"             .,,,,,,,,,,,,,,,,,        \n",
"          ,,,,,,,,,,,*/*,,,,,,,,,,     \n",
"      .,,,,,,,,,,*****/(((/,,,,,,,,,,. \n",
"   ,,,,,,,,,,,********/(((((((*,,,,***.\n",
"./,,,,,,,,***********/.*((((((((/*****.\n",
".((((*,**************/    ./((((/****/.\n",
".(((((********((/****/        *(/*/.   \n",
" .((((****./((((/****,,                \n",
"    .//.   /((((/*,,,,,,,,.            \n",
"           /((((((/*,,,,,**,           \n",
"           /((((((((((*****,           \n",
"           /((((((((((*****,           \n",
"           /((((/**/((***              \n",
"           /((((/****/                 \n",
"           /((((/****/                 \n",
"           /((((/****/                 \n",
"           *((((/****/                 \n",            
"              *(/*                     \n\n", 
"     Initialing Tensorflow model.      \n\n"   
)
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from six.moves import cPickle as pickle
import cv2

# In[2]:


pickle_file = "D:/Computer Vision/RM/Test5/Dataset_new.pkl" #'D:/Deep Learning/My Dataset/mnist.pkl'

with open(pickle_file, 'rb') as f:
    #train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    all_set, _ = pickle.load(f, encoding="Latin1")
    f.close()
    
'''
train_x = train_set[0]
train_y_cls = train_set[1]
valid_x = valid_set[0]
valid_y_cls = valid_set[1]
test_x = test_set[0]
test_y_cls = test_set[1]
data_x = np.concatenate((train_x, valid_x[0:5000]), axis=0)
data_y_cls = np.concatenate((train_y_cls, valid_y_cls[0:5000]), axis=0)
valid_x = valid_x[5000:]
valid_y_cls = valid_y_cls[5000:]
print("Train set: ")
print(train_x.shape)
print(train_y_cls.shape)
print("\n")
'''

all_x = all_set[0]
all_y = all_set[1]
data_x = all_x[0:112000]
data_y_cls = all_y[0:112000]
valid_x = all_x[112000:126000]
valid_y_cls = all_y[112000:126000]
test_x = all_x[126000:]
test_y_cls = all_y[126000:]

#print("Train set: ")
#print(data_x.shape)
#print(data_y_cls.shape)
#print("\n")
#print("Valid set: ")
#print(valid_x.shape)
#print(valid_y_cls.shape)
#print("\n")
#print("Test set: ")
#print(test_x.shape)
#print(test_y_cls.shape)


# In[3]:


# Useful variable.
img_size = 28
img_size_flat = 784
img_shape = (28, 28)
num_classes = 10
num_channels = 1


# In[4]:


# Tokenize y_cls to y matrix. 
def tokenize_y(y_cls): 
    y_mat = np.zeros((len(y_cls), num_classes), dtype=np.int)
    for i in np.arange(0, len(y_cls)): 
        y_mat[i][y_cls[i]] = 1
    return y_mat


# In[5]:


data_y = tokenize_y(data_y_cls)
test_y = tokenize_y(test_y_cls)
valid_y = tokenize_y(valid_y_cls)


# In[6]:


print(data_y[0], data_y_cls[0])
print(data_y.shape, test_y.shape)


# In[7]:


# Sample img. 
im = data_x[0].reshape(28, 28)
plt.imshow(im, cmap="binary")


# In[8]:


# Display purpose
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


# ## Model Structure
# Input Image: (28, 28, 1)<br>
# &nbsp;↓<br>
# Convolutional Layer 1:
# 16 x Filter-Weights (5, 5)<br>
# &nbsp;↓<br>
# Output: (14, 14, 16)<br> 
# &nbsp;↓<br>
# Convolutional Layer 2:
# 16x36 x Filter-weights (5, 5)<br>
# &nbsp;↓<br> 
# Output: (7, 7, 36)<br>
# &nbsp;↓<br>
# Fully-connected layer: (128 neurons)<br>
# &nbsp;↓<br> 
# Output layer (Fully-connected layer 2): (10 neurons)<br><br>

# In[9]:


# Convolutional Layer 1.
filter_size1 = 5
num_filters1 = 16

# Convolutional Layer 2.
filter_size2 = 5
num_filters2 = 36

# Fully-connected layer.
fc_size = 128


# In[10]:


# Random parameters.
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# ### Worker Functions

# In[11]:


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


# In[12]:


# Flatten layer to 1-dim before pass to Fully-connected layer. 
def flatten_layer(layer):
    # Get input layer shape. 
    layer_shape = layer.get_shape() # [num_images, img_height, img_width, num_channels]

    num_features = layer_shape[1:4].num_elements() # img_height * img_width * num_channels
    
    # Reshape. 
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


# In[13]:


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


# ### Placeholders

# In[14]:


# Placeholders.

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="x")
print(type(x), x.shape)

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) # -1 for auto dimension. 
print(type(x_image), x_image.shape)

# Output matrix
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true') # None for auto dimension. 

# Class
y_true_cls = tf.argmax(y_true, axis=1) # argmax get max 


# ### Construct Model

# In[15]:


# Convolutional Layer 1. 
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, 
                                            num_input_channels=num_channels, 
                                            filter_size=filter_size1, 
                                            num_filters=num_filters1,
                                            use_pooling=True)


# In[16]:


layer_conv1


# In[17]:


# Convolutional Layer 2. 
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)


# In[18]:


layer_conv2


# In[19]:


# Flatten layer. 
layer_flat, num_features = flatten_layer(layer_conv2)


# In[20]:


print(layer_flat, num_features)


# In[21]:


# Fully-Connected Layer 1. 
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)


# In[22]:


layer_fc1


# In[23]:


# Fully-Connected Layer 2 (Output). 
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)


# In[24]:


layer_fc2


# In[25]:


# Final Output. 
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)


# ### Cost Function Optimization

# In[26]:


# Cost function. 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Adam Optimizer. 
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[27]:


#accurancy = metrics.accuracy_score(y_pred_cls, y_true_cls)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ### Saver

# In[28]:


saver = tf.train.Saver()
save_dir = "D:/Computer Vision/RM/Digits/Checkpoints/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')


# ### Recover

# In[29]:


session = tf.Session()
session.run(tf.global_variables_initializer())

saver.restore(sess=session, save_path=save_path)


# ### Performance

# In[30]:


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

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
    
    return acc

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False): 
    acc = get_accuracy(test_x, test_y, test_y_cls, show_example_errors, show_confusion_matrix)
    print("Test accuracy: {0:>6.2%}".format(acc))


# In[31]:


def plot_example_errors(cls_pred, correct):
    
    # This function is called from print_test_accuracy(). #

    # Incorrect list. 
    incorrect = np.bitwise_not(correct)+2
    
    images = [test_x[i] for i, x in enumerate(incorrect) if x==1]
    cls_pred = [cls_pred[i] for i, x in enumerate(incorrect) if x==1]
    cls_true = [test_y_cls[i] for i, x in enumerate(incorrect) if x==1]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# In[32]:


def plot_confusion_matrix(cls_pred):
    
    # This is called from print_test_accuracy(). #
    
    # Get confusion matrix using sklearn.
    cm = confusion_matrix(y_true=test_y_cls, y_pred=cls_pred)
    
    print(cm)

    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# ### Real Stuff

# In[33]:


print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)


# In[34]:

img = cv2.imread("D:/Computer Vision/RM/Test5/3/3_0.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.bitwise_not(img)
test_x = np.array([img.ravel()], dtype=int)
feed_dict = {x: test_x[0:1, :], y_true: test_y[100:101, :]}

# Calculate the predicted class.
cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)

#print(cls_pred)
#plt.imshow(test_x[0].reshape(28, 28), cmap="binary")


def predict(img, output=False): 
    if len(img.ravel()) > 800: 
        gray = np.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else: 
        gray = np.bitwise_not(img)
    test_x = np.array([gray.ravel()], dtype=int)
    test_y = np.zeros((1, 10), dtype=int)
    feed_dict = {x: test_x[0:1, :], y_true: test_y[0:1, :]}
    
    # Calculate the predicted class.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    proba = session.run(layer_fc2, feed_dict=feed_dict)
    
    if output: 
        print(cls_pred)
        plt.imshow(test_x[0].reshape(28, 28), cmap="binary")
    
    return cls_pred, proba, gray


# ### DON'T FORGET TO CLOSE SESSION!!

# In[35]:


#session.close()

