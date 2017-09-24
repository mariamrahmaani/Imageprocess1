######################################################################
## This code is written as part of assignments for a class:
## Creative Applications of Deep Learning w/ Tensorflow. Kadenze, Inc.
## Part of the code provided by class author and the other part is written to solve each assignment
######################################################################
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher. Try installing the Python 3.5 version of anaconda ')

try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
except ImportError:
	print('You are missing some packages! We will try installing them before continuing!')
	#import pip
	#pip.main(['install', "numpy>=1.11.0"])
	#!pip install "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-image>=0.11.3" "scikit-learn>=0.17" "scipy>=0.17.0"
	import os
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.transform import resize
	from skimage import data
	from scipy.misc import imresize
	print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")

try:
	from libs import utils, gif 
	import IPython.display as ipyd
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")

		  
		    
##########################################
## DEF FUNC BEGIN
##########################################

# We'll tell matplotlib to inline any drawn figures like so:
def plot_relu_sigmoid_tanh():
	plt.style.use('ggplot')
	xs = np.linspace(-6, 6, 100)
	plt.plot(xs, np.maximum(xs, 0), label='relu')
	plt.plot(xs, 1 / (1 + np.exp(-xs)), label='sigmoid')
	plt.plot(xs, np.tanh(xs), label='tanh')
	plt.xlabel('Input')
	plt.xlim([-6, 6])
	plt.ylabel('Output')
	plt.ylim([-1.5, 1.5])
	plt.title('Common Activation Functions/Nonlinearities')
	plt.legend(loc='lower right')
	plt.savefig('Relu-Sigmoid-tanh')
	plt.show()
	return


def linear(x, n_output, name=None, activation=None, reuse=None):
	"""Fully connected layer Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply
Returns
    -------
    op : tf.Tensor
        Output of fully connected layer.
	"""
	if len(x.get_shape()) != 2:
		x = flatten(x, reuse=reuse)
	n_input = x.get_shape().as_list()[1]
	with tf.variable_scope(name or "fc", reuse=reuse):
		W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
		h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)
	if activation:
            h = activation(h)
	print('h=', h,'W= ',W)
	return h, W

	
		
def image_prep(filename, dirname):
	# First load an image
	img = plt.imread(dirname+filename)
	# Be careful with the size of your image.
	# Try a fairly small image to begin with,
	# then come back here and try larger sizes.
	img = imresize(img, (100, 100))
	plt.figure(figsize=(5, 5))
	plt.imshow(img)
	#plt.show()
	#save this image as "reference.png"
	plt.imsave(fname=dirname+"reference.png", arr=img)
	return (img)

def split_image(img):
    # We'll first collect all the positions in the image in our list, xs
    xs = []

    # And the corresponding colors for each of these positions
    ys = []

    # Now loop over the image
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # And store the inputs
            xs.append([row_i, col_i])
            # And outputs that the network needs to learn to predict
            ys.append(img[row_i, col_i])
    # we'll convert our lists to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    print('xs',xs)
    print('ys',ys)
    print('xs.shape: ',xs.shape)
    print('ys.shape: ',ys.shape)
    return xs, ys
	
def mul_placeholder():

	x = tf.placeholder(tf.float32, shape=(1024, 1024))
	y = tf.matmul(x, x)
	with tf.Session() as sess:
		rand_array = np.random.rand(1024, 1024)
		print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
	return
	
def normalize_std_score(xs):
    sess = tf.Session()
    xs_op = tf.reduce_mean(xs, axis=0)
    mean_xs = sess.run(xs_op)
    std_xs_op = tf.abs(tf.sqrt(tf.reduce_mean((xs-mean_xs)*(xs-mean_xs), axis=0)))
    std_xs = sess.run(std_xs_op)
    norm_xs = (xs - mean_xs)/std_xs
    print('norm_value: ',norm_xs)
    print('np.min(norm_value): ', np.min(norm_xs) , 'np.max(norm_value): ', np.max(norm_xs))
    assert(np.min(norm_xs) > -3.0 and np.max(norm_xs) < 3.0)
    return norm_xs
	
def normalize_feature_scaling(ys):
    norm_ys = ys / 255.0
    print('np.min(norm_ys): ',np.min(norm_ys), 'np.max(norm_ys): ', np.max(norm_ys))
    return norm_ys
	
def create_nn():
    tf.reset_default_graph()
    
	#Create a placeholder of None x 2 dimensions and dtype tf.float32
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2))
	
    # Create the placeholder, Y, with 3 output dimensions instead of 2.
    # This will be the output of the network, the R, G, B values.
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    
	# create 6 hidden layers. create a variable to say how many neurons we want for each of the layers
    # (20 to begin with)
    n_neurons = 20
    # Create the first linear + nonlinear layer which will
    # take the 2 input neurons and fully connects it to 20 neurons.
    # Use the `utils.linear` function to do this just like before,
    # but also remember to give names for each layer, such as
    # "1", "2", ... "5", or "layer1", "layer2", ... "layer6".
    h1, W1 = linear(X, n_neurons, name="layer1", activation=None, reuse=None)
    # Create another one:
    h2, W2 = linear(h1, n_neurons, name="layer2", activation=None, reuse=None)
    # and four more (or replace all of this with a loop if you can!):
    h3, W3 = linear(h2, n_neurons, name="layer3", activation=None, reuse=None)
    h4, W4 = linear(h3, n_neurons, name="layer4", activation=None, reuse=None)
    h5, W5 = linear(h4, n_neurons, name="layer5", activation=None, reuse=None)
    h6, W6 = linear(h5, n_neurons, name="layer6", activation=None, reuse=None)
    # Now, make one last layer to make sure your network has 3 outputs:
    Y_pred, W7 = linear(h6, 3, activation=None, name='pred')
    assert(X.get_shape().as_list() == [None, 2])
    assert(Y_pred.get_shape().as_list() == [None, 3])
    assert(Y.get_shape().as_list() == [None, 3])
    return
   
   
##########################################
## DEF FUNC END
##########################################

#linear(x, 2, name=None, activation=None, reuse=None)
#plot_relu_sigmoid_tanh()
#mul_placeholder()

## Change the image size to a 100X100 figure
refimage = image_prep("girl.jpg", ".\\img\\")

## Put pixel locations and pixel values in xs and xy
xs, ys = split_image(refimage)

## normalize xs and xy
norm_xs = normalize_std_score(xs.astype(np.float32))
norm_ys = normalize_feature_scaling(ys.astype(np.float32))

## Create the neural network
create_nn()