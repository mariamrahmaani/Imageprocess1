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
	plt.show()
	#save this image as "reference.png"
	plt.imsave(fname=dirname+"reference.png", arr=img)
	return

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
    return xs, ys
	
def mul_placeholder():

	x = tf.placeholder(tf.float32, shape=(1024, 1024))
	y = tf.matmul(x, x)
	with tf.Session() as sess:
		rand_array = np.random.rand(1024, 1024)
		print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
	return
	
#####################

#linear(x, 2, name=None, activation=None, reuse=None)
#plot_relu_sigmoid_tanh()
mul_placeholder()
image_prep("girl.jpg", ".\\img\\")

