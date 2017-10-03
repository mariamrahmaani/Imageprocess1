######################################################################
## This code is written as part of:
## Creative Applications of Deep Learning w/ Tensorflow. Kadenze, Inc. class
## Part of the code provided by class author and the other part is written to solve each problem
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

def get_celeb_images():
    print('##Start get_celeb_images##')
    dirname="C:\\Users\\rahmanim\\Documents\\1TF-Gits\\ImagesFromTheInternet"
    filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]
    # Make sure we have exactly 100 image files!
    filenames = filenames[:100]
    assert(len(filenames) == 100)
    #print(filenames)
    # Read every filename as an RGB image
    imgs = [plt.imread(fname)[..., :3] for fname in filenames]
    # Crop every image to a square
    imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs]
    # Then resize the square image to 100 x 100 pixels; mode='reflect'
    imgs = [resize(img_i, (100, 100), mode='reflect') for img_i in imgs]
    # Finally make our list of 3-D images a 4-D array with the first dimension the number of images:
    imgs = np.array(imgs).astype(np.float32)
    # Plot the resulting dataset:
    # Make sure you "run" this cell after you create your `imgs` variable as a 4-D array!
    # Make sure we have a 100 x 100 x 100 x 3 dimension array
    assert(imgs.shape == (100, 100, 100, 3))
    plt.figure(figsize=(10, 10))
    plt.imshow(utils.montage(imgs, saveto='dataset.png'))
    plt.title('Dataset of 100 images')
    #plt.show()
    return imgs
	
	
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
	print('##Start linear##')
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
	print('##Start image_prep##')
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
    print('##split_image##')
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
	print('##Start mul_placeholder##')
	x = tf.placeholder(tf.float32, shape=(1024, 1024))
	y = tf.matmul(x, x)
	with tf.Session() as sess:
		rand_array = np.random.rand(1024, 1024)
		print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
	return
	
	
	
def normalize_std_score(xs):
    print('##Start normalize_std_score##')
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
    print('##Start normalize_feature_scaling##')
    norm_ys = ys / 255.0
    print('np.min(norm_ys): ',np.min(norm_ys), 'np.max(norm_ys): ', np.max(norm_ys))
    return norm_ys
	
	
	
def create_nn():
    print('##Start create_nn##')
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
    return Y_pred
   
   
   
def build_model(xs, ys, n_neurons, n_layers, activation_fn,
                final_activation_fn, cost_type):
    print('##Start build_model##')
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    if xs.ndim != 2:
        raise ValueError(
            'xs should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
    if ys.ndim != 2:
        raise ValueError(
            'ys should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
        
    n_xs = xs.shape[1]
    n_ys = ys.shape[1]
    
    X = tf.placeholder(name='X', shape=[None, n_xs],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, n_ys],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = linear(
        current_input, n_ys,
        activation=final_activation_fn,
        name='pred')[0]
    
    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')
    
    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}

	
	
	
def train(imgs,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=10,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):
    print('##Start train##')
    N, H, W, C = imgs.shape
    all_xs, all_ys = [], []
    for img_i, img in enumerate(imgs):
        xs, ys = split_image(img)
        all_xs.append(np.c_[xs, np.repeat(img_i, [xs.shape[0]])])
        all_ys.append(ys)
    xs = np.array(all_xs).reshape(-1, 3)
    xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
    ys = np.array(all_ys).reshape(-1, 3)
    ys = ys / 127.5 - 1

    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn,
                            cost_type)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])
        sess.run(tf.global_variables_initializer())
        gifs = []
        costs = []
        step_i = 0
        for it_i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i * batch_size:
                              (batch_i + 1) * batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost

            print('iteration {}/{}: cost {}'.format(
                    it_i + 1, n_iterations, training_cost / n_batches))

            # Also, every 20 iterations, we'll draw the prediction of our
            # input xs, which should try to recreate our image!
            if (it_i + 1) % gif_step == 0:
                costs.append(training_cost / n_batches)
                ys_pred = model['Y_pred'].eval(
                    feed_dict={model['X']: xs}, session=sess)
                img = ys_pred.reshape(imgs.shape)
                gifs.append(img)
    return gifs   
		
		
		
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
y_pred = create_nn()
print(y_pred)

## Write the cost function (Put part 3 to code)

celeb_imgs = get_celeb_images()
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(celeb_imgs).astype(np.uint8))
#plt.show()
# It doesn't have to be 100 images, explore!
imgs = np.array(celeb_imgs).copy()

# Change the parameters of the train function and
# explore changing the dataset
gifs = train(imgs=imgs)
montage_gifs = [np.clip(utils.montage((m * 127.5) + 127.5), 0, 255).astype(np.uint8) for m in gifs]
plt.imshow(utils.montage(montage_gifs, saveto='montage_gifs.gif'))

