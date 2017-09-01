import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda ')
# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!") 
# This cell includes the provided libraries from the zip file
try:
    from libs import utils
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.")
# We'll tell matplotlib to inline any drawn figures like so:
# %matplotlib inline
plt.style.use('ggplot')
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
#plt.show()

## Get the mean image using TensoreFlow

sess = tf.Session()
# Now create an operation that will calculate the mean of your images
mean_img_op = tf.reduce_mean(imgs,0)
# And then run that operation using your session
mean_img = sess.run(mean_img_op)
# Then plot the resulting mean image:
# Make sure the mean image is the right size!
assert(mean_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(mean_img)
plt.imsave(arr=mean_img, fname='mean.png')
plt.show()


