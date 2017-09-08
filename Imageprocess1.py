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
plt.title('Dataset of 100 images')
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
plt.title('Mean of 100 images')
plt.imsave(arr=mean_img, fname='mean.png')
#plt.show()

## calculate standard deviation
## Create a tensorflow operation to give you the standard deviation
# First compute the difference of every image with a
# 4 dimensional mean image shaped 1 x H x W x C
mean_img_4d = np.array(mean_img)
print("mean_img_4d")
print(mean_img_4d.shape)

subtraction = imgs - mean_img_4d
# Now compute the standard deviation by calculating the
# square root of the expected squared differences
std_img_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, axis=0))
# Now calculate the standard deviation using your session
std_img = sess.run(std_img_op)

# Then plot the resulting standard deviation image:
# Make sure the std image is the right size!
assert(std_img.shape == (100, 100) or std_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
std_img_show = std_img / np.max(std_img)
plt.title('Standard Deviation of 100 images')
#print(np.max(std_img))
#print(std_img.shape)
plt.imshow(std_img_show)
plt.imsave(arr=std_img_show, fname='std.png')
#plt.show()

## Normalize the Dataset
norm_imgs_op = tf.abs((imgs - mean_img_4d)/std_img)

norm_imgs = sess.run(norm_imgs_op)
#print(np.min(norm_imgs), np.max(norm_imgs))
#print(imgs.dtype)

# Plot the resulting normalized dataset montage:
# Make sure we have a 100 x 100 x 100 x 3 dimension array
assert(norm_imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10, 10))
plt.title('Normalized image')
plt.imshow(utils.montage(norm_imgs, 'normalized.png'))
# Apply another type of normalization to 0-1 just for the purposes of plotting the image. 
# If we didn't do this, the range of our values would be somewhere between -1 and 1, and matplotlib would not be able to interpret the entire range of values. 
# By rescaling our -1 to 1 valued images to 0-1, we can visualize it better.
norm_imgs_show = (norm_imgs - np.min(norm_imgs)) / (np.max(norm_imgs) - np.min(norm_imgs))
plt.figure(figsize=(10, 10))
plt.title('Normalized image with 0-1 rescaling')
plt.imshow(utils.montage(norm_imgs_show, 'normalized.png'))
plt.show()

## Convolve with a Gabor filter
ksize = 32
kernel = np.concatenate([utils.gabor(ksize)[:, :, np.newaxis] for i in range(3)], axis=2)

print(kernel.shape)
                     
# Now make the kernels into the shape: [ksize, ksize, 3, 1]:
kernel_4d = np.reshape(kernel, (ksize, ksize, 3, 1))

print(kernel_4d.shape)

assert(kernel_4d.shape == (ksize, ksize, 3, 1))

plt.figure(figsize=(5, 5))
plt.imshow(kernel_4d[:, :, 0, 0], cmap='gray')
plt.imsave(arr=kernel_4d[:, :, 0, 0], fname='kernel.png', cmap='gray')
plt.show()
# Perform the convolution with the 4d tensors:

#convolved = utils.convolve(...

# convolved_show = (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))
# print(convolved_show.shape)
#plt.figure(figsize=(10, 10))
#plt.imshow(utils.montage(convolved_show[..., 0], 'convolved.png'), cmap='gray')





