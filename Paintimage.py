
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
# %matplotlib inline
plt.style.use('ggplot')

# Bit of formatting because I don't like the default inline code style:
#from IPython.core.display import HTML
#HTML("""<style> .rendered_html code { 
#    padding: 2px 4px;
#    color: #c7254e;
#    background-color: #f9f2f4;
#    border-radius: 4px;
#} </style>""")

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

