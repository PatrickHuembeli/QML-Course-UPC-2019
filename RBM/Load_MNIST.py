# Example file how to load MNIST from local folder.
# Download files from this site http://yann.lecun.com/exdb/mnist/


import struct
import gzip
import numpy as np
from matplotlib import pyplot as plt

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        
# path to the .gz file
path = 'data/t10k-images-idx3-ubyte.gz'
# Run the loader function
images = read_idx(path)

# Show the first image of the list (indicated with the index [0])
plt.imshow(images[0], interpolation='nearest')
plt.show()            