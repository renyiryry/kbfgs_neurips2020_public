# ['MNIST-autoencoder-relu-N1-1000-sum-loss', 'CURVES-autoencoder-relu-sum-loss', 'FacesMartens-autoencoder-relu']:
import mat4py
import numpy as np

import gzip
import os
import urllib.request
import sys

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data
    
    
def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

def maybe_download(SOURCE_URL, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


class DataSet(object):
    def __init__(self, images, labels, if_autoencoder, input_reshape):
        self._num_examples = images.shape[0]
        if input_reshape == 'fully-connected':
            images = np.swapaxes(images, 2, 3)
            images = np.swapaxes(images, 1, 2)
            images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
        images = images.astype(np.float32)
        if if_autoencoder:
            labels = images
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(name_dataset, name_model, home_path, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    
#     train_dir = '../data/' + name_dataset + '_data'
    train_dir = 'data/' + name_dataset + '_data'
    
    VALIDATION_SIZE = 0
      
    if name_dataset == 'MNIST':
        print('Begin laoding data...')
        if_autoencoder = True
        
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        train_images = extract_images(local_file)

        local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
        test_images = extract_images(local_file)

        # see "Reducing the Dimensionality of Data with Neural Networks"
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        train_labels = train_images
        test_labels = test_images
        
    elif name_dataset == 'FACES':
        if_autoencoder = True
        
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'newfaces_rot_single.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
    
        images_ = mat4py.loadmat(local_file)
        images_ = np.asarray(images_['newfaces_single'])
        images_ = np.transpose(images_)

        train_images = images_[:103500]
        test_images = images_[-41400:]

        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels = train_images
        test_labels = test_images   
        
    elif name_dataset == 'CURVES':
        if_autoencoder = True
        
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'digs3pts_1.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)

        import mat4py
    
        images_ = mat4py.loadmat(local_file)
            
        train_images = np.asarray(images_['bdata'])
        test_images = np.asarray(images_['bdatatest'])
        
        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels = train_images
        test_labels = test_images

    else:
        print('error: Dataset not supported.')
        sys.exit()
        
    validation_images = train_images[:1]
    validation_labels = train_labels[:1]
    
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    input_reshape = 'fully-connected'
    
    data_sets.train = DataSet(train_images, train_labels, if_autoencoder, input_reshape)
    data_sets.validation = DataSet(validation_images, validation_labels, if_autoencoder, input_reshape)
    data_sets.test = DataSet(test_images, test_labels, if_autoencoder, input_reshape)

    return data_sets


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]
