import gzip
import struct
import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Tuple, Union, Iterable, Any
from needle import backend_ndarray as nd


def _min_max_scaler(
    data: np.ndarray,
    *,
    max_range: float,
    min_range: float,
    axis: Optional[Tuple[int]] = None,
):
    """inspired by sklearn"""
    std = (data - data.min(axis=axis)) / (data.max(axis=axis) - data.min(axis=axis))
    return std * (max_range - min_range) + min_range


def _read_mnist_images(image_filename: str) -> np.ndarray:
    with gzip.open(image_filename, 'rb') as file:
        _, number_of_images, number_of_rows, number_of_cols = struct.unpack(">4I", file.read(16))
        data = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).astype(np.float32)
        data = data.reshape(number_of_images, number_of_rows * number_of_cols)
        return _min_max_scaler(data, min_range=0, max_range=1)


def _read_mnist_labels(label_filename: str) -> np.ndarray:
    with gzip.open(label_filename, 'rb') as file:
        file.read(8)  # skip magic number and number of labels
        labels = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        return labels


def parse_mnist(image_filename: str, label_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    return _read_mnist_images(image_filename), _read_mnist_labels(label_filename)


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            # return img[:, ::-1, :]
            return np.flip(img, 1)
            

        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        height, width, _ = img.shape
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        img = img[
            shift_x + self.padding:shift_x + self.padding + height,
            shift_y + self.padding:shift_y + self.padding + width,
        ]
        return img


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def _init_ordering(self) -> None:
        self.current_order = 0

        if not self.shuffle and self.ordering:
            return

        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        self.ordering = np.array_split(
            indices, 
            range(self.batch_size, len(self.dataset), self.batch_size),
        )
        

    def __iter__(self):
        self._init_ordering()
        return self
    
    def __next__(self):
        if self.current_order >= len(self.ordering):
            raise StopIteration
                    
        batch = (self.dataset[i] for i in self.ordering[self.current_order])
        # batch_feats, batch_labels = zip(*batch)
        # batch_feats = Tensor.make_const(np.stack(batch_feats))
        # batch_labels = Tensor.make_const(np.stack(batch_labels))

        self.current_order += 1
        return tuple(Tensor.make_const(np.stack(batch_type)) for batch_type in zip(*batch)) 



class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        self.height = 28
        self.width = 28
        self.num_channels = 1

    def __getitem__(self, index) -> object:
        imgs, labels = self.images[index], self.labels[index]
        is_single_item = imgs.ndim < 2

        if is_single_item:
            imgs = imgs[None, :]
            labels = np.array([labels])
        
        imgs = imgs.reshape(imgs.shape[0], self.height, self.width, self.num_channels)
        imgs = np.array([self.apply_transforms(img) for img in imgs])

        if is_single_item:
            imgs = imgs[0]
            labels = labels[0]

        return imgs, labels

    def __len__(self) -> int:
        return self.images.shape[0]


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION