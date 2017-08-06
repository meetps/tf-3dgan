"""
load_data.py
[python] 3.6
"""
import os, sys
import numpy as np
import urllib.request
import tarfile
import pickle

cifar_size = 32
cifar_channel = 3
cifar10_batch_num = 5
cifar10_batch_size = 10000

def _unpickle(path):
	print("Loading: %s" % path)
	with open(path, mode='rb') as f:
		data = pickle.load(f, encoding='bytes')
	return data

def _convert_cifar(raw):
	raw = np.array(raw, dtype=float) / 255.0	# map to [0,1]
	images = raw.reshape([-1, cifar_channel, cifar_size, cifar_size])
	images = images.transpose([0, 2, 3, 1])	# size = (-1,32,32,3)
	return images

def one_hot(classes, num_classes=None):
	if num_classes is None:
		num_classes = np.max(classes) - 1
	return np.eye(num_classes, dtype=float)[classes]

def load_cifar10(dataset_dir='../cifar10_data/'):
	"""
	Load ciafr10 train, test data.
	- train:	.data, .labels
	- test:		.data, .labels
	- labels:	[list]
	=> Return	: cifar10_obj (train, test, labels)

	[Usage]
	>>> cifar10  = load_cifar10()
	>>> trainObj = cifar10.train
	>>> testObj  = cifar10.test
	>>> lables	 = cifar10.labels 
	>>> trainObj.data
	array([[[[ 0.23137255,  0.24313725,  0.24705882],
	         [ 0.16862745,  0.18039216,  0.17647059],
	         [ 0.19607843,  0.18823529,  0.16862745],
	         ...

	>>> trainObj.labels
	array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
	       [ 0.,  0.,  0., ...,  0.,  0.,  1.],
	       [ 0.,  0.,  0., ...,  0.,  0.,  1.],
	       ...
	"""
	filepath = get_cifar10_path(dataset_dir)
	train_batch_num	= cifar10_batch_num
	train_total_num	= cifar10_batch_num * cifar10_batch_size
	train_batch		= os.path.join(filepath, 'data_batch')
	test_batch		= os.path.join(filepath, 'test_batch')
	labels_batch	= os.path.join(filepath, 'batches.meta')
	#####
	## Labels (class names)
	#####
	labels	= _unpickle(labels_batch)[b'label_names']
	labels	= [x.decode('utf-8') for x in labels]
	num_classes = len(labels)
	#####
	## Training data
	#####
	train	= np.zeros(shape=[train_total_num, cifar_size, cifar_size, cifar_channel], dtype=float)
	train_labels	= np.zeros(shape=[train_total_num], dtype=int)
	for i in range(train_batch_num):
		data	= _unpickle("%s_%s" % (train_batch,i+1))
		raw		= data[b'data']	# size = (10000,3072)
		trainbatch			= _convert_cifar(raw)
		trainbatch_labels	= np.array(data[b'labels'])
		train[i*len(trainbatch) : (i+1)*len(trainbatch), :] = trainbatch
		train_labels[i*len(trainbatch) : (i+1)*len(trainbatch)] = trainbatch_labels
	# return train, train_labels, one_hot(train_labels, num_classes)
	#####
	## Testing data
	#####
	data	= _unpickle(test_batch)
	raw		= data[b'data']		# size = (10000,3072)
	test	= _convert_cifar(raw)
	test_labels	= np.array(data[b'labels'])
	# return test, test_labels, one_hot(test_labels, num_classes)
	class dataObj:
		def __init__ (self, data, labels, num_classes):
			self.data	= data.reshape(-1, cifar_size * cifar_size * cifar_channel)
			self.labels	= one_hot(labels, num_classes)
			self.index	= 0
		def next_batch (self, batch_size):
			index	= self.index
			bs	= batch_size
			d	= self.data.reshape(len(self.data), -1)
			l	= self.labels
			if index + bs <= len(d):
				batch_data	 = d[index : index + bs]
				batch_labels = l[index : index + bs]
				self.index += bs
			else:
				batch_data	 = np.concatenate((d[index:], d[:(index+bs)-len(d)]), axis=0)
				batch_labels = np.concatenate((l[index:], l[:(index+bs)-len(d)]), axis=0)
				self.index += bs - len(d)
			# self.data	 = np.concatenate((d[bs:], d[:bs]), axis=0)
			# self.labels	 = np.concatenate((l[bs:], l[:bs]), axis=0)
			return batch_data, batch_labels
	class cifarObj:
		def __init__ (self, trainObj, testObj, labels):
			self.train = trainObj
			self.test  = testObj
			self.labels= labels
	trainObj	= dataObj(train, train_labels, num_classes)
	testObj		= dataObj(test, test_labels, num_classes)
	cifar10_obj	= cifarObj(trainObj, testObj, labels)
	return cifar10_obj


def get_cifar10_path(dataset_dir):
	"""
	Download cifar-10-python.tar.gz, and extract to cifar-10-batches-py.

	=> Return	: '../ciafr10_data/cifar-10-batches-py'
	"""

	DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

	if not os.path.exists(dataset_dir):
		os.makedirs(dataset_dir)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dataset_dir, filename)	# filepath = '../cifar10_data/cifar-10-python.tar.gz'
	if not os.path.exists(filepath):
		def _download_progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(url=DATA_URL, filename=filepath, reporthook=_download_progress)
		print()
		print("Download finished.")
	print("Extracting %s" % filepath)
	with tarfile.open(filepath, 'r:gz') as tar:
		tar.extractall(dataset_dir)		# extracting 'cifar-10-python.tar.gz' into '../cifar10_data/'
		dataset_dir = os.path.join(dataset_dir, tar.getmembers()[0].name)
		print("Extracting completely.")
	return dataset_dir					# return '../cifar10_data/cifar-10-batches-py'
