import timeit

from numpy import copy, random, arange
import numpy as np

S = 1000

random.seed(0)
data = random.randint(256, size=(S, S, 4), dtype=np.uint8)
mask = random.randint(256, size=(S, S, 1), dtype=np.uint8)


def f1(data, mask):
	npimg = data.reshape(S * S, 4)
	return npimg
	# mask = mask.reshape(S * S, 1)
	# raw_data = np.concatenate((npimg, mask), axis=1)
	# return raw_data.reshape(S * S * 5)


def f2(data, mask):
	img = data[:, :, :3].reshape(S * S, 3)
	alpha = data[:, :, 3].reshape(S * S, 1)
	return img
	# mask = mask.reshape(S * S, 1)
	# raw_data = np.concatenate((img, alpha, mask), axis=1)
	# return raw_data.reshape(S * S * 5)


#TODO dstack

t1 = timeit.timeit(lambda: f1(data, mask), number=1000)
print('T1', t1)
t2 = timeit.timeit(lambda: f2(data, mask), number=1000)
print('T2', t2)
