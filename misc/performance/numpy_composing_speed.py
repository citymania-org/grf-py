import timeit

from numpy import copy, random, arange
import numpy as np

S = 1000
x, y, w, h = 20, 30, 500, 700

random.seed(0)
data = random.randint(256, size=(S, S, 4), dtype=np.uint8)
npmask = random.randint(256, size=(S, S, 1), dtype=np.uint8)
npimg = data[:, :, :3]
npalpha = data[:, :, 3]
npimg = npimg[y:y+h, x:x+w]
npalpha = npalpha[y:y+h, x:x+w]
npmask= npmask[y:y+h, x:x+w]
data = data[y:y+h, x:x+w]

# a = data[:, :, :3][y:y+h, x:x+w]
# b = data[:, :, 3][y:y+h, x:x+w]
a = npimg
b = npalpha

print(a.dtype == np.uint16)
print(a.data, a.data.strides, a.data.format, a.data.suboffsets, id(a))
print(b.data, b.data.strides, b.data.format, b.data.suboffsets, id(b))
# print(a.base is b.base)
# print(a.strides)
# print(b.strides)

def restore_rgba_image(npimg, npalpha):
	# We rely on base of both arrays to still be the original array
	base = npimg.base
	if base is not npalpha.base:
		return None

	ii = npimg.__array_interface__
	ai = npalpha.__array_interface__

	# Check that it starts on the same pixel but skips RGB components
	if ii['data'][0] + 3 != ai['data'][0]:
		return None

	# Check that they have the same strides (offsets to next element)
	if ii['strides'][:2] != ai['strides'] or ii['strides'][2] != 1:
		return None

	# Check that they have the same shape
	if ii['shape'][:2] != ai['shape'] or ii['shape'][2] != 3:
		return None

	# Check that all other metadata matches
	if ii['descr'] != ai['descr'] or ii['typestr'] != ai['typestr'] or ii['version'] != ai['version']:
		return None

	if base.shape[:2] == npalpha.shape:
		return base

	# Sanity check for crop calculation
	if npimg.dtype != np.uint8 or npalpha.dtype != np.uint8:
		return None

	sx, sy = npalpha.shape
	x, y, z = base.strides
	flat_offset = ii['data'][0] - base.__array_interface__['data'][0]
	ox, oy, oz = flat_offset // x, (flat_offset % x) // y, (flat_offset % y) // z

	return npimg.base[ox:ox + sx, oy:oy + sy, oz:oz + 4]


# print('one', check_same_image(a, b))


def f1(data, npimg, npalpha, npmask):
	w, h, _ = npimg.shape
	return np.concatenate((
		data.reshape(w * h, 4),
		npmask.reshape(w * h, 1)
	), axis=1)


def f2(data, npimg, npalpha, npmask):
	w, h, _ = npimg.shape
	nprgba = restore_rgba_image(npimg, npalpha)
	if nprgba is not None:
		return np.concatenate((
			nprgba.reshape(w * h, 4),
			npmask.reshape(w * h, 1)
		), axis=1)
	return np.concatenate((
		npimg.reshape(w * h, 3),
		npalpha.reshape(w * h, 1),
		npmask.reshape(w * h, 1)
	), axis=1)


def f3(data, npimg, npalpha, npmask):
	w, h, _ = npimg.shape
	return np.concatenate((
		npimg.reshape(w * h, 3),
		npalpha.reshape(w * h, 1),
		npmask.reshape(w * h, 1)
	), axis=1)


#TODO dstack

t1 = timeit.timeit(lambda: f1(data, npimg, npalpha, npmask), number=100)
print('T1', t1)
t2 = timeit.timeit(lambda: f2(data, npimg, npalpha, npmask), number=100)
print('T2', t2)
t3 = timeit.timeit(lambda: f3(data, npimg, npalpha, npmask), number=100)
print('T3', t3)
