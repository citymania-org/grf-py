import timeit

from numpy import copy, random, arange
import numpy as np


random.seed(0)
data = random.randint(256, size=10**5, dtype=np.uint32)

# d = {4: 0, 9: 5, 14: 10, 19: 15, 20: 0, 21: 1, 22: 2, 23: 3, 24: 0}
a = arange(0, 256)
random.shuffle(a)
d = dict(enumerate(a))
dk = np.array(tuple(d.keys()), dtype=np.uint32)
dv = np.array(tuple(d.values()), dtype=np.uint32)
dkp = np.array(tuple(d.keys()) + (2**32-1, ), dtype=np.uint32)
dvp = np.array(tuple(d.values()) + (0, ), dtype=np.uint32)

def f1(a, d):
	b = copy(a)
	for k, v in d.items():
		b[a==k] = v
	return b

def f2(a, d):
	for i in range(len(a)):
		a[i] = d.get(a[i], a[i])
	return a

def f3(a, dk, dv):
	mp = arange(0, max(a)+1)
	mp[dk] = dv
	return mp[a]


def f4(a, d):
    colors_to_remap = np.array(list(d.keys()))
    print(a)
    print(np.isin(a, colors_to_remap))
    print(a[np.isin(a, colors_to_remap)])
    print('CCTR', colors_to_remap)
    print('REMAP', [d[c] for c in colors_to_remap])
    a[np.isin(a, colors_to_remap)] = np.array([d[c] for c in colors_to_remap])
    return a

# def f4(a, dk, dv):
# 	print('a', a)
# 	print('dk', dk)
# 	print('dv', dv)
# 	sort_idx = np.argsort(dk)
# 	print('sort', sort_idx)
# 	idx = np.searchsorted(dk, a, sorter=sort_idx)
# 	print('idx', idx)
# 	print('dvs', dv[sort_idx])
# 	out = dv[sort_idx][idx]
# 	mask = (out == 0)
# 	out[mask] = a[mask]

a = copy(data)
res = f2(a, d)

assert (f1(data, d) == res).all()
assert (f3(data, dk, dv) == res).all()
assert (f4(data, d) == res).all()
# assert (f4(data, d) == res).all()
# assert (f4(data, dkp, dvp) == res).all()


# t4 = timeit.timeit(lambda: f4(data, dk, dv), number=100)
# print('T4', t4)

t1 = timeit.timeit(lambda: f1(data, d), number=100)
print('T1', t1)
# t2 = timeit.timeit(lambda: f2(data, d), number=100)
# print('T2', t2)
t3 = timeit.timeit(lambda: f3(data, dk, dv), number=100)
print('T3', t3)
t4 = timeit.timeit(lambda: f4(data, d), number=100)
print('T4', t4)
