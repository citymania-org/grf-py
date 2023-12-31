import array
import time
import timeit
from typing import Union, ByteString

import numpy as np

import grf


N = 1000
S = 200

def slice_crop(data):
    npcheck = data[:, :, 3]
    return npcheck.any(0), npcheck.any(1)


def where_crop(data):
    return (data.any((0, 2), where=[0, 0, 0, 1]),
            data.any((1, 2), where=[0, 0, 0, 1]))


def generate_test():
    lx = np.random.randint(S // 2)
    mx = S - np.random.randint(S // 2)
    ly = np.random.randint(S // 2)
    my = S - np.random.randint(S // 2)
    a = np.zeros((S, S, 4), dtype=np.uint8)
    for y in range(ly, my):
        for x in range(lx, mx):
            for i in range(4):
                a[y, x, i] = int(np.random.rand() < 0.01)
    return a


t1 = t2 = 0
for i in range(N):
    t = generate_test()
    t0 = time.time()
    a1 = slice_crop(t)
    t1 += time.time() - t0
    t0 = time.time()
    a2 = where_crop(t)
    t2 += time.time() - t0
    if not np.array_equal(a1[0], a2[0]):
        print('ERROR cols')
        print('a1:\n', a1[0])
        print('a2:\n', a2[0])
        print('cmp:\n', a1[0] == a2[0])
        break
    if not np.array_equal(a1[1], a2[1]):
        print('ERROR rows')
        print('a1:\n', a1[1])
        print('a2:\n', a2[1])
        print('cmp:\n', a1[1] == a2[1])
        break
    print('.', end=['', f' {i + 1}/{N}\n'][i % 100 == 99], flush=True)
    # t1 += timeit.timeit(lambda: slice_crop(t))
    # t2 += timeit.timeit(lambda: where_crop(t))

print('slice_crop', t1)
print('where_crop', t2)
