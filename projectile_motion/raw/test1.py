import functools
import time
import timeit
import numpy as np

import numba as nb




@nb.njit
def test():
    # start = time.time()
    print(np.clip(np.array([0.6]), -1, 1))
    # print(time.time() - start)


test()

# test:np.ndarray = np.random.randint(0, 1000, size=500)
test1 = np.linspace(1, 100, 500)



# res = timeit.timeit(functools.partial(np.argsort, test, **{"kind":"quicksort"}), number=10000)
# print(res/100)

# np.argsort
print(test1[test1.argsort()[:2]])

A = np.ones((2, 3))
A[1] = 0

res = np.any(A, axis=1)

print(res)
