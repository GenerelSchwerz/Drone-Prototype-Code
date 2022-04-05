from numbalsoda import lsoda_sig, lsoda
from matplotlib import pyplot as plt
import numpy as np
import numba as nb
import time

a_glob=np.array([1.5,1.5])

@nb.cfunc(lsoda_sig)
def f_global(t, u_, du, p): # variable a is global
    u = nb.carray(u_, (2,))
    du[0] = u[0]-u[0]*u[1]*a_glob[0]
    du[1] = u[0]*u[1]-u[1]*a_glob[1]

    
@nb.cfunc(lsoda_sig)
def f_local(t, u_, du, p): # variable a is local
    u = nb.carray(u_, (2,))
    a = np.array([1.5,1.5]) 
    du[0] = u[0]-u[0]*u[1]*a[0]
    du[1] = u[0]*u[1]-u[1]*a[1]

    
@nb.cfunc(lsoda_sig)
def f_param(t, u_, du, p): # pass in a as a paameter
    u = nb.carray(u_, (2,))
    du[0] = u[0]-u[0]*u[1]*p[0]
    du[1] = u[0]*u[1]-u[1]*p[1]

funcptr_glob = f_global.address
funcptr_local = f_local.address
funcptr_param = f_param.address
t_eval = np.linspace(0.0,20.0,201)
np.random.seed(0)
a = np.array([1.5,1.5])

@nb.njit(parallel=True)
def main(n, flag):
#     a = np.array([1.5,1.5])
    u1 = np.empty((n,len(t_eval)), np.float64)
    u2 = np.empty((n,len(t_eval)), np.float64)
    for i in nb.prange(n):
        u0 = np.empty((2,), np.float64)
        u0[0] = np.random.uniform(4.5,5.5)
        u0[1] = np.random.uniform(0.7,0.9)
        if flag ==1: # global
            usol, success = lsoda(funcptr_glob, u0, t_eval, rtol = 1e-8, atol = 1e-8)
        if flag ==2: # local
            usol, success = lsoda(funcptr_local, u0, t_eval, rtol = 1e-8, atol = 1e-8)
        if flag ==3: # param
            usol, success = lsoda(funcptr_param, u0, t_eval, data = a, rtol = 1e-8, atol = 1e-8)
        u1[i] = usol[:,0]
        u2[i] = usol[:,1]
    return u1, u2

@nb.njit(parallel=False)
def main_series(n, flag): # same function as above but with parallel flag = False
    #     a = np.array([1.5,1.5])
    u1 = np.empty((n,len(t_eval)), np.float64)
    u2 = np.empty((n,len(t_eval)), np.float64)
    for i in nb.prange(n):
        u0 = np.empty((2,), np.float64)
        u0[0] = np.random.uniform(4.5,5.5)
        u0[1] = np.random.uniform(0.7,0.9)
        if flag ==1: # global
            usol, success = lsoda(funcptr_glob, u0, t_eval, rtol = 1e-8, atol = 1e-8)
        if flag ==2: # local
            usol, success = lsoda(funcptr_local, u0, t_eval, rtol = 1e-8, atol = 1e-8)
        if flag ==3: # param
            usol, success = lsoda(funcptr_param, u0, t_eval, data = a, rtol = 1e-8, atol = 1e-8)
        u1[i] = usol[:,0]
        u2[i] = usol[:,1]
    return u1, u2

n = 100
# calling first time for JIT compiling
u1, u2 = main(n,1)
u1, u2 = main(n,2)
u1, u2 = main(n,3)

u1, u2 = main_series(n,1)
u1, u2 = main_series(n,1)
u1, u2 = main_series(n,1)

# Running code for large number 
n = 10000
a1 = time.time()
u1, u2 = main(n,1) # global
a2 = time.time()
print("global pararllel:", a2 - a1) # this is fast


a1 = time.time()
u1, u2 = main(n,2) # local
a2 = time.time()


print("local pararllel:", a2 - a1) # this is slow

a1 = time.time()
u1, u2 = main(n,3) # param
a2 = time.time()
print("param pararllel:", a2 - a1) # this is fast and almost identical performance as global

a1 = time.time()
u1, u2 = main_series(n,1) # global
a2 = time.time()
print("global series:", a2 - a1) # this is faster than local + parallel

a1 = time.time()
u1, u2 = main_series(n,2) # local
a2 = time.time()
print("local series:", a2 - a1) # this is slow

a1 = time.time()
u1, u2 = main_series(n,3) # param
a2 = time.time()
print("param series:", a2 - a1) # this is fast and almost identical performance as global