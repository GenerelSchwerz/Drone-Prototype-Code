import functools
import math
import timeit
from numbalsoda import lsoda_sig, lsoda
from matplotlib import pyplot as plt
import numpy as np
import numba as nb
import time

g = 9.81

@nb.njit(fastmath=True)
def get_pitch_wo_drag(src, dest, speed) -> np.float:
    pitch = np.arctan2(dest[1] - src[1], dest[0] - src[0])
    offset = np.arcsin((np.sqrt((((dest[0] - src[0]) ** 2) + (dest[1] - src[1]) ** 2))* g) / (2 * (speed ** 2))) if speed else 0
    return pitch + offset


#t = unused.
#u_ = [x, y, vx, vy] so len(4)
#du = pointer to put values into.
#p_ = [k, g, a_x, a_y] so len(4)
@nb.cfunc(lsoda_sig, cache=True)
def f_param(t, u_, du, p_): # pass in a as a paameter
    u = nb.carray(u_, (4,))
    p = nb.carray(p_, (4,))
    k, g, a_x, a_y = p
    x,y, vx, vy = u
    dx_dt = vx
    dy_dt = vy
    dvx_dt = a_x-k*vx # for proj. motion, a_x = 0
    dvy_dt = a_y-g-k*vy # for proj. motion, a_y = 0

    du[0] = dx_dt
    du[1] = dy_dt
    du[2] = dvx_dt
    du[3] = dvy_dt


funcptr_param = f_param.address
np.random.seed(0)

@nb.njit(parallel=True)
def main(n, t_eval):
    a = np.array([0.01, 9.81, 0., 0.])
    x = np.empty((n,len(t_eval)), np.float64)
    y = np.empty((n,len(t_eval)), np.float64)
    dx = np.empty((n,len(t_eval)), np.float64)
    dy = np.empty((n,len(t_eval)), np.float64)
    for i in nb.prange(n):
        u0 = np.empty((4,), np.float64)
        # print(u0)
        u0[0] = np.float64(0) #np.random.uniform(4.5,5.5)
        u0[1] = np.float64(0) #np.random.uniform(0.7,0.9)
        u0[2] = np.float64(1) #np.random.uniform(4.5,5.5)
        u0[3] = np.float64(1) #np.random.uniform(0.7,0.9)
        usol, success = lsoda(funcptr_param, u0, t_eval, data = a, rtol = 1e-8, atol = 1e-8)
        # print(usol)
        x[i] = usol[:,0]
        y[i] = usol[:,1]
        dx[i] = usol[:,0]
        dy[i] = usol[:,1]
    return x, y


@nb.njit(parallel=False)
def find_dist_and_pitch(theta_range, t_eval, start, dest, speed, params):
    distances = np.empty((len(theta_range), len(t_eval)))
    for i in nb.prange(len(theta_range)):
        theta = theta_range[i]
        u0 = np.append(start, np.array([np.cos(theta) * speed, np.sin(theta) * speed]), 0)
        usol, success = lsoda(funcptr_param, u0, t_eval, data = params, rtol = 1e-8, atol = 1e-8)
        x_sub = usol[:,0] - dest[0]
        y_sub = usol[:,1] - dest[1]
        distances[i] = np.sqrt(np.square(x_sub) + np.square(y_sub))
    # print("duration:", time.time() - start)
    # print("min dist:", test.flat[np.argmin(test)], np.degrees(theta_range[int(np.floor(np.argmin(test) / len(t_eval)))])) 
    
    min = distances.argmin()
    return distances.flat[min], theta_range[int(np.floor(min / t_eval.size))]


@nb.njit(cache=True)
def get_closest_point(usol, dest):
    x_sub = usol[:,0] - dest[0]
    y_sub = usol[:,1] - dest[1]
    distances = np.sqrt(np.square(x_sub) + np.square(y_sub))
    min = distances.argmin()
    return min, distances.flat[min] 
    
@nb.njit()
def which_direction(t_eval, start, dest, speed, theta, params):
    u0 = np.append(start, np.array([np.cos(theta) * speed, np.sin(theta) * speed]), 0)
    usol, success = lsoda(funcptr_param, u0, t_eval, data = params, rtol = 1e-8, atol = 1e-8)
    ind, p = get_closest_point(usol, dest)
    return usol[ind][1] - dest[1] < 0



def full_test(src, dest, v, options, t_eval, tolerance):
    for x in range(10, 100):
        for z in range(-30, 30):
  
            starting_pitch = get_pitch_wo_drag(src, dest, v)
            # print(np.degrees(starting_pitch))
            should_pos = which_direction(t_eval, src, dest, v, starting_pitch, options)
            # print(should_pos)

            if should_pos:
                r_eval = np.arange(starting_pitch, starting_pitch + np.radians(15), np.radians(0.25))
            else:
                r_eval = np.arange(starting_pitch - np.radians(15), starting_pitch, np.radians(0.25))

            start_sim = time.time()
            test: np.ndarray = find_dist_and_pitch(r_eval, t_eval, src, dest, v, options)
            # assert test, "didn't find solution."
         
            assert test[0] < tolerance, f"{x}, {z}, {v}"

            end_sim = time.time()
            timey = end_sim - start_sim
            print(x, z, v)
            print("simulation took {:.2f} ms.".format(timey * 1000))

def test():
    src = np.array([0., 0.])
    dest = np.array([50, -30])
    v = 91.
    options = np.array([0.01, 9.81, 0., 0.])
    t_eval = np.arange(0, 3, 0.002)
    starting_pitch = get_pitch_wo_drag(src, dest, v)
    print(np.degrees(starting_pitch))
    should_pos = which_direction(t_eval, src, dest, v, starting_pitch, options)
    print(should_pos)

    if should_pos:
        r_eval = np.arange(starting_pitch, starting_pitch + np.radians(15), np.radians(0.25))
    else:
        r_eval = np.arange(starting_pitch - np.radians(15), starting_pitch, np.radians(0.25))

    # calling first time for JIT compiling
    u1, u2 = main(10, t_eval)
    res = timeit.timeit(functools.partial(main, 1000, t_eval), number=1)
    find_dist_and_pitch(r_eval, t_eval, src, dest, v, options)
    start = time.time()
    test: np.ndarray = find_dist_and_pitch(r_eval, t_eval, src, dest, v, options)
    print("duration: {:.2f} ms".format((time.time() - start) * 1000))
    print("min dist: {:.2f} cm. Angle: {:f}".format(test[0] * 100, np.degrees(test[1])))




if __name__ == "__main__":
    src = np.array([0., 0.])
    dest = np.array([50, -30])
    v = 50.
    options = np.array([0.01, g, 0., 0.])
    t_eval = np.arange(0, 2.5, 0.002)
    tolerance = 0.1

    # for JIT
    full_test(src, np.array([1, 1]), 5, np.array([0.00, 0, 0., 0.]), np.arange(0, 1, 0.1), 10)

    res = timeit.timeit(functools.partial(full_test, src, dest, v, options, t_eval, tolerance), number=1)
    print(res)
    print("Average duration: {:f} ms".format(res / 5400 * 1000))
    print("fps: {:f}".format( 5400 / res))