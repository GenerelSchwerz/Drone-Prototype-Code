import random
import numpy as np
import numba as nb
from numba import njit


import vector

@njit
def dir_from_yaw_pitch_speed(yaw, pitch, speed) -> vector.VectorNumpy3D:
    if speed:
        x = speed * np.sin(-yaw)
        y = speed * np.sin(pitch)
        z = speed * np.cos(-yaw)
        vx_mag = np.hypot(x, z)
        vx_rat = np.sqrt(vx_mag * vx_mag - y * y)
        all_rat = vx_rat / vx_mag 
        return vector.obj(x=x * all_rat, y=y, z=z * all_rat)
    else:
       return vector.obj(x=0., y=0., z=0.)










@njit
def test():

    # speed tests for 
    for x in np.linspace(0, np.pi * 2, 3):
        for y in np.linspace(0, np.pi * 2, 3):
            for z in np.linspace(.1, 1., 10):
                dir_from_yaw_pitch_speed(x, y, z)

if __name__ == "__main__":
    test()
    # test = dir_from_yaw_pitch_speed(np.pi * 1, np.pi * 0, 1)
    # print(test)

    #.isclose(vector.obj(x=0, y=1, z=0)
