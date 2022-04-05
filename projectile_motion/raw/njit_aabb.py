import numpy as np
from numba import njit
from vector import Vector3D
import vector


AABBType = np.ndarray
basically_inf = np.float64(100000)

@njit
def aabb_from_xyzr(x, y, z, r):
    return np.array([x - r, y - r, z - r, x + r, y + r, z + r], dtype=np.float64)

@njit
def distance_to(src: Vector3D, dest: Vector3D):
    return np.sqrt((src.x - dest.x) ** 2 + (src.y - dest.y) ** 2 + (src.z - dest.z) ** 2)

@njit
def lerp(f, f2, f3):
    return f2 + f * (f3 - f2)

@njit
def get_center(aabb: AABBType):
    return vector.obj(x=lerp(0.5, aabb[0], aabb[3]), y=lerp(0.5, aabb[1], aabb[4]),z=lerp(0.5, aabb[2], aabb[5]))

@njit(fastmath=True)
def intersects_segment(aabb: AABBType, org: Vector3D, dest: Vector3D) -> bool:
    dir = dest.subtract(org).unit()
    d = distance_from_ray(aabb, org, dir)
    return d < distance_to(org, dest) and d > 0

@njit(fastmath=True)
def distance_from_ray(aabb: AABBType, org: Vector3D, dir: Vector3D) -> float:
        ro: np.ndarray = np.array([org.x, org.y, org.z])
        rd = dir.unit()
        rd: np.ndarray = np.array([dir.x, dir.y, dir.z])
        rd_inv: np.ndarray = np.reciprocal(rd)
        min, max = np.split(aabb, 2)
        dims = ro.size

        lo = -basically_inf
        hi = +basically_inf

        for i in range(0, dims):
            dim_lo = (min[i] - ro[i]) * rd_inv[i]
            dim_hi = (max[i] - ro[i]) * rd_inv[i]

            if dim_lo > dim_hi:
                tmp = dim_lo
                dim_lo = dim_hi
                dim_hi = tmp
            if dim_hi < lo or dim_lo > hi:
                return basically_inf
            
            if dim_lo > lo:
                lo = dim_lo
            if dim_hi < hi:
                hi = dim_hi


        return basically_inf if lo > hi else lo
