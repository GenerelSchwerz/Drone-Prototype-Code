from vector import Vector, Vector3D, VectorObject3D
import numpy as np
import vector
from numba import njit, jit, float64
from numba.experimental import jitclass

# temporary.

@njit
def distance_to(src: Vector3D, dest: Vector3D):
    return np.sqrt((src.x - dest.x) ** 2 + (src.y - dest.y) ** 2 + (src.z - dest.z) ** 2)

@njit
def lerp(f, f2, f3):
    return f2 + f * (f3 - f2)


aabb_specs = [
    ("minX", float64),
    ("minY", float64),
    ("minZ", float64),
    ("maxX", float64),
    ("maxY", float64),
    ("maxZ", float64)
]


@jitclass(aabb_specs)
class AABB(object):
    def __init__(self, minX, minY, minZ, maxX, maxY, maxZ):
        self.minX = minX
        self.minY = minY
        self.minZ = minZ
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ


    def __str__(self):
        return f"AABB(x0: {self.minX}, y0: {self.minY},z0: {self.minZ},x1: {self.maxX},y1: {self.maxY},z1: {self.maxZ})"

    @staticmethod
    def from_xyzr(x, y, z, r):
        return AABB(x - r, y - r, z - r, x + r, y + r, z + r)

    @staticmethod
    def from_vectors(min: Vector3D, max: Vector3D):
        return AABB(min.x, min.y, min.z, max.x, max.y, max.z)

    @staticmethod
    def from_arrays(min: np.ndarray, max: np.ndarray):
        return AABB(min[0], min[1], min[2], max[0], max[1], max[2])


    # @njit
    def to_vectors(self):
        return vector.obj(x=self.minX, y=self.minY, z=self.minZ), vector.obj(x=self.maxX, y=self.maxY, z=self.maxZ)

    # @njit
    def to_arrays(self):
        return np.array([self.minX, self.minY, self.minZ]), np.array([self.maxX, self.maxY, self.maxZ])

    # @njit
    def to_array(self):
        return np.array([self.minX, self.minY, self.minZ, self.maxX, self.maxY, self.maxZ])
    
    # @njit
    def to_vertices(self):
           return [
            vector.obj(self.minX, self.minY, self.minZ),
            vector.obj(self.minX, self.minY, self.maxZ),
            vector.obj(self.minX, self.maxY, self.minZ),
            vector.obj(self.minX, self.maxY, self.maxZ),
            vector.obj(self.maxX, self.minY, self.minZ),
            vector.obj(self.maxX, self.minY, self.maxZ),
            vector.obj(self.maxX, self.maxY, self.minZ),
            vector.obj(self.maxX, self.maxY, self.maxZ),
        ]

    # @njit
    def get_center(self):
        return vector.obj(x=lerp(0.5, self.minX, self.maxX), y=lerp(0.5, self.minY, self.maxY),z=lerp(0.5, self.minZ, self.maxZ))

    def get_center_arr(self):
        return np.array([lerp(0.5, self.minX, self.maxX), lerp(0.5, self.minY, self.maxY), lerp(0.5, self.minZ, self.maxZ)])



    def get_bottom_center(self):
        return vector.obj(x=lerp(0.5, self.minX, self.maxX), y=self.minY,z=lerp(0.5, self.minZ, self.maxZ))



    # @njit
    def set(self, x0, y0, z0, x1, y1, z1):
        self.minX = x0, 
        self.minY = y0
        self.minZ = z0
        self.maxX = x1
        self.maxY = y1
        self.maxZ = z1

    # @njit
    def clone(self):
        return AABB(self.minX, self.minY, self.minZ, self.maxX, self.maxY, self.maxZ)

    # @njit
    def intersects_ray(self, org: Vector3D, dir: Vector3D) -> bool:
        d = self.distance_from_ray(org, dir)
        return d == np.Inf

    # @njit
    def intersects_segment(self, org: Vector3D, dest: Vector3D) -> bool:
        dir = dest.subtract(org).unit()
        d = self.distance_from_ray(org, dir)
        return d < distance_to(org, dest) and d > 0


    # @njit(parallel=True)
    def distance_from_ray(self, org: Vector3D, dir: Vector3D) -> float:
        ro: np.ndarray = np.array([org.x, org.y, org.z])
        rd = dir.unit()
        rd: np.ndarray = np.array([dir.x, dir.y, dir.z])
        rd_inv: np.ndarray = np.reciprocal(rd)
        min, max = self.to_arrays()
        dims = ro.size

        lo = -np.Inf
        hi = +np.Inf

        for i in range(0, dims):
            dim_lo = (min[i] - ro[i]) * rd_inv[i]
            dim_hi = (max[i] - ro[i]) * rd_inv[i]

            if dim_lo > dim_hi:
                tmp = dim_lo
                dim_lo = dim_hi
                dim_hi = tmp
            if dim_hi < lo or dim_lo > hi:
                return np.Inf
            
            if dim_lo > lo:
                lo = dim_lo
            if dim_hi < hi:
                hi = dim_hi


        return np.Inf if lo > hi else lo

    
    # @njit
    def distance_to_vec(self, pos: Vector3D):
        dx = np.max([self.minX - pos.x, 0, pos.x - self.maxX])
        dy = np.max([self.minY - pos.y, 0, pos.y - self.maxY])
        dz = np.max([self.minZ - pos.z, 0, pos.z - self.maxZ])
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    

