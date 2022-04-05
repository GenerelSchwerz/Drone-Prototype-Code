import time
import vector
import numpy as np
import awkward as ak  # at least version 1.2.0
import numba as nb
from numba import njit, float32, float64
from numba.experimental import jitclass
from numba.typed import List
from vector import Vector3D, VectorNumpy3D, VectorObject3D

from aabb import AABB
from njit_aabb import *


# Air density (kg.m-3), acceleration due to gravity (m.s-2).
rho_air = np.float(1.225)
g = np.float(9.81)


proj_specs = [
    ('c', float64),
    ('r', float64),
    ('a', float64),
    ('m', float64),            # a simple scalar field
    ('k', float64),
    ('k_over_m', float64),
]


@jitclass(spec=proj_specs)
class Projectile:
    def __init__(self, drag_coefficient: float64, radius: float64, mass: float64):
        # Drag coefficient, projectile radius (m), area (m2) and mass (kg).
        self.c = drag_coefficient
        self.r = radius
        self.a = np.pi * radius**2
        self.m = mass
        # For convenience, define  this constant.
        self.k = 0.5 * self.c * rho_air * self.a
        self.k_over_m = self.k/self.m


dest_specs = [
    ('x', float64),
    ('y', float64),
    ('z', float64),
    ('x_tol', float64),
    ('z_tol', float64),            # a simple scalar field
    ('tol', float64),          # an array field
]


@jitclass(spec=dest_specs)
class Destination:
    def __init__(self, goal_x: float64, goal_y: float64, goal_z: float64, x_allowance: float64, y_allowance: float64, z_allowance: float64):
        self.x = goal_x
        self.y = goal_y
        self.z = goal_z
        self.x_tol: np.float = np.float64(x_allowance)
        self.y_tol: np.float = np.float64(y_allowance)
        self.z_tol: np.float = np.float64(z_allowance)
        self.tol: np.float = np.sqrt(
            self.x_tol ** 2 + self.y_tol ** 2 + self.z_tol ** 2)

    def __str__(self):
        return f"Destination(x: {self.x}, z: {self.z})"

    def from_xzr(x: int, z: int, r: int):
        return Destination(x, z, r, r)


@njit
def get_yaw_to_target(src: Vector3D, dest: Vector3D):
    return np.arctan2(dest.x - src.x,  dest.z - src.x)


@njit  # no par
def add_with_scale(org: Vector3D, plus: Vector3D, scalar: float64):
    return org.add(plus.scale(scalar))


@njit  # no par
def offset_creation(vel: Vector3D, k_over_m: np.float):
    # offsets: np.ndarray = np.zeros(3)
    # offsets.itemset(0, -vel[0] * resistances[0]), offsets.itemset(1, -vel[1] * resistances[1] - g), offsets.itemset(2, -vel[2] * resistances[2])
    # print(-vel[0] * resistances[0])
    return vector.obj(x=-vel.x * k_over_m, y=-vel.y * k_over_m - g, z=-vel.z * k_over_m)


# May potentially skip. This is bad practice.
@njit  # no par
def made_it(goal: Vector3D, current: Vector3D, next: Vector3D, tolerance: float64):
    # flag1 = np.sqrt((goal.x - current.x) ** 2 +
    #                 (goal.z - current.z) ** 2) < tolerance
    # flag2 = np.sqrt((goal.x - next.x) ** 2 +
    #                 (goal.z - next.z) ** 2) < tolerance
    # flag3 = current.y > next.y
    # return flag1 and flag2 and flag3
    return distance_to(current, goal) < tolerance


@njit
def distance_to(src: Vector3D, dest: Vector3D):
    return np.sqrt((src.x - dest.x) ** 2 + (src.y - dest.y) ** 2 + (src.z - dest.z) ** 2)


@njit
def dir_from_yaw_pitch_speed(yaw, pitch, speed) -> vector.VectorNumpy3D:
    if speed:
        x = speed * np.sin(yaw)
        y = speed * np.sin(pitch)
        z = speed * np.cos(yaw)
        vx_mag = np.hypot(x, z)
        vx_rat = np.sqrt(vx_mag * vx_mag - y * y)
        all_rat = vx_rat / vx_mag
        return vector.obj(x=x * all_rat, y=y, z=z * all_rat)
    else:
        return vector.obj(x=0., y=0., z=0.)


@njit
def get_pitch_wo_drag(src: Vector3D, dest: Vector3D, v0s: float64) -> np.float:
    pitch = np.arctan2(
        dest.y - src.y, np.sqrt((dest.x - src.x) ** 2 + (dest.z - src.z) ** 2))
    offset = np.arcsin((np.sqrt(dest.z ** 2 + dest.x ** 2)
                       * g) / (2 * (v0s ** 2))) if v0s else 0
    return pitch + offset


@njit(fastmath=True)
def no_air_res(proj: Projectile, dest: Vector3D, v0: Vector3D, max_time: float64, fps: float64, tolerance: float64, points_back=False):
    vel = v0
    total_frames = max_time * fps
    inv_fps = 1. / fps
    current_pos = vector.obj(x=0., y=0., z=0.)
    next_pos: VectorObject3D = add_with_scale(current_pos, v0, inv_fps)

    if points_back:
        storage: np.ndarray = np.empty(total_frames * 3, dtype=float64)

    for frame in range(total_frames):
        if points_back:
            storage[3 * frame] = next_pos.x
            storage[3 * frame + 1] = next_pos.y
            storage[3 * frame + 2] = next_pos.z

        offsets = offset_creation(vel, proj.k_over_m)

        if made_it(dest, current_pos, next_pos, tolerance):
            return True, storage

        if distance_to(next_pos, dest) > distance_to(current_pos, dest):
            return False, storage

        # print("current:", current_pos, "next:", next_pos)
        current_pos = add_with_scale(current_pos, vel, inv_fps)
        vel = add_with_scale(vel, offsets, inv_fps)
        next_pos = add_with_scale(current_pos, vel, inv_fps)

    return False, storage


@njit(fastmath=True)
def no_air_res_njit_aabb(proj: Projectile, dest: AABBType, dest_center: Vector3D, v0: Vector3D, max_time: float64, fps: float64, points_back=False):
    vel = v0
    total_frames = max_time * fps
    inv_fps = 1. / fps
    current_pos = vector.obj(x=0., y=0., z=0.)
    next_pos: VectorObject3D = add_with_scale(current_pos, v0, inv_fps)

    if points_back:
        storage: np.ndarray = np.zeros(total_frames * 3, dtype=float64)

    for frame in range(total_frames):
        if points_back:
            storage[3 * frame] = next_pos.x
            storage[3 * frame + 1] = next_pos.y
            storage[3 * frame + 2] = next_pos.z

        offsets = offset_creation(vel, proj.k_over_m)
        if intersects_segment(dest, current_pos, next_pos):
            return True, storage

        if distance_to(next_pos, dest_center) > distance_to(current_pos, dest_center):
            return False, storage

        # print("current:", current_pos, "next:", next_pos)
        current_pos = add_with_scale(current_pos, vel, inv_fps)
        vel = add_with_scale(vel, offsets, inv_fps)
        next_pos = add_with_scale(current_pos, vel, inv_fps)

    return False, storage


@njit(fastmath=True)
def no_air_res_aabb(proj: Projectile, dest: AABB, dest_center: Vector3D, v0: Vector3D, max_time: float64, fps: float64, points_back=False):
    vel = v0
    total_frames = max_time * fps
    inv_fps = 1. / fps
    current_pos = vector.obj(x=0., y=0., z=0.)
    next_pos: VectorObject3D = add_with_scale(
        current_pos, v0, inv_fps).to_Vector3D()

    if points_back:
        storage: np.ndarray = np.zeros(total_frames * 3, dtype=float64)

    for frame in range(total_frames):
        if points_back:
            storage[3 * frame] = next_pos.x
            storage[3 * frame + 1] = next_pos.y
            storage[3 * frame + 2] = next_pos.z

        offsets = offset_creation(vel, proj.k_over_m)

        # this is apparently missing sometimes. What the **fuck?**
        if dest.intersects_segment(current_pos, next_pos):
            return True, storage

        if distance_to(next_pos, dest_center) > distance_to(current_pos, dest_center):
            return False, storage

        # print("current:", current_pos, "next:", next_pos)
        current_pos = add_with_scale(current_pos, vel, inv_fps)
        vel = add_with_scale(vel, offsets, inv_fps)
        next_pos = add_with_scale(current_pos, vel, inv_fps)

    return False, storage


@njit()
def format_storage(storage: np.ndarray):
    wanted_size = storage.size / 3
    if wanted_size % 3:
        raise Exception("Bad storage.")
    return np.reshape(storage, (int(storage.size / 3), 3))


@njit
def test(speed, tol):
    made_it_count = 0
    # made_it = []
    failed_count = 0
    # failed = []
    proj = Projectile(0.47, 6e-3, 40e-3)
    for x in np.arange(10, 100, 1):  # range(10, 100):
        z = 0
        # for z in range(10, 100):
        for y in np.arange(-30, 30, 1):  # range(-30, 30):
            current = vector.obj(x=0., y=0., z=0.)
            goal = vector.obj(x=x, y=y, z=z)
            goal_aabb = goal
            yaw = get_yaw_to_target(current, goal)
            pitch = get_pitch_wo_drag(current, goal, speed)
            vel = dir_from_yaw_pitch_speed(yaw, pitch, speed)
            resolution, storage = no_air_res(
                proj, goal_aabb, vel, 10, 60, tol, points_back=True)
            # print(x, y, z, resolution)
            if resolution:
                made_it_count += 1
                # made_it.append((x, y, z))
            else:
                failed_count += 1
                # failed.append((x, y, z))

    print(made_it_count, failed_count)


@njit
def test_njit_aabb(speed, tol):
    made_it_count = 0
    failed_count = 0
    proj = Projectile(0.47, 6e-3, 40e-3)
    for x in np.arange(10, 100, 1):  # range(10, 100):
        z = 0
        # for z in range(10, 100):
        for y in np.arange(-30, 30, 1):  # range(-30, 30):
            current = vector.obj(x=0., y=0., z=0.)
            goal = aabb_from_xyzr(x, y, z, tol)

            goal_center = get_center(goal)
            yaw = get_yaw_to_target(current, goal_center)
            pitch = get_pitch_wo_drag(current, goal_center, speed)
            vel = dir_from_yaw_pitch_speed(yaw, pitch, speed)
            made_it, points = no_air_res_njit_aabb(
                proj, goal, goal_center, vel, 10, 60, points_back=True)
            # print(x, y, z, resolution)
            if made_it:
                made_it_count += 1
                # made_it.append((x, y, z))
            else:
                failed_count += 1
                # failed.append((x, y, z))

    print(made_it_count, failed_count)


@njit
def test_aabb(speed, tol):
    made_it_count = 0
    failed_count = 0
    proj = Projectile(0.47, 6e-3, 40e-3)
    for x in np.arange(10, 100, 1):  # range(10, 100):
        z = 0
        # for z in range(10, 100):
        for y in np.arange(-30, 30, 1):  # range(-30, 30):
            # for y in np.linspace(-30, 30, 60):
            # y = 0.
            current = vector.obj(x=0., y=0., z=0.)
            goal = AABB(x - tol, y - tol, z - tol, x + tol, y + tol, z + tol)
            goal_center = goal.get_center()
            yaw = get_yaw_to_target(current, goal_center)
            pitch = get_pitch_wo_drag(current, goal_center, speed)
            vel = dir_from_yaw_pitch_speed(yaw, pitch, speed)
            made_it, points = no_air_res_aabb(
                proj, goal, goal_center, vel, 10, 60, points_back=True)
            # print(x, y, z, made_it)
            if made_it:
                made_it_count += 1
                # made_it.append((x, y, z))
            else:
                failed_count += 1
                # print(x, y, z)
                # failed.append((x, y, z))

    print(made_it_count, failed_count)


def test_everything(speed, tol, iters_each):
    time_njit_aabb = timeit.timeit(
        lambda: test_njit_aabb(speed, tol), number=iters_each)
    print("Raytracing (njit): {:.2f} ms".format(time_njit_aabb * 1000))

    time_aabb = timeit.timeit(lambda: test_aabb(speed, tol), number=iters_each)
    print("Raytracing (jitclass): {:.2f} ms".format(time_aabb * 1000))

    time_raw = timeit.timeit(lambda: test(speed, tol), number=iters_each)
    print("Raw: {:.2f} ms".format(time_raw * 1000))

    print("Raytracing (njit) is {:.2f}% slower than raw.".format(
        100 - (time_raw/time_njit_aabb) * 100))
    print("Raytracing (jitclass) is {:.2f}% slower than raw.".format(
        100 - (time_raw/time_aabb) * 100))
    print("Raytracing (njit) is {:.2f}% slower than Raytracing (jitclass).".format(
        100 - (time_aabb/time_njit_aabb) * 100))


@njit
def single_test(goal):
    proj = Projectile(0.47, 6e-3, 40e-3)
    current = vector.obj(x=0., y=0., z=0.)

    goal_center = goal.get_center()
    yaw = get_yaw_to_target(current, goal_center)
    pitch = get_pitch_wo_drag(current, goal_center, 50)
    vel = dir_from_yaw_pitch_speed(yaw, pitch, 50)
    made_it, points = no_air_res_aabb(
        proj, goal, goal_center, vel, 3, 60, points_back=True)
    points = format_storage(points)

    print(made_it, "\n", points)


if __name__ == "__main__":
    import timeit

    x = 60.
    y = 10.
    z = 60.
    r = 0.05
    v = 50.
    goal = AABB(x - r, y - r, z - r, x + r, y + r, z + r)
    # goal = aabb_from_xyzr(x, y, z, r)
    single_test(goal=goal)
    # allow compiling ^ above.

    # test_aabb(v, r)

    test_everything(v, r, 10)
