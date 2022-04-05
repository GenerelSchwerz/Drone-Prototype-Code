import functools
import math
import time
import matplotlib.pyplot as plt
import vector
import numpy as np
import awkward as ak  # at least version 1.2.0
import numba as nb
from numba import njit, jit, vectorize, float32, float64
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
    def __init__(self, drag_coefficient: float, radius: float, mass: float):
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
    def __init__(self, goal_x: float, goal_y: float, goal_z: float, x_allowance: float, y_allowance: float, z_allowance: float):
        self.x = goal_x
        self.y = goal_y
        self.z = goal_z
        self.x_tol: float = np.float64(x_allowance)
        self.y_tol: float = np.float64(y_allowance)
        self.z_tol: float = np.float64(z_allowance)
        self.tol: float = math.sqrt(
            self.x_tol ** 2 + self.y_tol ** 2 + self.z_tol ** 2)

    def __str__(self):
        return f"Destination(x: {self.x}, z: {self.z})"

    def from_xzr(x: int, z: int, r: int):
        return Destination(x, z, r, r)


@njit()
def get_yaw_to_target(src: Vector3D, dest: Vector3D):
    return math.atan2(dest.x - src.x,  dest.z - src.x)


@njit  # no par
def add_with_scale(org: Vector3D, plus: Vector3D, scalar: float) -> Vector3D:
    return org.add(plus.scale(scalar))


@njit(fastmath=False) # no par
def offset_creation(vel: Vector3D, k_over_m: float):
    # offsets: np.ndarray = np.zeros(3)
    # offsets.itemset(0, -vel[0] * resistances[0]), offsets.itemset(1, -vel[1] * resistances[1] - g), offsets.itemset(2, -vel[2] * resistances[2])
    # print(-vel[0] * resistances[0])
    return vector.obj(x=-vel.x * k_over_m, y=-vel.y * k_over_m - g, z=-vel.z * k_over_m)


# May potentially skip. This is bad practice.
@njit(fastmath=False)  # no par
def made_it(goal: Vector3D, current: Vector3D, next: Vector3D, tolerance: float):
    # flag1 = math.sqrt((goal.x - current.x) ** 2 +
    #                 (goal.z - current.z) ** 2) < tolerance
    # flag2 = math.sqrt((goal.x - next.x) ** 2 +
    #                 (goal.z - next.z) ** 2) < tolerance
    # flag3 = current.y > next.y
    # return flag1 and flag2 and flag3
    return distance_to(current, goal) < tolerance


@njit(fastmath=False, cache=True)
def distance_to(src: Vector3D, dest: Vector3D):
    return math.sqrt((src.x - dest.x) ** 2 + (src.y - dest.y) ** 2 + (src.z - dest.z) ** 2)


@njit(fastmath=False, cache=True)
def distance_to_arr(src: Vector3D, dest: np.ndarray):
    return math.sqrt((src.x - dest[0]) ** 2 + (src.y - dest[1]) ** 2 + (src.z - dest[2]) ** 2)


@njit(cache=True)
def dir_from_yaw_pitch_speed(yaw, pitch, speed) -> vector.VectorNumpy3D:
    if speed:
        x = speed * np.sin(yaw)
        y = speed * np.sin(pitch)
        z = speed * np.cos(yaw)
        vx_mag = np.hypot(x, z)
        vx_rat = math.sqrt(vx_mag * vx_mag - y * y)
        
        all_rat = vx_rat / vx_mag
        return vector.obj(x=x * all_rat, y=y, z=z * all_rat)
    else:
        return vector.obj(x=0., y=0., z=0.)


@njit(fastmath=False, cache=True)
def get_pitch_wo_drag(src: Vector3D, dest: Vector3D, v0s: float) -> float:
    pitch = math.atan2(
        dest.y - src.y, math.sqrt((dest.x - src.x) ** 2 + (dest.z - src.z) ** 2))
    offset = math.asin((math.sqrt((dest.z - src.z) ** 2 + (dest.x - src.x) ** 2)
                       * g) / (2 * (v0s ** 2))) if v0s else 0
    if math.isnan(offset): 
        # print("src: ({:f} {:f} {:f}), dest: ({:f} {:f} {:f}), v0s: {:f}".format(src.x, src.y, src.z, dest.x, dest.y, dest.z, v0s))
        raise Exception("Will not make this shot. Hands down.")
    return pitch + offset


@njit()
def sim(src: Vector3D, proj: Projectile, dest: AABB, dest_center: Vector3D, vel: Vector3D, max_time: float, fps: float):
    total_frames = max_time * fps
    inv_fps = 1. / fps
    current_pos = src
    next_pos: VectorObject3D = add_with_scale(current_pos, vel, inv_fps)
    goal_center = dest.get_bottom_center()

    for frame in range(total_frames):
        offsets = offset_creation(vel, proj.k_over_m)



        # this is apparently missing sometimes. What the **fuck?**
        if dest.intersects_segment(current_pos, next_pos):
            return True, current_pos

        if vel.y < 0 and current_pos.y < goal_center.y and (current_pos.x > goal_center.x and current_pos.z > goal_center.z):
            return False, current_pos
        
        if distance_to(next_pos, dest_center) > distance_to(current_pos, dest_center) and vel.y < 0:
            return False, current_pos
        


        # print("current:", current_pos, "next:", next_pos)
        current_pos = add_with_scale(current_pos, vel, inv_fps)
        vel = add_with_scale(vel, offsets, inv_fps)
        next_pos = add_with_scale(current_pos, vel, inv_fps)

    return False, current_pos


@njit()
def sim_with_points(src: Vector3D, proj: Projectile, goal: AABB, goal_center: Vector3D, vel: Vector3D, max_time: float, fps: float):
    total_frames = max_time * fps
    inv_fps = 1. / fps
    current_pos = src
    next_pos: VectorObject3D = add_with_scale(current_pos, vel, inv_fps)
    storage: np.ndarray = np.zeros(total_frames * 3, dtype=float64)
    goal_center = goal.get_bottom_center()

    for frame in range(total_frames):
        storage[3 * frame] = next_pos.x
        storage[3 * frame + 1] = next_pos.y
        storage[3 * frame + 2] = next_pos.z

        offsets = offset_creation(vel, proj.k_over_m)

        

        
        # this is apparently missing sometimes. What the **fuck?**
        if goal.intersects_segment(current_pos, next_pos):
            return True, storage

        if vel.y < 0 and current_pos.y < goal_center.y and (current_pos.x > goal_center.x and current_pos.z > goal_center.z):
            return False, storage
        
        if distance_to(next_pos, goal_center) > distance_to(current_pos, goal_center) and vel.y < 0:
            return False, storage

 

        # print("current:", current_pos, "next:", next_pos)
        current_pos = add_with_scale(current_pos, vel, inv_fps)
        vel = add_with_scale(vel, offsets, inv_fps)
        next_pos = add_with_scale(current_pos, vel, inv_fps)

    return False, storage


@njit
def find_iter_dir(src: Vector3D, proj: Projectile, dest: AABB, dest_center: Vector3D, vel: Vector3D, max_time: float, fps: float):
    res, points = sim_with_points(
        src, proj, dest, dest_center, vel, max_time, fps)
    points = format_storage(points)
    vfunc = np.array([distance_to_arr(dest_center, p) for p in points])

    # print(points[vfunc.argmin()][1], dest_center.y, distance_to_arr(dest_center, points[vfunc.argmin()]))
    if res:
        return not res, points[vfunc.argmin()][1] > dest_center.y, points
    else:
        return not res, points[vfunc.argmin()][1] > dest_center.y, points


@njit(fastmath=False)
def no_air_res_aabb(src: Vector3D, proj: Projectile, dest: AABB, speed: float, yaw: float, org_pitch: float, max_time: float, fps: float, points_back=False) -> tuple[bool, np.ndarray]:
    vel = dir_from_yaw_pitch_speed(yaw, org_pitch, speed)
    dest_center = dest.get_bottom_center()
    needs_change, should_pos, points = find_iter_dir(
        src, proj, dest, dest_center, vel, max_time, fps)

    def_p = np.zeros((1, 3), dtype=np.float64)

    if needs_change:
        # print("fuck")
        # print(should_pos)
        mult = 1/48
        pitches = np.linspace(
            0, (np.pi * mult if should_pos else np.pi * -mult), 91)
        step = np.pi * mult * 1/90
        midstep_count = 3
        midstep = step / midstep_count
        last_dist = np.Inf

        for index, pitch in enumerate(pitches):
            vel = dir_from_yaw_pitch_speed(
                yaw, org_pitch + pitch, speed)
            res, nearest = sim(src, proj, dest, dest_center,
                               vel, max_time, fps)
            # dist = distance_to(dest_center, nearest)
            if res:
                if points_back:
                    vel = dir_from_yaw_pitch_speed(
                        yaw, org_pitch + pitch, speed)
                    res, p = sim_with_points(src, proj, dest, dest_center,
                                             vel, max_time, fps)
                    p = format_storage(p)
                    return True, p
                else:
                    return True, def_p

            vel = dir_from_yaw_pitch_speed(
                yaw, org_pitch - pitch, speed)
            res, nearest1 = sim(src, proj, dest, dest_center,
                                vel, max_time, fps)
            # dist1 = distance_to(dest_center, nearest1)
            if res:
                if points_back:
                    vel = dir_from_yaw_pitch_speed(
                        yaw, org_pitch - pitch, speed)
                    res, p = sim_with_points(src, proj, dest, dest_center,
                                             vel, max_time, fps)
                    p = format_storage(p)
                    return True, p
                else:
                    return True, def_p

            # if dist < dist1:
            #     dist = dist
            #     display = pitch
            #     nearest = nearest
            # else:
            #     dist = dist1
            #     display = -pitch
            #     nearest = nearest1
            # print(org_pitch + display, dist, dist < 0.1 * 2)
          
            # if dist > last_dist and nearest.y < dest_center.y:
            #     return False, def_p

            # if dist < 0.3:
            #     debug = False
            #     if dist > last_dist:
            #         debug = True
            #         print(dist, last_dist)
            #         return False, def_p
            #     last_dist = dist

                # for step in range(1, midstep_count):
                #     vel = dir_from_yaw_pitch_speed(
                #         yaw, org_pitch + pitch + midstep * (index * midstep_count + step), speed)
                #     res, nearest = sim(src, proj, dest, dest_center,
                #                        vel, max_time, fps)
                #     dist = distance_to(dest_center, nearest)
                #     if res:
                #         if debug:
                #             print("made it:", dist)
                #         if points_back:
                #             vel = dir_from_yaw_pitch_speed(
                #                 yaw, org_pitch + pitch + midstep * (index * midstep_count + step), speed)
                #             res, p = sim_with_points(src, proj, dest, dest_center,
                #                                      vel, max_time, fps)
                #             p = format_storage(p)
                #             return True, p
                #         else:
                #             return True, def_p

                #     vel = dir_from_yaw_pitch_speed(
                #         yaw, org_pitch - pitch - midstep * (index * midstep_count + step), speed)
                #     res, nearest1 = sim(src, proj, dest, dest_center,
                #                         vel, max_time, fps)
                #     dist1 = distance_to(dest_center, nearest1)
                #     if res:
                #         if points_back:
                #             vel = dir_from_yaw_pitch_speed(
                #                 yaw, org_pitch - pitch - midstep * (index * midstep_count + step), speed)
                #             res, p = sim_with_points(src, proj, dest, dest_center,
                #                                      vel, max_time, fps)
                #             p = format_storage(p)
                #             return True, p
                #         else:
                #             return True, def_p

                #     if dist < dist1:
                #         dist = dist
                #         display = midstep * (index * midstep_count + step)
                #         near = nearest
                #     else:
                #         dist = dist1
                #         display = - midstep * (index * midstep_count + step)
                #         near = nearest1

                    # print(org_pitch + display, dist)
                    # if dist > last_dist:
                    #     # print(near)
                    #     return False
                    # last_dist = dist

    else:
        # print("made it")
        return True, points

    return False, points


@njit()
def format_storage(storage: np.ndarray):
    wanted_size = storage.size / 3
    if wanted_size % 3:
        raise Exception("Bad storage.")
    return np.reshape(storage, (int(storage.size / 3), 3))




def test_aabb_graph(speed, tol, time, fps, debug=False, graph=False):
    # import matplotlib.pyplot as plt
    made_it_count = 0
    failed_count = 0

    if graph:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        line_x = np.linspace(0, 100, 100)
        line_z = np.linspace(-30, 30, 100)
        # Returns a tuple of line objects, thus the comma
        line1, = ax.plot(line_x, line_z, 'r-')



    proj = Projectile(0.47, 6e-3, 40e-3)
    for x in range(10, 100):  # range(10, 100):
        z = 0
        # for z in range(10, 100):
        for y in range(-30, 30):  # range(-30, 30):
            # y = 0.
            current = vector.obj(x=0., y=0., z=0.)
            goal = AABB(x - tol, y - tol, z - tol, x + tol, y + tol, z + tol)
            goal_center = goal.get_center()
            yaw = get_yaw_to_target(current, goal_center)
            pitch = get_pitch_wo_drag(current, goal_center, speed)
            vel = dir_from_yaw_pitch_speed(yaw, pitch, speed)
            made_it, points = no_air_res_aabb(
                current, proj, goal, speed, yaw, pitch, time, fps, points_back=False)
            # print(x, y, z, made_it)
            if made_it:
                made_it_count += 1
            else:
                failed_count += 1
                if debug:
                    print(x, y, z, made_it)

            if graph:
                size = points.size
                points = points.flat
                new_x = np.array([math.sqrt(points[int(3 * i)] ** 2 + points[int(3 * i + 2)] ** 2) for i in range(0, int(size / 3))])
                new_y = np.array([points[int(3 * i + 1)] for i in range(0, int(size / 3))])
                line1.set_xdata(new_x)
                line1.set_ydata(new_y)
                fig.canvas.draw()
                fig.canvas.flush_events()

    print(made_it_count, failed_count)


@njit()
def test_aabb(speed, tol, time, fps, debug=False):
    # import matplotlib.pyplot as plt
    made_it_count = 0
    failed_count = 0
    x_range = 80
    z_range = 0
    y_range = 30
    proj = Projectile(0.47, 6e-3, 40e-3)
    for y in range(-y_range, y_range):
        for x in range(0, x_range):
            # for z in range(-z_range, z_range):
                z = 1
                if x == 0 and y == 0 and z == 0: 
                    continue
              
                current = vector.obj(x=0., y=0., z=0.)
                goal = AABB(x - tol, y - tol, z - tol, x + tol, y + tol, z + tol)
                goal_center = goal.get_center()
                yaw = get_yaw_to_target(current, goal_center)
                pitch = get_pitch_wo_drag(current, goal_center, speed)
                vel = dir_from_yaw_pitch_speed(yaw, pitch, speed)
                made_it, points = no_air_res_aabb(
                    current, proj, goal, speed, yaw, pitch, time, fps, points_back=False)
                # print(x, y, z, made_it)
                if made_it:
                    made_it_count += 1
                else:
                    failed_count += 1
                    if debug:
                        print(x, y, z, made_it)

    print(made_it_count, failed_count)


def test_everything(speed, tol, time, fps, iters_each, debug=False):
    iters = 80 * 30 * 2
    time_aabb = timeit.timeit(functools.partial(test_aabb,
        speed, tol, time, fps, debug), number=iters_each)
    print("Raytracing (jitclass): {:.2f} ms".format(time_aabb * 1000))
    print("fps: {:.2f}".format(iters * iters_each / time_aabb))
    print("Average calc time: {:f} ms".format(
        (time_aabb / iters * iters_each)))



def single_test(goal, speed, time, fps, debug=False, graph=False):
    proj = Projectile(0.47, 6e-3, 40e-3)
    current = vector.obj(x=0., y=0., z=0.)

    goal_center = goal.get_center()
    yaw = get_yaw_to_target(current, goal_center)
    pitch = get_pitch_wo_drag(current, goal_center, speed)
    vel = dir_from_yaw_pitch_speed(yaw, pitch, speed)
    made_it, points = no_air_res_aabb(
        current, proj, goal, speed, yaw, pitch, time, fps, points_back=True)
    # points = format_storage(points)

    if graph:
        size = points.size
        points = points.flat
        # x = np.array([1 for i in range(points.size, 3)])
        x = np.array([math.sqrt(points[int(3 * i)] ** 2 + points[int(3 * i + 2)] ** 2) for i in range(0, int(size / 3))])
        y = np.array([points[int(3 * i + 1)] for i in range(0, int(size / 3))])
        plt.plot(x, y)
        plt.xlabel('x /m')
        plt.ylabel('y /m')
        plt.show()

    if debug:
        print(made_it, "\n", points)


if __name__ == "__main__":
    import timeit

    x = 10
    y = -15
    z = 10
    r = 0.03
    v = 40.
    t = 3
    fps = 165
    goal = AABB(x - r, y - r, z - r, x + r, y + r, z + r)
    # goal = aabb_from_xyzr(x, y, z, r)
    single_test(goal, v, t, fps, debug=False, graph=False)

    # test_aabb(v, r, t, fps, debug=False)
    test_everything(v, r, t, fps, 5, debug=False)
