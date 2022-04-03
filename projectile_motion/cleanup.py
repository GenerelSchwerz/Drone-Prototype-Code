from functools import cache
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import itertools


# Air density (kg.m-3), acceleration due to gravity (m.s-2).
rho_air = 1.225
g = 9.81


class Projectile:
    def __init__(self, drag_coefficient: int, radius: int, mass: int):
        # Drag coefficient, projectile radius (m), area (m2) and mass (kg).
        self.c = drag_coefficient
        self.r = radius
        self.a = np.pi * radius**2
        self.m = mass
        # For convenience, define  this constant.
        self.k = 0.5 * self.c * rho_air * self.a
        self.k_over_m = self.k/self.m


class Destination:
    def __init__(self, goal_x: np.float, goal_z: np.float, x_allowance: np.float, z_allowance: np.float):
        self.x = goal_x
        self.z = goal_z
        self.x_tol = np.float64(x_allowance)
        self.z_tol = np.float64(z_allowance)
        self.tol = self.x_tol + self.z_tol

    def __str__(self):
        return f"Destination(x: {self.x}, z: {self.z})"

    def from_xzr(x: np.float, z: np.float, r: np.float):
        return Destination(x, z, r, r)

    @property
    @cache
    def goals(self):
        def past_x_mid(t, u):
            return u[0] - self.x
        past_x_mid.terminal = True
        past_x_mid.direction = 1

        def past_z_mid(t, u):
            # The distance passed
            return u[2] - self.z
        past_z_mid.terminal = True
        past_z_mid.direction = -1
        return (past_x_mid, past_z_mid)
  
    @property
    @cache
    def terminate_goals(self):
        return list(filter(lambda goal: hasattr(goal, "terminal"), self.goals))

    def reached_destination(self, soln):
        return self.distance_to_sol(soln) < self.tol

    def distance_to_sol(self, soln):
        lst = list(itertools.chain(*soln.y_events))
        if not len(lst):
            print(soln, "\n", self.x, self.z)
            sol = soln.sol(soln.t)
            places = []
            for index in range(len(sol[0])):
                place = sol[0][index], sol[1][index], sol[2][index], sol[3][index]
                places.append(place)

            found = min(map(lambda place: np.hypot(
                (place[0] - self.x), (place[2] - self.z)), places))
            return found

        found = min(map(lambda place: np.hypot(
            (place[0] - self.x), (place[2] - self.z)), lst))
        return found

    def closest_point_to(self, soln):
        lst = list(itertools.chain(*soln.y_events))
        if not len(lst):
            print(soln, "\n", self.x, self.z)
            sol = soln.sol(soln.t)
            places = []
            for index in range(len(sol[0])):
                place = sol[0][index], sol[1][index], sol[2][index], sol[3][index]
                places.append(place)

            found = min(map(lambda index_place: (index_place[0], np.hypot(
                (index_place[1][0] - self.x), (index_place[1][2] - self.z))), enumerate(places)), key=lambda index_place: index_place[1])
            return places[found[0]]

        found = min(map(lambda place: (place, np.hypot(
            (place[0] - self.x), (place[2] - self.z))), lst), key=lambda place: place[1])
        return found[0]


class InitialShotInfo:
    def __init__(self, v0: int, phi0: int):
        self.v = v0
        self.phi = phi0

    def to_vec_2(self):
        return (self.v * np.cos(self.phi), self.v * np.sin(self.phi))


class ShotInfo:
    def __init__(self, projectile: Projectile, destination: Destination, v0: np.float, max_time: int):
        self.proj = projectile
        self.v0 = v0
        self.dest = destination
        self.time_info = (0, max_time, max_time - 0)

    def get_pitch_wo_drag(self):
        pitch = np.arctan2(self.dest.z, self.dest.x)
        offset = np.arcsin(np.hypot(self.dest.z, self.dest.x) * g / 2 * (self.v0 **2)) if self.v0 else 0
        return pitch + offset

    def brute_shot_pitch(self):
        for pitch in np.linspace(np.pi * 0.5, np.pi * -0.5, num=5760):
            soln = new_sim(self.proj, InitialShotInfo(self.v0, pitch), self.dest, self.time_info[0], self.time_info[1])
            if self.dest.reached_destination(soln):
                return soln


    def identify_dir(self, org_pitch) -> bool:
        soln0 = new_sim(self.proj, InitialShotInfo(
            self.v0, org_pitch), self.dest, self.time_info[0], self.time_info[1])

        closest = self.dest.closest_point_to(soln0)
        if np.isclose(self.dest.z, closest[2]):
            check = self.dest.x - closest[0] > 0
        else:
            check = self.dest.z - closest[2] > 0

        return check

    def debug_smart_shot_pitch(self):
        org_pitch = self.get_pitch_wo_drag()
        pitches, step = np.linspace(0, np.pi * (1/12), num=180, retstep=True)
        midstep_count = 4
        midstep = step / midstep_count
        is_pos = self.identify_dir(org_pitch, step)
        last_dist = 100000

        if not is_pos:
            pitches = np.apply_along_axis(lambda x: -x, 0, pitches)
            midstep = -midstep

        for pitch in pitches:
            soln = new_sim(self.proj, InitialShotInfo(
                self.v0, org_pitch + pitch), self.dest, self.time_info[0], self.time_info[1])
            dist = self.dest.distance_to_sol(soln)
            if self.dest.reached_destination(soln):
                return soln
            if dist > last_dist:
                return soln
            last_dist = dist

            if np.abs(dist - self.dest.tol) < self.dest.tol * 2:
                for step in range(1, midstep_count):
                    soln = new_sim(self.proj, InitialShotInfo(self.v0, org_pitch + pitch + step * midstep), self.dest, self.time_info[0], self.time_info[1])
                    dist = self.dest.distance_to_sol(soln)
                    if self.dest.reached_destination(soln):
                        return soln
                    if dist > last_dist:
                        return soln
                    last_dist = dist

    def calc_shot_with_pitch(self, pitch: np.float):
        soln = new_sim(self.proj, InitialShotInfo(self.v0, pitch), self.dest, self.time_info[0], self.time_info[1])
        if self.dest.reached_destination(soln):
            print(soln, "\npitch:", pitch)
        else:
            print(soln, "\nfailed to get to goal.")
        return soln


def new_sim(proj: Projectile, init: InitialShotInfo, dest: Destination, time_start: int, time_end: int):
    # We can update this to 3D motion, and by extension support horizontal wind, by just doing extra shit.
    # fuckin' look at minecraft code, idiot. Lol.
    def deriv(t, u):
        x, xdot, z, zdot = u
        speed = np.hypot(xdot, zdot)
        xdotdot = -proj.k_over_m * speed * xdot
        zdotdot = -proj.k_over_m * speed * zdot - g
        return xdot, xdotdot, zdot, zdotdot

    u0 = 0, init.v * np.cos(init.phi), 0., init.v * np.sin(init.phi)

    soln = solve_ivp(deriv, (time_start, time_end), u0, dense_output=True,
                     events=dest.goals)
    return soln


def find_latest_event_time(soln):
    events = list(filter(lambda events: len(events), soln.t_events))
    if len(events):
        return max(map(lambda x: max(x), events))
    else:
        return soln.t[-1:][0]