from functools import cache
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

    def __str__(self):
        return f"x: {self.x}\tz: {self.z}"

    def from_xzr(x: np.float, z: np.float, r: np.float):
        return Destination(x, z, r, r)

    @property
    @cache
    def goals(self):
        def past_x_min(t, u):
            return u[0] - self.x + self.x_tol
        # past_x_min.terminal = True
        past_x_min.direction = 1

        def past_x_mid(t, u):
            return u[0] - self.x
        # past_x_min.terminal = True
        past_x_mid.direction = 1

        def past_x_max(t, u):
            return u[0] - self.x - self.x_tol
        # past_x_max.terminal = True
        past_x_max.direction = 1

        def past_z_min(t, u):
            # The distance passed
            return u[2] - self.z + self.z_tol
        # past_z_min.terminal = True
        past_z_min.direction = -1

        def past_z_mid(t, u):
            # The distance passed
            return u[2] - self.z
        # past_z_min.terminal = True
        past_z_mid.direction = -1

        def past_z_max(t, u):
            # The distance passed
            return u[2] - self.z - self.z_tol
        # past_z_max.terminal = True
        past_z_max.direction = -1

        # def max_height(t, u):
        #     # The maximum height is obtained when the z-velocity is zero.
        #     return u[3]
        return (past_x_min, past_x_mid, past_x_max, past_z_min, past_z_mid, past_z_max)

    @property
    @cache
    def terminate_goals(self):
        return list(filter(lambda goal: hasattr(goal, "terminal"), self.goals))

    def reached_destination(self, soln):
        # x-min, x-max, z-min, z-max
        events = list(filter(lambda events: len(events), soln.y_events))

        for event_list in events:
            for place in event_list:
                dif_x = np.abs(place[0] - self.x)
                dif_z = np.abs(place[2] - self.z)
                if (dif_x < self.x_tol or np.isclose(dif_x, self.x_tol)) and (dif_z < self.z_tol or np.isclose(dif_z, self.z_tol)):
                    return True
        return False

    def distance_to_sol(self, soln):

        events = list(filter(lambda events: len(events), soln.y_events))

        for event_list in events:
            for place in event_list:
                dif_x = np.abs(place[0] - self.x)
                dif_z = np.abs(place[2] - self.z)
                if (dif_x < self.x_tol or np.isclose(dif_x, self.x_tol)) and (dif_z < self.z_tol or np.isclose(dif_z, self.z_tol)):
                    return np.hypot(dif_x, dif_z)

        t = np.linspace(0, find_latest_event_time(soln), 100)

        # Retrieve the solution for the time grid and plot the trajectory.
        sol = soln.sol(t)

        places = []
        for index in range(len(sol[0])):
            place = sol[0][index], sol[1][index], sol[2][index], sol[3][index]
            places.append(place)


        return min(map(lambda place: np.hypot((place[0] - self.x), (place[2] - self.z)), places))


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
        offset = np.arcsin((np.sqrt(self.dest.z ** 2 + self.dest.x ** 2)
                           * g) / (2 * self.v0 ** 2)) if self.v0 else 0
        # print(pitch, offset)
        return pitch + offset

    def brute_shot_pitch(self):
        for pitch in np.linspace(np.pi * 0.5, np.pi * -0.5, num=20000):
            print(pitch, np.degrees(pitch))
            soln = new_sim(self.proj, InitialShotInfo(
                self.v0, pitch), self.dest, self.time_info[0], self.time_info[1])
            if self.dest.reached_destination(soln):
                return soln
        print("didn't find sol.")

    def smart_shot_pitch(self):

        # 0.29697389427528886
        org_pitch = self.get_pitch_wo_drag()
        keep_sign = org_pitch >= np.radians(-20)
        # print(org_pitch, org_pitch * 180/np.pi, keep_sign)
        for pitch in np.linspace(0, np.pi * (1/24), num=90):
            soln = new_sim(self.proj, InitialShotInfo(
                self.v0, org_pitch + pitch), self.dest, self.time_info[0], self.time_info[1])
            if self.dest.reached_destination(soln):
                print("made it.")
                # print(soln, "\nMade it!\npitch:", org_pitch + pitch, "\ndist:", self.dest.distance_to_sol(soln))
                return soln
            soln = new_sim(self.proj, InitialShotInfo(
                self.v0, org_pitch - pitch), self.dest, self.time_info[0], self.time_info[1])
            # print(pitch, np.degrees(pitch), self.dest.distance_to_sol(soln))
            if self.dest.reached_destination(soln):
                print("made it.")
                # print(soln, "\nMade it!\npitch:", org_pitch + pitch, "\ndist:", self.dest.distance_to_sol(soln))
                return soln

        print(soln, "\ndidn't find sol.")
        return soln

    def calc_shot_with_pitch(self, pitch: np.float):
        soln = new_sim(self.proj, InitialShotInfo(self.v0, pitch),
                       self.dest, self.time_info[0], self.time_info[1])
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


def graph_simulation(soln):
    # A fine grid of time points from 0 until impact time.
    t = np.linspace(0, find_latest_event_time(soln), 100)

    # Retrieve the solution for the time grid and plot the trajectory.
    sol = soln.sol(t)
    x, z = sol[0], sol[2]
    print('Range to target, xmax = {:.2f} m'.format(x[-1]))
    print('Maximum height, zmax = {:.2f} m'.format(max(z)))
    plt.plot(x, z)
    plt.xlabel('x /m')
    plt.ylabel('z /m')
    plt.show()


def test(tol):
    proj = Projectile(0.47, 6e-3, 40e-3)
    for x in range(10, 100):
        for z in range(-10, 10):
            v = np.hypot(x, z) * 1.5
            print(x, z, v)
            dest = Destination.from_xzr(x, z, tol)
            shot = ShotInfo(proj, dest, v, 2)

            start_sim = time.time()

            # soln = shot.calc_shot_with_pitch(shot.get_pitch_wo_drag())
            # soln = shot.calc_shot_pitch()
            soln = shot.smart_shot_pitch()

            assert dest.reached_destination(soln), f"{x}, {z}, {v}"

            end_sim = time.time()
            print("sim took:", end_sim - start_sim)


def single_test(x, z, allow, v, t):
    proj = Projectile(0.47, 6e-3, 40e-3)
    dest = Destination.from_xzr(x, z, allow)
    shot = ShotInfo(proj, dest, v, t)

    start_sim = time.time()

    # soln = shot.calc_shot_with_pitch(shot.get_pitch_wo_drag())
    # soln = shot.calc_shot_pitch()
    soln = shot.smart_shot_pitch()
    print(dest.distance_to_sol(soln))

    end_sim = time.time()
    print("sim took:", end_sim - start_sim)

    if soln:
        graph_simulation(soln)


if __name__ == "__main__":
    x = 100
    z = -10
    allow = 0.1
    v = 80
    t = 2

    single_test(x, z, allow, v, t)
    # test(allow)
