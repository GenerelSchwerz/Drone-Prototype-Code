from functools import cache
import time
from typing import Tuple, Type
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal as almost_eq

# Drag coefficient, projectile radius (m), area (m2) and mass (kg).
c = 0.47
r = 0.05
A = np.pi * r**2
m = 0.2
# Air density (kg.m-3), acceleration due to gravity (m.s-2).
rho_air = 1.28
g = 9.81
# For convenience, define  this constant.
k = 0.5 * c * rho_air * A

# Initial speed and launch angle (from the horizontal).


class Projectile:
    def __init__(self, drag_coefficient: int, radius: int, mass: int):
        self.c = drag_coefficient
        self.r = radius
        self.a = np.pi * radius**2
        self.m = mass
        self.k = 0.5 * self.c * rho_air * self.a


class Destination:
    def __init__(self, goal_x: np.float, goal_z: np.float, x_allowance: np.float, z_allowance: np.float):
        self.x = goal_x
        self.z = goal_z
        self.x_tol = np.float64(x_allowance + 0.0000000000000001)
        self.z_tol = np.float64(z_allowance + 0.0000000000000001)

    def from_xzr(x: np.float, z: np.float, r: np.float):
        return Destination(x, z, r, r)

    @property
    @cache
    def goals(self):
        def past_x_min(t, u):
            return u[0] - self.x + self.x_tol
        # past_x_min.terminal = True
        past_x_min.direction = 1

        def past_x_max(t, u):
            return u[0] - self.x - self.x_tol
        past_x_max.terminal = True
        past_x_max.direction = 1

        def past_z_min(t, u):
            # The distance passed
            return u[2] - self.z + self.z_tol
        # past_z_min.terminal = True
        past_z_min.direction = -1

        def past_z_max(t, u):
            # The distance passed
            return u[2] - self.z - self.z_tol
        past_z_max.terminal = True
        past_z_max.direction = -1

        def max_height(t, u):
            # The maximum height is obtained when the z-velocity is zero.
            return u[3]
        return (past_x_min, past_x_max, past_z_min, past_z_max, max_height)

    @property
    @cache
    def terminate_goals(self):
        return list(filter(lambda goal: hasattr(goal, "terminal"), self.goals))

    def reached_destination(self, soln):
        # x-min, x-max, z-min, z-max
        events = list(filter(lambda events: len(events), soln.y_events[:4]))

        for event_list in events:
            list(map(lambda place: print(np.float64(np.abs(
                place[0] - self.x)), self.x_tol, np.float64(np.abs(place[2] - self.z)), self.z_tol), event_list))
            print(list(map(lambda place: print((np.abs(
                place[0] - self.x), self.x_tol, np.abs(place[2] - self.z)), self.z_tol), event_list)))

        return len(events) and any(map(lambda event: any(map(lambda place: np.float64(np.abs(place[0] - self.x)) < self.x_tol and np.float64(np.abs(place[2] - self.z)) < self.z_tol, event)), events))

        # def within_allowance(index):
        #     not not len(soln.t_events[index]) and not len(soln.t_events[index - 1])

        # # list(map(lambda info: print(info[1](0, soln.y_events[info[0]][0])), enumerate(self.terminate_goals)))
        # return all(map(lambda info: within_allowance(info[0]), enumerate(self.terminate_goals)))


class InitialShotInfo:
    def __init__(self, v0: int, phi0: int):
        self.v = v0
        self.phi = phi0


class ShotInfo:
    def __init__(self, projectile: Projectile, initial_info: InitialShotInfo, destination: Destination, time_start: int, time_end: int):
        self.proj = projectile
        self.initial = initial_info
        self.dest = destination
        self.time_info = (time_start, time_end, time_end - time_start)


def new_sim(proj: Projectile, init: InitialShotInfo, dest: Destination, time_start: int, time_end: int):
    def deriv(t, u):
        x, xdot, z, zdot = u
        speed = np.hypot(xdot, zdot)
        xdotdot = -proj.k/m * speed * xdot
        zdotdot = -proj.k/m * speed * zdot - g
        return xdot, xdotdot, zdot, zdotdot

    u0 = 0, init.v * np.cos(init.phi), 0., init.v * np.sin(init.phi)

    soln = solve_ivp(deriv, (time_start, time_end), u0, dense_output=True,
                     events=dest.goals)

    print(dest.reached_destination(soln))
    return soln



def find_latest_event_time(soln):
    events = list(filter(lambda events: len(events), soln.t_events))
    return max(list(map(lambda x: max(x), events)))


def graph_simulation(soln):
    print(soln)
    print('Time to target = {:.2f} s'.format(soln.t_events[0][0]))
    print('Time to highest point = {:.2f} s'.format(soln.t_events[1][0]))

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


if __name__ == "__main__":
    v0 = 50
    phi0 = np.radians(65)
    start_sim = time.time()

    proj = Projectile(0.47, 0.05, 0.2)
    init = InitialShotInfo(50, np.radians(65))
    # dest = Destination(64.115772, 1, 0, 1.1)
    dest = Destination.from_xzr(64.115772, 1, 0.3)

    # 6.33623381

    soln = new_sim(proj, init, dest, 0, 50)
    # soln = simulate(v0, phi0)
    end_sim = time.time()
    print("sim took:", end_sim - start_sim)
    graph_simulation(soln)
