from functools import cache
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np
import itertools

from numba import float64
from numba.experimental import jitclass

# Air density (kg.m-3), acceleration due to gravity (m.s-2).
rho_air = 1.225
g = 9.81


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


class Destination:
    def __init__(self, goal_x: int, goal_z: int, x_allowance: int, z_allowance: int):
        self.x = goal_x
        self.z = goal_z
        self.x_tol: np.float = np.float64(x_allowance)
        self.z_tol: np.float = np.float64(z_allowance)
        self.tol: np.float = self.x_tol + self.z_tol

    def __str__(self):
        return f"Destination(x: {self.x}, z: {self.z})"

    def from_xzr(x: int, z: int, r: int):
        return Destination(x, z, r, r)

    @property
    @cache
    def goals(self):
        def past_x_mid(t: np.float, u: tuple[np.float, np.float, np.float, np.float]):
            return u[0] - self.x
        # past_x_mid.terminal = True
        past_x_mid.direction = 1

        def past_z_mid(t: np.float, u: tuple[np.float, np.float, np.float, np.float]):
            # The distance passed
            return u[2] - self.z
        # past_z_mid.terminal = True
        past_z_mid.direction = -1
        return (past_x_mid, past_z_mid)

    def reached_destination(self, soln: OdeResult) -> bool:
        return self.distance_to_sol(soln) < self.tol

    def distance_to_sol(self, soln: OdeResult) -> np.float:
        lst = list(itertools.chain(*soln.y_events))
        if not len(lst):
            sol = soln.sol(soln.t)
            places = []
            for index in range(len(sol[0])):
                place = sol[0][index], sol[2][index]
                places.append(place)

            found = min(map(lambda place: np.hypot(
                (place[0] - self.x), (place[1] - self.z)), places))
            return found

        found = min(map(lambda place: np.hypot(
            (place[0] - self.x), (place[2] - self.z)), lst))
        return found

    def closest_point_to(self, soln: OdeResult) -> tuple[np.float, np.float, np.float, np.float]:
        lst = list(itertools.chain(*soln.y_events))
        if not len(lst):
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
