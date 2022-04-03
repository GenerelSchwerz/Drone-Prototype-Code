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
        past_x_min.terminal = True
        past_x_mid.direction = 1

        def past_x_max(t, u):
            return u[0] - self.x - self.x_tol
        past_x_max.terminal = True
        past_x_max.direction = 1

        def past_z_min(t, u):
            # The distance passed
            return u[2] - self.z + self.z_tol
        # past_z_min.terminal = True
        past_z_min.direction = -1

        def past_z_mid(t, u):
            # The distance passed
            return u[2] - self.z
        past_z_min.terminal = True
        past_z_mid.direction = -1

        def past_z_max(t, u):
            # The distance passed
            return u[2] - self.z - self.z_tol
        # past_z_max.terminal = True
        past_z_max.direction = -1

        # def max_height(t, u):
        #     # The maximum height is obtained when the z-velocity is zero.
        #     return u[3]
        return (past_x_mid, past_z_mid)
        #return (past_x_min, past_x_mid, past_x_max, past_z_min, past_z_mid, past_z_max)

    @property
    @cache
    def terminate_goals(self):
        return list(filter(lambda goal: hasattr(goal, "terminal"), self.goals))

    def reached_destination(self, soln):
        # x-min, x-max, z-min, z-max
        # events = list(filter(lambda events: len(events), soln.y_events))

        return self.distance_to_sol(soln) < self.tol
        # for event_list in events:
        #     for place in event_list:
        #         dif_x = np.abs(place[0] - self.x)
        #         dif_z = np.abs(place[2] - self.z)
        #         if (dif_x < self.x_tol or np.isclose(dif_x, self.x_tol)) and (dif_z < self.z_tol or np.isclose(dif_z, self.z_tol)):
        #             return True

        # t = np.linspace(0, find_latest_event_time(soln), 100)

        # # Retrieve the solution for the time grid and plot the trajectory.
        # sol = soln.sol(t)

        # places = []
        # for index in range(len(sol[0])):
        #     place = sol[0][index], sol[1][index], sol[2][index], sol[3][index]
        #     places.append(place)

        # return np.hypot(self.x_tol, self.z_tol) > min(map(lambda place: np.hypot((place[0] - self.x), (place[2] - self.z)), places))

        return False

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
        

        # for event_list in events:
        #     found = min(map(lambda index_place: (index_place[0], np.hypot(
        #         (index_place[1][0] - self.x), (index_place[1][2] - self.z))), enumerate(event_list)), key=lambda index_place: index_place[1])
        #     return event_list[found[0]]
        #     for place in event_list:
        #         dif_x = np.abs(place[0] - self.x)
        #         dif_z = np.abs(place[2] - self.z)
        #         if (dif_x < self.x_tol or np.isclose(dif_x, self.x_tol)) and (dif_z < self.z_tol or np.isclose(dif_z, self.z_tol)):
        #             return place

        # t = np.linspace(0, find_latest_event_time(soln), 1000)

        # # Retrieve the solution for the time grid and plot the trajectory.
        # sol = soln.sol(t)

        # places = []
        # for index in range(len(sol[0])):
        #     place = sol[0][index], sol[1][index], sol[2][index], sol[3][index]
        #     places.append(place)

        # found = min(map(lambda index_place: (index_place[0], np.hypot((index_place[1][0] - self.x), (index_place[1][2] - self.z))), enumerate(places)), key = lambda index_place: index_place[1])
        # return places[found[0]]



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
                           * g) / (2 * (self.v0 ** 2))) if self.v0 else 0
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

    def identify_dir(self, org_pitch, test_pitch) -> bool:

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
            # print(org_pitch + pitch, np.degrees(org_pitch + pitch), dist, np.abs(dist - self.dest.tol) < self.dest.tol * 2)
            
            if self.dest.reached_destination(soln):
                # print("made it.")
                # print(soln, "\nMade it!\npitch:", org_pitch + pitch, "\ndist:", self.dest.distance_to_sol(soln))
                return soln
            # print("last:", last_dist, "tmp:", tmp, tmp > last_dist)
            if dist > last_dist:
                print("missing by more now.")
                return soln
            last_dist = dist

            if np.abs(dist - self.dest.tol) < self.dest.tol * 2:
                for step in range(1, midstep_count):
                    soln = new_sim(self.proj, InitialShotInfo(
                        self.v0, org_pitch + pitch + step * midstep), self.dest, self.time_info[0], self.time_info[1])
                    dist = self.dest.distance_to_sol(soln)
                    # print(org_pitch + pitch + step * midstep, np.degrees(org_pitch + pitch + step * midstep), dist, np.abs(dist - self.dest.tol) < self.dest.tol * 3)
                    if self.dest.reached_destination(soln):
                        # print("made it.")
                        # print(soln, "\nMade it!\npitch:", org_pitch + pitch, "\ndist:", self.dest.distance_to_sol(soln))
                        return soln
                    if dist > last_dist:
                        print("missing by more now.")
                        return soln
                    last_dist = dist
    

        # print(soln, "\ndidn't find sol.")
        return

    def calc_shot_with_pitch(self, pitch: np.float):
        soln = new_sim(self.proj, InitialShotInfo(self.v0, pitch),
                       self.dest, self.time_info[0], self.time_info[1])
        if self.dest.reached_destination(soln):
            print(soln, "\npitch:", pitch)
        else:
            print(soln, "\nfailed to get to goal.")
        return soln


def new_sim(proj: Projectile, init: InitialShotInfo, dest: Destination, time_start: int, time_end: int):
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


def test_move(x: np.float, dx, z, dz, allow, v, t, iterations=100, graph=False):
    if graph:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        line_x = np.linspace(0, 100, 100)
        line_z = np.linspace(-30, 30, 100)
        # Returns a tuple of line objects, thus the comma
        line1, = ax.plot(line_x, line_z, 'r-')

    org_x = x
    org_z = z
    times = []
    start_all = time.time()
    end_all = time.time()
    for i in range(iterations):
        x = org_x + dx * (end_all - start_all)
        z = org_z + dz * (end_all - start_all)
        print(x, z)
        proj = Projectile(0.47, 6e-3, 40e-3)
        dest = Destination.from_xzr(x, z, allow)
        shot = ShotInfo(proj, dest, v, t)

        start_sim = time.time()
        soln = shot.debug_smart_shot_pitch()
        end_sim = time.time()
        timey = end_sim - start_sim
        times.append(timey)
        print("finished. sim took {:.2f} seconds.".format(timey))
        if soln and graph:
            print("made it?:", dest.reached_destination(soln))

            print("distance:", dest.distance_to_sol(soln), "m.")
            print("tolerance:", allow, "m")

            new_t = np.linspace(0, find_latest_event_time(soln), 100)

            # Retrieve the solution for the time grid and plot the trajectory.
            sol = soln.sol(new_t)
            new_x, new_z = sol[0], sol[2]
            line1.set_xdata(new_x)
            line1.set_ydata(new_z)
            fig.canvas.draw()
            fig.canvas.flush_events()
        elif graph:
            print("didn't make it.")
        else:
            print("made it but no graph enabled.")

        end_all = time.time()

    try:
        print("finished. Waiting to close. Just ctrl+C.")
        print("Average calc time: {:.2f} ms".format(
            sum(times) / len(times) * 1000))
        print("Moved {:.2f} meters x, {:.2f} meters z.".format(
            org_x - x, org_z - z))
        print("x:", (org_x - x) / (end_all - start_all), "m/s, z:",
              (org_z - z) / (end_all - start_all), "m/s.")
        time.sleep(120)
    except KeyboardInterrupt:
        print("exiting.")


def full_test(tol, v, t, graph=False):
    proj = Projectile(0.47, 6e-3, 40e-3)
    times = []
    if graph:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        line_x = np.linspace(0, 100, 100)
        line_z = np.linspace(-30, 30, 100)
        # Returns a tuple of line objects, thus the comma
        line1, = ax.plot(line_x, line_z, 'r-')



    for x in range(10, 100):
        for z in range(-30, 30):
            print(x, z, v)
            dest = Destination.from_xzr(x, z, tol)
            shot = ShotInfo(proj, dest, v, t)

            start_sim = time.time()
            soln = shot.debug_smart_shot_pitch()

            assert soln, "didn't find solution."
            assert dest.reached_destination(soln), f"{x}, {z}, {v}"

            end_sim = time.time()
            timey = end_sim - start_sim
            times.append(timey)
            print("sim took:",timey)

            if graph:
                print("made it?:", dest.reached_destination(soln))

                print("distance:", dest.distance_to_sol(soln), "m.")
                print("tolerance:", allow, "m")

                new_t = np.linspace(0, find_latest_event_time(soln), 100)

                # Retrieve the solution for the time grid and plot the trajectory.
                sol = soln.sol(new_t)
                new_x, new_z = sol[0], sol[2]
                line1.set_xdata(new_x)
                line1.set_ydata(new_z)
                fig.canvas.draw()
                fig.canvas.flush_events()

    print("Average calc time: {:.2f} ms".format(
            sum(times) / len(times) * 1000))

def single_test(x, z, allow, v, t, graph=False):
    proj = Projectile(0.47, 6e-3, 40e-3)
    dest = Destination.from_xzr(x, z, allow)
    shot = ShotInfo(proj, dest, v, t)

    start_sim = time.time()

    soln = shot.debug_smart_shot_pitch()

    print("made it?:", dest.reached_destination(soln))

    print("distance:", dest.distance_to_sol(soln), "m.")
    print("tolerance:", allow, "m")

    end_sim = time.time()
    print("sim took:", end_sim - start_sim)

    if soln and graph:
        graph_simulation(soln)


if __name__ == "__main__":
    x = 74  # current distance target is away from us in meters.
    dx = 5  # delta movement towards/away from us.
    z = -28  # current distance target is up/down from us.
    dz = 0  # delta of up/down movement.
    allow = 0.02 # allowance of error for aiming, in meters.
    v = 50  # speed of launch. I generalized this, set it to be whatever.
    t = 5  # time in seconds of allowed simulation.
    iterations = 100

    # single_test = actual use-case, can use it to track whatever.
    # single_test(x, z, allow, v, t, graph=True)

    # Demo of moving target, linear movement right now.
    # test_move(x, dx, z, dz, allow, v, t, iterations=iterations, graph=True)

    # full test showcasing range of utility. Tests variety of angles, speeds, etc.
    full_test(allow,v, t, graph=False)
