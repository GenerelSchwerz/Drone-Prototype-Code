import numpy as np
import nbkode



import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from classes import Projectile, Destination, g

class ShotInfo:
    def __init__(self, projectile: Projectile, destination: Destination, v0: int, max_time: int):
        self.proj = projectile
        self.v0: np.float = np.float64(v0)
        self.dest = destination
        self.min_time = 0
        self.max_time = max_time

    def get_pitch_wo_drag(self) -> np.float:
        pitch = np.arctan2(self.dest.z, self.dest.x)
        offset = np.arcsin((np.sqrt(self.dest.z ** 2 + self.dest.x ** 2)
                           * g) / (2 * (self.v0 ** 2))) if self.v0 else 0
        return pitch + offset

    def brute_shot_pitch(self):
        for pitch in np.linspace(np.pi * 0.5, np.pi * -0.5, num=5760):
            soln = new_sim(self.proj, self.dest, self.v0,
                           pitch, self.min_time, self.max_time)
            if self.dest.reached_destination(soln):
                return soln

    def identify_dir(self, org_pitch: np.float) -> bool:
        soln0 = new_sim(self.proj, self.dest, self.v0,
                        org_pitch, self.min_time, self.max_time)

        closest = self.dest.closest_point_to(soln0)
        if np.isclose(self.dest.z, closest[2]):
            check = self.dest.x - closest[0] > 0
        else:
            check = self.dest.z - closest[2] > 0

        return check

    def smart_shot_pitch(self):
        org_pitch = self.get_pitch_wo_drag()
        pitches, step = np.linspace(0, np.pi * (1/12), num=180, retstep=True)
        midstep_count = 4
        midstep = step / midstep_count
        is_pos = self.identify_dir(org_pitch)
        last_dist = 100000

        if not is_pos:
            pitches = np.apply_along_axis(lambda x: -x, 0, pitches)
            midstep = -midstep

        for index, pitch in enumerate(pitches):
            
            soln = new_sim(self.proj, self.dest, self.v0,
                           org_pitch + pitch, self.min_time, self.max_time)
            dist = self.dest.distance_to_sol(soln)
            # print(org_pitch, org_pitch + pitch, dist)
            if self.dest.reached_destination(soln):
                return soln
            if dist > last_dist:
                return soln
            last_dist = dist

            if np.abs(dist - self.dest.tol) < self.dest.tol * 2:
                for step in range(1, midstep_count):
                    
                    soln = new_sim(self.proj, self.dest, self.v0, org_pitch + midstep * (index * midstep_count + step), self.min_time, self.max_time)
                    dist = self.dest.distance_to_sol(soln)
                    # print(org_pitch, org_pitch + midstep * (index * midstep_count + step), dist, "internal")
                    if self.dest.reached_destination(soln):
                        return soln
                    if dist > last_dist:
                        return soln
                    last_dist = dist

        return soln

    def calc_shot_with_pitch(self, pitch: np.float):
        soln = new_sim(self.proj, self.dest, self.v0,
                       pitch, self.min_time, self.max_time)
        if self.dest.reached_destination(soln):
            print(soln, "\npitch:", pitch)
        else:
            print(soln, "\nfailed to get to goal.")
        return soln


def new_sim(proj: Projectile, dest: Destination, v0: int, phi0: int, time_start: int, time_end: int):
    # We can update this to 3D motion, and by extension support horizontal wind, by just doing extra shit.
    # fuckin' look at minecraft code, idiot. Lol.
   def deriv(t: np.float, u: tuple[np.float, np.float, np.float, np.float]):
        x, xdot, z, zdot = u
        speed = np.hypot(xdot, zdot)
        xdotdot = -proj.k_over_m * speed * xdot
        zdotdot = -proj.k_over_m * speed * zdot - g
        return xdot, xdotdot, zdot, zdotdot

   u0 = 0, v0 * np.cos(phi0), 0., v0 * np.sin(phi0)

   sol = nbkode.RungeKutta45(deriv, t0=time_start, y0=u0)
   t, y, t_events, y_events = sol.run_events(100, events=dest.goals)
   soln = {"t": t, "y": y, "t_events": t_events, "y_events": y_events}
   # soln = solve_ivp(deriv, (time_start, time_end), u0, dense_output=True,
   #                   events=dest.goals, method="LSODA")
   print(soln)
   return soln


def find_latest_event_time(soln: OdeResult):
    events = list(filter(lambda events: len(events), soln.t_events))
    if len(events):
        return max(map(lambda x: max(x), events))
    else:
        return soln.t[-1:][0]
