import time
from simulator import *
import matplotlib.pyplot as plt

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
        soln = shot.smart_shot_pitch()
        end_sim = time.time()
        timey = end_sim - start_sim
        times.append(timey)
        print("simulation took {:.2f} ms.".format(timey * 1000))
        if soln and graph:
            print("made it?:", dest.reached_destination(soln))
            print("distance from target: {:.2f} cm.".format(dest.distance_to_sol(soln) * 100))
       


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
        total = end_all - start_all
        avg = sum(times) / len(times) 
        print("Average calc time: {:.2f} ms".format(avg * 1000))
        print("fps: {:.2f}".format(1 / avg))
        print("Moved {:.2f} meters x ({:.2f} m/s), {:.2f} meters z ({:.2f} m/s).".format(org_x - x, (org_x - x) / total, org_z - z, (org_z - z) / total))
        # print("x:", (org_x - x) / (end_all - start_all), "m/s, z:",
        #       (org_z - z) / (end_all - start_all), "m/s.")
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

    for x in range(100, 10, -1):
        for z in range(-30, 30):
            print(x, z, v)
            dest = Destination.from_xzr(x, z, tol)
            shot = ShotInfo(proj, dest, v, t)

            start_sim = time.time()
            soln = shot.smart_shot_pitch()

            assert soln, "didn't find solution."
            assert dest.reached_destination(soln), f"{x}, {z}, {v}"

            end_sim = time.time()
            timey = end_sim - start_sim
            times.append(timey)
            print("simulation took {:.2f} ms.".format(timey * 1000))
            if graph:
                print("made it?:", dest.reached_destination(soln))
                print("distance from target: {:.2f} cm.".format(dest.distance_to_sol(soln) * 100))

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
    soln = shot.smart_shot_pitch()

    end_sim = time.time()

    print("simulation took {:.2f} ms.".format((end_sim - start_sim) * 1000))
    print("made it?:", dest.reached_destination(soln))
    print("distance from target: {:.2f} cm.".format(dest.distance_to_sol(soln) * 100))

    if soln and graph:
        graph_simulation(soln)




if __name__ == "__main__":
    x = 10  # current distance target is away from us in meters.
    dx = -30  # delta movement towards/away from us.
    z = 10  # current distance target is up/down from us.
    dz = 0  # delta of up/down movement.
    allow = 0.1 # allowance of error for aiming, in meters.
    v = 50  # speed of launch. I generalized this, set it to be whatever.
    t = 3  # time in seconds of allowed simulation.
    iterations = 50

    # single_test = actual use-case, can use it to track whatever.
    # single_test(x, z, allow, v, t, graph=False)

    # Demo of moving target, linear movement right now.
    # test_move(x, dx, z, dz, allow, v, t, iterations=iterations, graph=True)

    # full test showcasing range of utility. Tests variety of angles, speeds, etc.
    full_test(allow,v, t, graph=True)
