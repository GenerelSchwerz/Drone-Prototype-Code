import functools
import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import bisect



#basic parametrs
k = 0.01
g = 9.81
v = 100
target = 500

t_init, t_final, step_size = 0, 20, 0.001
t = np.arange(t_init, t_final, step_size)

z = np.zeros([len(t),4])

def model(z, t, params):
    x, y, vx, vy = z
    # x = z[0]
    # y = z[1]
    # vx = z[2]
    # vy = z[3]
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -k*vx
    dvy_dt = -g-k*vy
    dz_dt = np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])
    return dz_dt

@np.vectorize
def diff(theta):
    theta = np.radians(theta)
    params = [k, g, theta]
    
    x0, y0, vx0, vy0 = 0, 0, v*np.cos(theta), v*np.sin(theta)
    z0 = [x0, y0, vx0, vy0]


    sol = odeint(model, z0, t, args=(params,))
    # sol = solve_ivp(model, t, z0, args=(params,))
    # print(sol)
    x, y, vx, vy = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

    y = y[y>=0]
    x = x[:len(y)]
    vx = vx[:len(y)]
    vy = vy[:len(y)]

    xground = x[-2] + y[-2]*(x[-1]-x[-2])/(y[-1]-y[-2])
    diff = xground - target
    return diff

def plot():
    fig, ax = plt.subplots(figsize=(8, 5))
    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

def projectile(angle):
    theta = np.radians(angle)
    params = [k, g, theta]

    
    x0, y0, vx0, vy0 = 0, 0, v*np.cos(theta), v*np.sin(theta)
    z0 = [x0, y0, vx0, vy0]

    sol = odeint(model, z0, t, args=(params,))
    x, y, vx, vy = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

    y = y[y>=0]
    x = x[:len(y)]
    vx = vx[:len(y)]
    vy = vy[:len(y)]
    return x, y




def test():
    plot()
    theta = np.arange(0.1, 90, 0.1)
    differences = diff(theta)

    print(differences[np.abs(differences).argmin()])

    # print(timeit.timeit(functools.partial(diff, theta), number=10))


    # angle1 = bisect(diff, 10, 20)
    # angle2 = bisect(diff, 70, 80)
    # print('\n Angle1 = %0.2f'%angle1,'\n Angle2 = %0.2f'%angle2)
    # diff1 = diff(angle1)
    # diff2 = diff(angle2)
    # print('\n Difference for angle1 = %0.3f'%diff1,'\n Difference for angle2 = %0.11f'%diff2)


    # x1, y1 = projectile(angle1)
    # x2, y2 = projectile(angle2)
    # print(x2.size, y2.size)
    # plt.plot(x1, y1, ls='--', color='purple', label='$\\theta$ = %d$^\circ}$'%angle1)
    # plt.plot(x2, y2, ls='--', color='blue', label='$\\theta$ = %d$^\circ}$'%angle2)
    # plt.plot(500, 0, 'ro', markersize=10)
    # plt.plot(0, 0, 'ko', markersize=10)
    # plt.ylim(0, 500)
    # plt.xlabel('x', fontsize=14)
    # plt.ylabel('y', fontsize=14)
    # plt.title('Projectile Motion', fontsize=16)
    # plt.legend(frameon=False)
    # plt.annotate('Starting point', xy=(0,0), xytext=(50, 100), arrowprops=dict(arrowstyle='->'), fontsize=14)
    # plt.annotate('Target', xy=(500,0), xytext=(350, 100), arrowprops=dict(arrowstyle='->'), fontsize=14)
    # plt.show()


if __name__ == "__main__":
    res = timeit.timeit(test, number=1)
    print(res)
