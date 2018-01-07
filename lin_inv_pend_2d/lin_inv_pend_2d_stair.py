#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.constants import g

def linear_inv_pend(x, xdot, y, k, zmp, l_pend=2.0, dt=0.001):
    """
    State equation for linear inverted pendulum
    """
    x2dot = g / l_pend * (x - zmp)
    xdot_n = xdot + x2dot * dt
    x_n = x + xdot * dt
    y_n = y + k * xdot * dt
    return x_n, xdot_n, y_n

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    l_pend = 2.0
    x_start = 1e-1
    v_aim = 1.0
    x_end = 0.0
    n_step = 2  # step count
    stair_height = 0.4
    step_len = [2.0 for _ in range(n_step + 1)]
    energy = [-0.5 * g / l_pend * x_start**2,
              0.5 * v_aim**2,
              -0.5 * g / l_pend * x_end**2]

    x_f = [l_pend / (g * s) * (energy[i+1] - energy[i]) + s / 2.0 for i, s in enumerate(step_len[:n_step])]
    print("x_f", x_f)
    xdot_f = [np.sqrt(2.0 * energy[i] + g / l_pend * x**2) for x in x_f]

    p_sup = 0.0  # position of support foot
    x = x_start
    xdot = 0.0
    y = l_pend
    step = 0  # step number
    zmp = 0.0
    x_global = x + p_sup  # Cog in global

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    line, = ax.plot([p_sup, x_global], [0, y], '-o')
    plt.xlim(-1, sum(step_len[:n_step]) + 1)
    plt.ylim(-1, l_pend + stair_height * n_step + 1)
    def update_draw(i):
        global x_global, x, xdot, y, p_sup, step
        x, xdot, y = linear_inv_pend(x, xdot, y, stair_height / step_len[step],
                                     zmp, l_pend)
        if step < n_step and x >= x_f[step]:
            x = x_f[step] - step_len[step]
            xdot = xdot_f[step]
            p_sup += step_len[step]
            step += 1
        x_global = x + p_sup
        print(step, x, xdot)
        line.set_xdata([p_sup, x_global])
        line.set_ydata([step * stair_height, y])
        #if i % 10 == 0:
        #    plt.savefig("img_%04d.png" % i)
        return line,

    ani = animation.FuncAnimation(fig, update_draw,
                                  interval=1, blit=True)
    plt.show()
