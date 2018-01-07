#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.constants import g
XY = 2
XYZ = 3
dt = 0.001

def linear_inv_pend(x, xdot, p_star, l_pend=1.0, dt=0.001):
    """
    State equation for linear inverted pendulum
    """
    x2dot = g / l_pend * (x - p_star)
    xdot_n = xdot + x2dot * dt
    x_n = x + xdot * dt
    return x_n, xdot_n

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation
    l_pend = 0.8
    t_sup = 0.8
    t_c = np.sqrt(l_pend / g)
    ch = np.cosh(t_sup / t_c)
    sh = np.sinh(t_sup / t_c)
    a = 10.0
    b = 1.0
    d = a * (ch - 1.0)**2 + b * (sh / t_c)**2
    step_table = np.array([[0.0, 0.2],
                           [0.3, 0.2],
                           [0.3, 0.2],
                           [0.3, 0.2],
                           [0.0, 0.2]])
    n_step = step_table.shape[0]
    x = np.array([1e-3, 1e-3])
    xdot = np.array([0.0, 0.0])
    p_sup = np.array([0.0, 0.0])
    p_star = np.array([0.0, 0.0])

    t = 0
    step = 0  # step number

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line, = ax.plot([p_star[0], x[0]],
                    [p_star[1], x[1]],
                    [0, l_pend], '-o')
    ax.set_xlim3d(-0.5, sum(step_table[:, 0]) * 1.2)
    ax.set_ylim3d(-0.5, sum(step_table[:, 1]) * 1.2)
    ax.set_zlim3d(-0.5, l_pend * 1.2)
    def update_draw(i):
        global t, x, xdot, p_sup, p_star, step
        t = i * dt
        if step < n_step and t >= step * t_sup:
            p_sup += np.array([step_table[step][0],
                               -(-1.0)**step * step_table[step][1]])
            x_bar = np.array([step_table[step+1][0] * 0.5,
                              (-1.0)**step * step_table[step+1][1] * 0.5])
            v_bar = np.array([(ch + 1) / (t_c * sh * x_bar[0]),
                              (ch - 1) / (t_c * sh * x_bar[1])])
            xd = p_sup + x_bar
            xd_dot = v_bar
            p_star = -a * (ch - 1.0) / d * (xd - ch * x - t_c * sh * xdot) - b * sh / (t_c * d) * (xd_dot - sh / t_c * x - ch * xdot)
            step += 1
        x, xdot = linear_inv_pend(x, xdot, p_star, l_pend, dt)
        print(t, step, x, xdot, p_sup, p_star)
        line.set_xdata([p_star[0], x[0]])
        line.set_ydata([p_star[1], x[1]])
        line.set_3d_properties([0.0, l_pend])
        #if i % 10 == 0:
        #    plt.savefig("img_%04d.png" % i)
        return line,

    ani = animation.FuncAnimation(fig, update_draw,
                                  interval=1)
    plt.show()
