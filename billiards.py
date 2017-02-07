import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import rootfinding as rf
from boundary import *

default_bounces = 10
max_bounces = 500
default_deltay = 0
max_deltay = 1000

r0 = np.array([0.6, 0.2])
theta = 0.1
v0 = np.array([np.cos(theta), np.sin(theta)])

custom_rf = {'rootfind_open': rf.newton, 'rootfind_bracketing': rf.bisect}
# custom_rf = {}

# bdy = UnitCircleBoundary(**custom_rf)
bdy = BeanBoundary(0.16, 0.1, 2.0, 1.0, **custom_rf)


def billiard_propagator_gen(r, v, bdy):
    s = bdy.linear_intersect_cart(r, v)
    if np.any(np.sign(bdy.coords_cart(s) - r) != np.sign(v)):
        # Then we found the intercept in the wrong direction! Try again.
        s = bdy.linear_intersect_param(s, v)  # Excludes passed root s
    yield s
    while True:
        tangent = bdy.tangent_cart(s)
        v = 2*np.dot(v, tangent) * tangent/np.linalg.norm(tangent)**2 - v
        s = bdy.linear_intersect_param(s, v)
        yield s


def init_state(r, v, boundary):
    state = {'collisions': [],
             'trajectory': np.c_[r],
             'propagator': billiard_propagator_gen(r, v, bdy)}
    return state


def propagate(state, bounces):
    new_collisions = [state['propagator'].next() for _ in xrange(bounces)]
    state['collisions'].append(new_collisions)
    state['trajectory'] = np.c_[state['trajectory'],
                                bdy.coords_cart(np.array(new_collisions))]


bdyline = bdy.coords_cart(np.arange(0, 2*np.pi, 0.01))
plt.plot(bdyline[0], bdyline[1])
# Plot a line object "l"; I'll send data to it later 
traj_line, = plt.plot([0, 0], [0, 0], label='Trajectory')  
plt.axis('equal')

# Make Sliders (with their own axis artists) for interactive control.
axcolor = 'lightgoldenrodyellow'
axbounces = plt.axes([0.2, 0.03, 0.65, 0.02], axisbg=axcolor)
sbounces = Slider(axbounces, 'Number of bounces',
                  1, max_bounces, valinit=default_bounces)
axdeltay = plt.axes([0.2, 0.005, 0.65, 0.02], axisbg=axcolor)
sdeltay = Slider(axdeltay, r'$\Delta y_0$ / $(y_0 * 10^{-8})$',
                 -max_deltay, max_deltay, valinit=default_deltay)


def propagate_plotline(state, n, plotline):
    missing = n+1 - state['trajectory'].shape[1]
    if missing <= 0:
        plotline.set_xdata(state['trajectory'][0, 0:n+1])
        plotline.set_ydata(state['trajectory'][1, 0:n+1])
    else:
        propagate(state, missing)    
        plotline.set_xdata(state['trajectory'][0])
        plotline.set_ydata(state['trajectory'][1])
    plt.draw()


def reset_plotline(r, v, boundary, n, plotline):
    state = init_state(r, v, boundary)
    propagate_plotline(state, n, plotline)
    return state

# Initialize plotline data for default Slider settings.
state = reset_plotline(r0 * np.array([1, 1 + sdeltay.val*1e-8]), v0, bdy,
                       int(sbounces.val), traj_line)
# Configure our Sliders to update the plotlines.
# The Sliders will pass their current value to the on_changed event callback
sbounces.on_changed(lambda val: propagate_plotline(state, int(val), traj_line))
sdeltay.on_changed(lambda val: reset_plotline(r0 * np.array([1, 1 + val*1e-8]),
                                              v0, bdy, int(sbounces.val),
                                              traj_line))

plt.show()

