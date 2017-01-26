import numpy as np
import matplotlib.pyplot as plt
from boundary import *

bdy = UnitCircleBoundary()
x0 = np.array([0.7, 0.2])
vel = np.array([3.0/5, 4.0/5])
s0 = bdy.linear_intersect_cart(x0, vel)
if np.any(np.sign(bdy.coords_cart(s0) - x0) != np.sign(vel)):
    # We found the intercept in the wrong direction! Try again.
    s0 = bdy.linear_intersect_param(s0, vel)
collisions = [s0]
for i in range(10):
    tangent = bdy.tangent_cart(collisions[-1])
    vel = 2*np.dot(vel, tangent) * tangent/np.linalg.norm(tangent)**2 - vel
    collisions += [bdy.linear_intersect_param(collisions[-1], vel)]

bdyline = bdy.coords_cart(np.arange(0, 2*np.pi, 0.1))
trajectory = np.c_[x0, bdy.coords_cart(np.array(collisions))]
plt.plot(bdyline[0], bdyline[1])
plt.plot(trajectory[0], trajectory[1])
plt.show()

