import numpy as np
from boundary import *

bdy = UnitCircleBoundary()
x0 = np.array([0, 0])
vel = np.array([3.0/5, 4.0/5])
s0 = bdy.linear_intersect_cart(x0, vel)
if np.any(np.sign(bdy.coords_cart(s0) - x0) != np.sign(vel)):
    # We found the intercept in the wrong direction! Try again.
    s0 = bdy.linear_intersect_param(s0, vel)
collisions = [s0]
for i in range(10):
    tangent = bdy.tangent_cart(collisions[-1])
    vel = 2*np.dot(vel, tangent) * tangent/np.linalg.norm(tangent)**2 - vel
    print vel
    collisions += [bdy.linear_intersect_param(collisions[-1], vel)]

print collisions

