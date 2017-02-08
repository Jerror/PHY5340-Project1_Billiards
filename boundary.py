"""Classes of 1D convex boundaries providing methods for billiard sims."""
from __future__ import division
import abc
import numpy as np
import scipy.optimize as opt

class BilliardBoundary_abstract(object):
    """Define 1D convex boundary methods required for billiard sims."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def coords_cart(self, s):
        """Get the cartesian coordinates at parameter value s.
        return np.array([x, y])"""
        raise NotImplementedError('')

    @abc.abstractmethod
    def tangent_cart(self, s):
        """Get the cartesian tangent vector at parameter value s.
        return np.array([x, y])"""
        raise NotImplementedError('')

    @abc.abstractmethod
    def linear_intersect_cart(self, x0, v):
        """Get (the parameter value) at the intersection with given line.
        This variant expects cartesian start x0."""
        raise NotImplementedError('')

    @abc.abstractmethod
    def linear_intersect_param(self, s0, v):
        """Get (the parameter value) at the intersection with given line.
        This variant expects parametric start s0."""
        raise NotImplementedError('')

    def _linear_intersect_function(self, x0, v):
        """Return function whose root gives bdy intersect with given line."""
        return lambda s: np.dot(self.coords_cart(s) - x0,
                                np.array([v[1], -v[0]]))

    def _linear_inter_func_derivative(self, v):
        return lambda s: np.dot(self.tangent_cart(s), np.array([v[1], -v[0]]))


class ContinuousDifferentiableBoundary_abstract(BilliardBoundary_abstract):
    """Billiard methods for 1D continuous differentiable convex boundaries."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, domain, tol=2E-12, maxiter=50,
                 rootfind_open=opt.newton, rootfind_bracketing=opt.brentq,
                 param_rootfind='bracketing'):
        # domain is the period of the parameter (assumed cyclic, 0 origin)
        self.domain = domain
        self.tol = tol
        self.maxiter = maxiter
        self.rf_open = rootfind_open
        self.rf_bracketing = rootfind_bracketing
        self.param_rootfind = param_rootfind           

    @abc.abstractmethod
    def _linear_inter_func_d2(self, v):
        """Second derivative of linear intersection function, for second-order
        rootfinding methods."""
        raise NotImplementedError('')

    def linear_intersect_cart(self, x0, v, s0=0):
        """If fprime2 is not none, uses Halley's parabolic root finder.
        Otherwise, uses the Newton-Raphson method."""
        return self.rf_open(self._linear_intersect_function(x0, v),
                            s0 + self.domain/2,
                            fprime=self._linear_inter_func_derivative(v),
                            fprime2=self._linear_inter_func_d2(v),
                            tol=self.tol, maxiter=self.maxiter,
                            full_output=True)
        if info.converged:
            return s % self.domain
        else:
            raise RuntimeError(repr(info))

    def linear_intersect_param(self, s0, v):
        if self.param_rootfind == 'bracketing':
            return self._linear_intersect_param_bracketing(s0, v)
        elif self.param_rootfind == 'open':
            return self._linear_intersect_param_open(s0, v)
        else: raise RuntimeError('')
 
    def _linear_intersect_param_open(self, s0, v):
        """Using newton's method."""
        return linear_intersect_cart(self.coords_cart(s0), v, s0=s0) 

    def _linear_intersect_param_bracketing(self, s0, v):
        """Using Brent's method with quadratic interpolation.
        This method searches in a specified interval, so I can exclude s0."""
        x0 = self.coords_cart(s0)
        s, info = self.rf_bracketing(self._linear_intersect_function(x0, v),
                                     s0 + 1e-8*self.domain,
                                     s0 + (1 - 1e-8)*self.domain,
                                     xtol=self.tol, maxiter=self.maxiter,
                                     full_output=True)
        if info.converged:
            return s % self.domain
        else:
            raise RuntimeError(repr(info))


class UnitCircleBoundary(ContinuousDifferentiableBoundary_abstract):
    """Circular boundary of unit radius parameterized by angle."""

    def __init__(self, **kwargs):
        super(UnitCircleBoundary, self).__init__(2*np.pi, **kwargs)

    def coords_cart(self, s):
        return np.array([np.cos(s), np.sin(s)])

    def tangent_cart(self, s):
        return np.array([-np.sin(s), np.cos(s)])  # already normalized

    def _linear_inter_func_d2(self, v):
        return lambda s:np.dot(-self.coords_cart(s), np.array([v[1], -v[0]]))


class BeanBoundary(ContinuousDifferentiableBoundary_abstract):
    """Shape defined by r(s) = 1 + a*cos(c*s) + b*sin(d*s)"""

    def __init__(o, a, b, c, d, **kwargs):
        super(BeanBoundary, o).__init__(2*np.pi, **kwargs)
        o.a = a
        o.b = b
        o.c = c
        o.d = d

    def coord_polar(o, s):
        return 1 + o.a * np.cos(o.c * s) + o.b * np.sin(o.d * s)

    def coords_cart(o, s):
        return o.coord_polar(s) * np.array([np.cos(s), np.sin(s)])

    def derivative_polar(o, s):
        return -o.a*o.c * np.sin(o.c * s) + o.b*o.d * np.cos(o.d * s)

    def tangent_cart(o, s):
        tan = (o.derivative_polar(s) * np.array([np.cos(s), np.sin(s)])
               + o.coord_polar(s) * np.array([-np.sin(s), np.cos(s)]))
        return tan / np.linalg.norm(tan)

    def _linear_inter_func_d2(o, v):
        """Not worthwhile to use higher order solver for this boundary."""
        return None

