"""Classes of 1D convex boundaries providing methods for billiard sims."""
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

    def __init__(self, domain):
        # domain is the period of the parameter (assumed cyclic, 0 origin)
        self.domain = domain

    @abc.abstractmethod
    def _linear_inter_func_d2(self, v):
        """Second derivative of linear intersection function, for second-order
        rootfinding methods."""
        raise NotImplementedError('')

    def linear_intersect_cart(self, x0, v):
        """Using Halley's parabolic root finder"""
        return opt.newton(self._linear_intersect_function(x0, v), np.pi,
                          fprime=self._linear_inter_func_derivative(v),
                          fprime2=self._linear_inter_func_d2(v))

    def linear_intersect_param(self, s0, v):
        """Using Brent's method with quadratic interpolation.
        This method searches in a specified interval, so I can exclude s0."""
        x0 = self.coords_cart(s0)
        s = opt.brentq(self._linear_intersect_function(x0, v),
                       s0 + 0.001*self.domain, s0 + 0.999*self.domain)
        return s % self.domain


class UnitCircleBoundary(ContinuousDifferentiableBoundary_abstract):
    """Circular boundary of unit radius parameterized by angle."""

    def __init__(self):
        super(UnitCircleBoundary, self).__init__(2*np.pi)

    def coords_cart(self, s):
        return np.array([np.cos(s), np.sin(s)])

    def tangent_cart(self, s):
        return np.array([-np.sin(s), np.cos(s)])

    def _linear_inter_func_d2(self, v):
        return lambda s:np.dot(-self.coords_cart(s), np.array([v[1], -v[0]]))

