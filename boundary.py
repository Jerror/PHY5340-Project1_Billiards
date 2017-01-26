"""Classes of 1D convex boundaries providing methods for billiard sims."""
import abc
import numpy as np

class BilliardBoundary_abstract(object):
    """Define 1D convex boundary methods required for billiard sims."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def coords_cart(self, s):
        """Get the cartesian coordinates at parameter value s.        
        return np.array([x, y])"""
        raise NotImplementedError('')

    @abc.abstractmethod
    def linear_intersect(self, start, direction):
        """Get (the parameter value) at the intersection with given line.
        return s"""
        raise NotImplementedError('')

    @abc.abstractmethod
    def tangent_unit_cart(self, s):
        """Get the normalized cartesian tangent vector at parameter value s.
        return np.array([x, y])"""
        raise NotImplementedError('')

class UnitCircleBoundary(BilliardBoundary_abstract):
    """Circular boundary of unit radius parameterized by angle."""
    
    def coords_cart(self, s):
        return np.array([np.cos(s), np.sin(s)])

    def linear_intersect(self, start, direction):
        
