import abc

from flax import struct
import jax.numpy as jnp

from hj_reachability import utils

from typing import Any

Array = Any


@struct.dataclass
class BoundedSet(metaclass=abc.ABCMeta):
    """Abstract base class for representing bounded subsets of Euclidean space."""

    @abc.abstractmethod
    def extreme_point(self, direction: Array) -> Array:
        """Computes the point `x` in the set such that the dot product `x @ direction` is greatest."""

    @property
    @abc.abstractmethod
    def bounding_box(self) -> "Box":
        """Returns an axis-aligned bounding box for the set."""

    @property
    def max_magnitudes(self) -> Array:
        """Returns the maximum magnitude (per dimension) of points in the set."""
        return jnp.maximum(jnp.abs(self.bounding_box.lo), jnp.abs(self.bounding_box.hi))

    @property
    def ndim(self) -> int:
        """Returns the dimension of the Euclidean space the set lies within."""
        return self.bounding_box.ndim


@struct.dataclass
class Box(BoundedSet):
    """Class for representing axis-aligned boxes."""
    lo: Array
    hi: Array

    def extreme_point(self, direction: Array) -> Array:
        """Computes the point `x` in the set such that the dot product `x @ direction` is greatest."""
        return jnp.where(direction < 0, self.lo, self.hi)

    @property
    def bounding_box(self) -> "Box":
        """Returns an axis-aligned bounding box for the set."""
        return self

    @property
    def ndim(self) -> int:
        """Returns the dimension of the Euclidean space the set lies within."""
        return self.lo.shape[-1]


@struct.dataclass
class Ball(BoundedSet):
    """Class for representing Euclidean (L2) balls."""
    center: Array
    radius: Array

    def extreme_point(self, direction: Array) -> Array:
        """Computes the point `x` in the set such that the dot product `x @ direction` is greatest."""
        return self.center + self.radius * utils.unit_vector(direction)

    @property
    def bounding_box(self) -> "Box":
        """Returns an axis-aligned bounding box for the set."""
        return Box(self.center - jnp.expand_dims(self.radius, -1), self.center + jnp.expand_dims(self.radius, -1))
