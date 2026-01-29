from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np

__all__ = ["BEMGeometry"]


class BEMGeometry:
    def __init__(self, Nr, Ntheta):
        self.Nr = Nr
        self.Ntheta = Ntheta

        self.mu = np.linspace(0.0, 0.99999, Nr)
        self.theta = np.linspace(0.0, 2 * np.pi, Ntheta)

        self.theta_mesh, self.mu_mesh = np.meshgrid(self.theta, self.mu)

    @property
    def shape(self):
        return self.Nr, self.Ntheta

    @property
    def dmu(self):
        return self.mu[1] - self.mu[0]

    @property
    def dtheta(self):
        return self.theta[1] - self.theta[0]

    def cartesian(self, yaw: float, tilt: float) -> Tuple[ArrayLike, ...]:
        """
        Returns the grid point locations in cartesian coordinates
        nondimensionialized by rotor radius. Origin is located at hub center.

        Note: effect of yaw and tilt angles on grid points is not yet implemented.
        """
        # Probable sign error here.
        X = np.zeros_like(self.mu_mesh)
        Y = self.mu_mesh * np.sin(self.theta_mesh)  # lateral
        Z = self.mu_mesh * np.cos(self.theta_mesh)  # vertical

        return X, Y, Z

    def annulus_average(self, X: ArrayLike):
        X_azim = 1 / (2 * np.pi) * np.trapezoid(X, self.theta_mesh, axis=-1)

        return X_azim

    def rotor_average(self, X: ArrayLike):
        # Takes annulus average quantities and performs rotor average

        X_rotor = 2 * np.trapezoid(X * self.mu, self.mu)
        return X_rotor
