from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np

__all__ = ["BEMGeometry"]


class BEMGeometry:
    def __init__(self, Nr, Ntheta):
        self.Nr = Nr
        self.Ntheta = Ntheta

        self.mu = np.linspace(0.0, 0.9999, Nr)
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
    
# ---------- Function for averaging over the rotor ---------- #

    def annulus_average(self, X):
        X = np.asarray(X)
        theta = np.asarray(self.theta).reshape(-1)
        #Ensure last axis is Nθ
        if X.shape[-1] != theta.shape[0]:
            raise ValueError(f"Mismatch: X.shape={X.shape}, theta.shape={theta.shape}")
        # Integrate over Nθ (last axis)
        return (1 / (2 * np.pi)) * np.trapezoid(X, theta, axis=-1)

    def rotor_average(self, X):
        X = np.asarray(X)
        mu = np.asarray(self.mu).reshape(-1)  # (Nr,)
        # Ensure last axis is Nr
        if X.shape[-1] != mu.shape[0]:
            raise ValueError(f"Mismatch: X.shape={X.shape}, mu.shape={mu.shape}")
        # Integrate over Nr (last axis)
        return 2 * np.trapezoid(X * mu, mu, axis=-1)
    
# ---------- Function for adjusting axes for vectorization ---------- #

def expand_to_Np(x):  # add setpoint axis
    x = np.atleast_1d(x)
    if x.ndim < 3:
        return x[None, ...]
    else:
        return x

def expand_to_Nr_Ntheta(x):  # add Nr and Nθ axis
    x = np.atleast_1d(x)
    if x.ndim < 3:
        return x[:, None, None]
    else:
        return x
    
def expand_to_Nr(x): # add Nr axis
    x = np.atleast_1d(x)
    if x.ndim < 2:
        return x[:, None]
    else:
        return x
    
def expand_to_Ntheta(x): # add Nθ axis
    x = np.atleast_1d(x)
    if x.ndim < 3:
        return x[:, :, None]
    else:
        return x
