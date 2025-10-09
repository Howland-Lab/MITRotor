from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .BEMSolver import AerodynamicProperties

__all__ = ["TipLossModel", "NoTipLoss", "PrandtlTipLoss"]


class TipLossModel(ABC):
    @abstractmethod
    def __call__(
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ) -> ArrayLike:
        ...


class NoTipLoss(TipLossModel):
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        return np.ones_like(geometry.mu_mesh)


class PrandtlTipLoss(TipLossModel):
    def __init__(self, root_loss: bool = True):
        self.root_loss = root_loss

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        f_tip = (
            rotor.N_blades / 2 * (1 - geometry.mu_mesh) / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
        )
        F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -0.9999, 0.9999))

        if self.root_loss:
            f_hub = (
                rotor.N_blades
                / 2
                * (geometry.mu_mesh - R_hub)
                / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
            )
            F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -0.9999, 0.9999))

            return F_hub * F_tip

        else:
            return F_tip
        
class PrandtlTipLoss2(TipLossModel):
    def __init__(self, root_loss: bool = True):
        self.root_loss = root_loss

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        f_tip = (
             rotor.N_blades / 2 * (1 - geometry.mu_mesh)  * np.sqrt(1 + tsr**2)
        )
        F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -0.9999, 0.9999))

        if self.root_loss:
            f_hub = (
                rotor.N_blades
                / 2
                * (geometry.mu_mesh - R_hub)
                / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
            )
            F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -0.9999, 0.9999))

            return F_hub * F_tip

        else:
            return F_tip
        

class ShenTipLoss(TipLossModel):
    def __init__(self, root_loss: bool = True):
        self.root_loss = root_loss

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        f_tip = (
            rotor.N_blades / 2 * (1 - geometry.mu_mesh) / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
        )
        g = np.exp(-0.125 * (rotor.N_blades * tsr - 21)) + 0.1

        F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(g*f_tip, -100, 100)), -0.9999, 0.9999))

        if self.root_loss:
            f_hub = (
                rotor.N_blades
                / 2
                * (geometry.mu_mesh - R_hub)
                / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
            )
            F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -0.9999, 0.9999))

            return F_hub * F_tip

        else:
            return F_tip
        

### Goldstein
def f_goldstein_factor(l_bar, B, N):
    """
    Computes Goldstein factor G
    l_bar: dimensionless torsional parameter h/(2 pi R)
    B: number of blades
    N: number of radial positions
    """
    R = 1  # Using dimensionless quantities
    w = 1
    vr = np.linspace(0, 1, N)  # Radial positions
    G = f_goldstein_circulation(l_bar * R, w, R, B, vr) * B / (2 * np.pi * l_bar * R * w)
    return G, vr

def f_goldstein_circulation(l_bar, w, R, B, vr):
    """
    Computes Goldstein circulation using superposition of helix
    """
    vr = vr[1:]  # skip point 0
    N = len(vr)
    
    # Control Points (CPs) in between vortices
    vrCP = ((3 / (2 * N) + np.arange(N) / N)) * R
    
    # Influence matrix A
    A = np.zeros((N, N))
    Gamma_h = 1  # Unitary circulation
    psi_h = 0    # Azimuthal position

    for j in range(N):  # loop on vortex radii
        for i in range(N):  # loop on control point radii
            A[i, j] = f_ui_helix_n_theory(Gamma_h, vrCP[i], vr[j], l_bar, psi_h, B)

    # Boundary conditions
    U = np.zeros(N)
    U[:-1] = w / (1 + (l_bar ** 2) / ((vrCP[:-1] / R) ** 2))
    A[-1, :] = 1  # Condition: sum of gamma = 0
    U[-1] = 0

    # Solve system
    Gamma_t = np.linalg.solve(A, U)
    GammaGoldstein = np.cumsum(Gamma_t)
    GammaGoldstein = np.insert(GammaGoldstein, 0, 0)  # insert initial zero
    return GammaGoldstein

def f_ui_helix_n_theory(Gamma, r, r0, l, psih, B):
    """
    Computes induction from N-helices
    """
    C0z = (l ** 2 + r0 ** 2) ** 0.25 / (l ** 2 + r ** 2) ** 0.25
    C1z = l / 24 * ((3 * r ** 2 - 2 * l ** 2) / (l ** 2 + r ** 2) ** 1.5 +
                    (2 * l ** 2 + 9 * r0 ** 2) / (l ** 2 + r0 ** 2) ** 1.5)

    exp_term_r = np.exp(np.sqrt(l ** 2 + r ** 2) / l)
    exp_term_r0 = np.exp(np.sqrt(l ** 2 + r0 ** 2) / l)
    pexi = (r / r0) * (l + np.sqrt(l ** 2 + r0 ** 2)) / (l + np.sqrt(l ** 2 + r ** 2)) * exp_term_r / exp_term_r0
    mexi = 1 / pexi
    t = psih

    if abs(r) < r0:
        tmp = 1 / ((mexi * np.exp(-1j * t)) ** B - 1)
        vz = 1 / (2 * np.pi * l) + (1 / (2 * np.pi * l)) * C0z * np.real(tmp + (C1z / B) * np.log(1 + tmp))
    elif abs(r) > r0:
        tmp = 1 / ((pexi * np.exp(-1j * t)) ** B - 1)
        vz = (1 / (2 * np.pi * l)) * C0z * np.real(-tmp + (C1z / B) * np.log(1 + tmp))
    else:
        vz = 0

    uz = -B * Gamma * vz
    return uz


class GoldsteinTipLoss(TipLossModel):
    def __init__(self):
        pass

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        l_bar = 1 / (tsr)  # dimensionless torsional parameter h/(2 pi R)
        B = rotor.N_blades
        N = len(geometry.mu)  # number of radial positions
        G, vr = f_goldstein_factor(l_bar, B, N)
        lambda_r = tsr * vr
        F_go = (1 + lambda_r**2) / lambda_r**2 * G

        # Interpolate G to mu_mesh
        # G_interp = np.interp(geometry.mu_mesh, vr, G)
        F_interp = np.interp(geometry.mu_mesh, vr, F_go)

        # return G_interp
        return np.clip(F_interp, 0, 1)
