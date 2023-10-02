from abc import ABC, abstractmethod
import numpy as np


class TiplossModel(ABC):
    @abstractmethod
    def __call__(self, mu, phi):
        pass


def build_tiploss_model(tiploss_model, rotor):
    if isinstance(tiploss_model, TiplossModel):
        return tiploss_model

    elif isinstance(tiploss_model, str):
        if tiploss_model is None:
            return NoTiploss()
        elif tiploss_model == "Prandtl":
            return Prandtl(B=rotor.N_blades)
        elif tiploss_model == "PrandtlRootTip":
            return PrandtlRootTip(rotor.hub_radius / rotor.R, B=rotor.N_blades)
        else:
            raise ValueError(f"Unknown model name {tiploss_model}")

    else:
        raise TypeError("tiploss_model must be a string or TiplossModel instance")


class NoTiploss(TiplossModel):
    def __call__(mu, phi):
        return np.ones_like(mu)


class Prandtl(TiplossModel):
    def __init__(self, B=3):
        self.B = B

    def __call__(self, mu, phi, B=3):
        f = self.B / 2 * (1 - mu) / (mu * np.abs(np.sin(phi)))
        F = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f, -100, 100)), -1, 1))
        return np.maximum(F, 0.01)


class PrandtlRootTip(TiplossModel):
    """
    Returns the tip loss including root correction. R_hub should be
    nondimensional.
    """

    def __init__(self, R_hub, B=3):
        self.R_hub = R_hub
        self.B = B

    def __call__(self, mu, phi):
        f_tip = self.B / 2 * (1 - mu) / (np.maximum(mu, 0.0001) * np.abs(np.sin(phi)))
        F_tip = (
            2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -1, 1))
        )
        f_hub = (
            self.B
            / 2
            * (mu - self.R_hub)
            / (np.maximum(mu, 0.0001) * np.abs(np.sin(phi)))
        )
        F_hub = (
            2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -1, 1))
        )

        return np.maximum(F_hub * F_tip, 0.00001)
