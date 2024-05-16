def test_geomoetry_imports():
    from MITRotor import BEMGeometry

def test_aerodynamic_model_imports():
    from MITRotor import AerodynamicModel, DefaultAerodynamics, KraghAerodynamics, AerodynamicProperties


def test_tangential_induction_model_imports():
    from MITRotor import TangentialInductionModel, DefaultTangentialInduction, NoTangentialInduction


def test_tip_loss_model_imports():
    from MITRotor import TipLossModel, NoTipLoss, PrandtlTipLoss


def test_momentum_model_imports():
    from MITRotor import (
        MomentumModel,
        ConstantInduction,
        ClassicalMomentum,
        HeckMomentum,
        UnifiedMomentum,
        MadsenMomentum,
    )


def test_bem_imports():
    from MITRotor import BEM, BEMSolution

def test_reference_turbine_imports():
    from MITRotor import IEA15MW, IEA10MW, IEA3_4MW
