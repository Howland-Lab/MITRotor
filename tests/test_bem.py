from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Geometry import BEMGeometry


def test_IEA15MW():
    rotor = IEA15MW()


def test_BEM_initialise():
    rotor = IEA15MW()
    geometry = BEMGeometry(10, 20)
    bem = BEM(rotor=rotor, geometry=geometry)


def test_BEM_initial_guess():
    rotor = IEA15MW()
    geometry = BEMGeometry(10, 20)
    bem = BEM(rotor=rotor, geometry=geometry)
    x0 = bem.initial_guess(0.0, 7.0)


def test_BEM_residual():
    rotor = IEA15MW()
    geometry = BEMGeometry(10, 20)
    bem = BEM(rotor=rotor, geometry=geometry)
    x0 = bem.initial_guess(0.0, 7.0)
    bem.residual(x0, 0.0, 7, 0)


def test_BEM_solve():
    rotor = IEA15MW()
    geometry = BEMGeometry(10, 20)
    bem = BEM(rotor=rotor, geometry=geometry)
    bem(0.0, 7.0, 0.0)
