from MITRotor import Geometry
import numpy as np
from pytest import approx


def test_geometry_init():
    Geometry.BEMGeometry(10, 20)


def test_geometry_shape():
    geom = Geometry.BEMGeometry(10, 20)
    assert geom.shape == (10, 20)


def test_delta():
    geom = Geometry.BEMGeometry(10, 20)
    dmu = np.diff(geom.mu)[0]
    dtheta = np.diff(geom.theta)[0]

    assert geom.dmu == dmu
    assert geom.dtheta == dtheta


def test_cartesian():
    geom = Geometry.BEMGeometry(10, 20)

    X, Y, Z = geom.cartesian(0, 0)

    assert X.shape == geom.shape
    assert Y.shape == geom.shape
    assert Z.shape == geom.shape
    assert all(x == 0 for x in X.ravel())


def test_annulus_average():
    geom = Geometry.BEMGeometry(10, 20)
    X = np.ones(geom.shape)

    ave = geom.annulus_average(X)

    assert len(ave) == geom.Nr
    assert all(x == approx(1) for x in ave)


def test_rotor_average():
    geom = Geometry.BEMGeometry(10, 20)
    X = np.ones(geom.shape)

    ave = geom.annulus_average(X)

    assert len(ave) == geom.Nr
    assert all(x == approx(1) for x in ave)
