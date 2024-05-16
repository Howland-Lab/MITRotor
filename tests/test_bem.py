from MITRotor import BEM, IEA15MW


def test_IEA15MW():
    IEA15MW()


def test_BEM_initialise():
    rotor = IEA15MW()
    BEM(rotor=rotor)


def test_default_models():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)


def test_BEM_initial_guess():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    bem.initial_guess(0.0, 7.0)


def test_BEM_residual():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    x0 = bem.initial_guess(0.0, 7.0)
    bem.residual(x0, 0.0, 7, 0)


def test_BEM_solve():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    sol = bem(0.0, 7.0, 0.0)

    # Check power coefficient is positive and less than Betz limit.
    assert (sol.Cp() > 0) and (sol.Cp() < 16 / 27)
