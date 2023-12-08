import numpy as np
from scipy import interpolate


class Airfoil:
    @classmethod
    def from_windio_airfoil(cls, airfoil: dict, R: float):
        assert len(airfoil["polars"]) == 1

        grid = np.array(airfoil["polars"][0]["c_l"]["grid"])
        return cls(
            airfoil["name"],
            grid,
            airfoil["polars"][0]["c_l"]["values"],
            airfoil["polars"][0]["c_d"]["values"],
        )

    def __init__(self, name, grid, cl, cd):
        self.name = name
        self.Cl_interp = interpolate.interp1d(grid, cl, fill_value="extrapolate")
        self.Cd_interp = interpolate.interp1d(grid, cd, fill_value="extrapolate")

    def __repr__(self):
        return f"Airfoil: {self.name}"

    def Cl(self, angle):
        return self.Cl_interp(angle)

    def Cd(self, angle):
        return self.Cd_interp(angle)


class BladeAirfoils:
    @classmethod
    def from_windio(cls, windio: dict, hub_radius, R, N=120):
        blade = windio["components"]["blade"]
        airfoils = windio["airfoils"]
        D = windio["assembly"]["rotor_diameter"]

        airfoil_grid = np.array(blade["outer_shape_bem"]["airfoil_position"]["grid"])
        airfoil_grid_adjusted = (hub_radius + airfoil_grid * (R - hub_radius)) / R
        airfoil_order = blade["outer_shape_bem"]["airfoil_position"]["labels"]

        airfoils = {x["name"]: Airfoil.from_windio_airfoil(x, R) for x in windio["airfoils"]}

        return cls(D, airfoil_grid_adjusted, airfoil_order, airfoils, N=N)

    def __init__(self, D, airfoil_grid, airfoil_order, airfoils, N=120):
        self.D = D

        aoa_grid = np.linspace(-np.pi, np.pi, N)
        cl = np.array([airfoils[name].Cl(aoa_grid) for name in airfoil_order])
        cd = np.array([airfoils[name].Cd(aoa_grid) for name in airfoil_order])

        self.cl_interp = interpolate.RegularGridInterpolator((airfoil_grid, aoa_grid), cl, bounds_error=False, fill_value=None)
        self.cd_interp = interpolate.RegularGridInterpolator((airfoil_grid, aoa_grid), cd, bounds_error=False, fill_value=None)

    def Cl(self, x, inflow):
        return self.cl_interp((x, inflow))

    def Cd(self, x, inflow):
        return self.cd_interp((x, inflow))

    def __call__(self, x, inflow):
        return self.Cl(x, inflow), self.Cd(x, inflow)


class RotorDefinition:
    @classmethod
    def from_windio(cls, windio: dict):
        name = windio["name"]

        P_rated = windio["assembly"]["rated_power"]
        hub_height = windio["assembly"]["hub_height"]
        rotorspeed_max = windio["control"]["torque"]["VS_maxspd"]
        tsr_target = windio["control"]["torque"]["tsr"]

        hub_diameter = windio["components"]["hub"]["diameter"]
        hub_radius = hub_diameter / 2
        cone = windio["components"]["hub"]["cone_angle"]  # deg
        blade = windio["components"]["blade"]

        blade_length = blade["outer_shape_bem"]["reference_axis"]["z"]["values"][-1]

        N_blades = windio["assembly"]["number_of_blades"]
        D = windio["assembly"]["rotor_diameter"]

        # Rotor radius adjusted to include cone angle and hub diameter
        R = blade_length * np.cos(np.deg2rad(cone)) + hub_radius

        data_twist = blade["outer_shape_bem"]["twist"]
        data_chord = blade["outer_shape_bem"]["chord"]

        # grid including hub center and cone angle
        twist_grid = (hub_radius + np.array(data_twist["grid"]) * (R - hub_radius)) / R
        twist_func = interpolate.interp1d(twist_grid, data_twist["values"], fill_value="extrapolate")
        # grid including hub center and cone angle
        chord_grid = (hub_radius + np.array(data_chord["grid"]) * (R - hub_radius)) / R
        chord_func = interpolate.interp1d(chord_grid, data_chord["values"], fill_value="extrapolate")

        solidity_func = lambda mu: np.minimum(
            N_blades * chord_func(mu) / (2 * np.pi * np.maximum(mu, 0.0001) * R),
            1,
        )

        airfoil_func = BladeAirfoils.from_windio(windio, hub_radius, R)

        return cls(
            twist_func,
            solidity_func,
            airfoil_func,
            N_blades,
            R,
            P_rated,
            rotorspeed_max,
            hub_height,
            tsr_target,
            hub_radius,
            name=name,
        )

    def __init__(
        self,
        twist_func,
        solidity_func,
        airfoil_func,
        N_blades,
        R,
        P_rated,
        rotorspeed_max,
        hub_height,
        tsr_target,
        hub_radius,
        name=None,
    ):
        self.name = name
        self.N_blades = N_blades

        self.R = R
        self.P_rated = P_rated
        self.rotorspeed_max = rotorspeed_max
        self.hub_height = hub_height
        self.tsr_target = tsr_target
        self.hub_radius = hub_radius

        self.twist_func = twist_func
        self.solidity_func = solidity_func
        self.airfoil_func = airfoil_func

    def twist(self, mu):
        return self.twist_func(mu)

    def solidity(self, mu):
        return self.solidity_func(mu)

    def clcd(self, mu, aoa):
        return self.airfoil_func(mu, aoa)
