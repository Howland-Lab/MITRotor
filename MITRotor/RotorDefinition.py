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

        self.cl_interp = interpolate.RectBivariateSpline(airfoil_grid, aoa_grid, cl)
        self.cd_interp = interpolate.RectBivariateSpline(airfoil_grid, aoa_grid, cd)

    def Cl(self, x, inflow):
        return self.cl_interp(x, inflow, grid=False)

    def Cd(self, x, inflow):
        return self.cd_interp(x, inflow, grid=False)

    def __call__(self, x, inflow, tsr=None, apply_3D_stall_correction=False, c_on_r=None):
        a = 1
        b = 1
        d = 1
        Cl_2D = self.Cl(x, inflow)
        Cd_2D = self.Cd(x, inflow)

        if apply_3D_stall_correction:
            # Apply 3D stall correction (Du and Selig 1998)

            # Compute alpha_0 (angle of attack at which Cl=0)
            # aoa_line = np.radians(np.linspace(-10, 10, 100))
            # Clp_line = self.Cl(x, aoa_line)
            # alpha_0 = aoa_line[np.argmin(np.abs(Clp_line))]
            # Clp = 
            
            # Compute slope of Cl curve
            CL1 = self.Cl(x, np.radians(0) * np.ones_like(x))
            CL2 = self.Cl(x, np.radians(5) * np.ones_like(x))
            m = (CL2 - CL1) / np.radians(5 - 0)
            alpha_0 = -CL1 / m

            tsr_mod = 1 / np.sqrt(1 + tsr**2)

            exp1 = np.minimum((d/(tsr_mod * x + 1e-3)), 10)
            exp2 = np.minimum((d/(2*tsr_mod * x + 1e-3)), 10)

            fl = (
                (1 / (m)) * 
                ((1.6 * c_on_r / 0.1267) * 
                ((a - c_on_r**exp1) / 
                 (b + c_on_r**exp1)) - 1)
            )
            fd = (
                (1 / (m)) * 
                ((1.6 * c_on_r / 0.1267) * 
                ((a - c_on_r**exp2) / 
                 (b + c_on_r**exp2)) - 1)
            )
            # breakpoint()

            Clp = m * (inflow - alpha_0)

            Cd0 = self.Cd(x, alpha_0)

            del_Cl = np.maximum(0, fl * (Clp - Cl_2D))
            del_Cd = np.maximum(0, fd * (Cd_2D - Cd0))

            Cl_3D = Cl_2D + del_Cl
            Cd_3D = Cd_2D + del_Cd

            return Cl_3D, Cd_3D
        
        else:
            # No 3D stall correction
            return Cl_2D, Cd_2D


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

    def clcd(self, mu, aoa, tsr=None, apply_3D_stall_correction=False):
        solidity = self.solidity(mu)
        c_on_r = solidity * (2 * np.pi) / (self.N_blades)
        return self.airfoil_func(mu, aoa, tsr, apply_3D_stall_correction, c_on_r)
