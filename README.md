# MITRotor
This repository is used for setting up and solving customized blade element models (BEMs). These models can either be solved on their own, or used within the `MITWindfarm` package to explore various wind turbine rotor and wake dynamics. `MITRotor` allows the user to specify rotor definition, geometry, aerodynamic properties, tip loss method, axial induction calculation method, and tangential induction calculation method.


# Installation
To install this Python package follow one of the following methods.

### Direct installation from Github
To install directly from Github into the current Python environment, run:
```bash
pip install git+https://github.com/Howland-Lab/MITRotor.git
```


### Install from cloned repository
If you prefer to download the repository first (for example, to run the example and paper figure scripts), you can first clone the repository, either using http:
```bash
git clone https://github.com/Howland-Lab/MITRotor.git
```
or ssh:
```bash
git clone git@github.com:Howland-Lab/MITRotor.git
```
then, install locally using pip using `pip install .` for the base installation, or `pip install .[examples]` to install the extra dependencies required to run the examples scripts scripts. To include developer packages in the installation, use `pip install -e .[dev]`. 

```bash
cd MITRotor
pip install .
```