import numpy as np
import polars as pl
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
from MITRotor.Momentum import UnifiedMomentumLUT, UnifiedMomentum
from MITRotor import BEM, IEA15MW, BEMGeometry
import time


# use the same (easily convering example and up the number of setpoints and see how it scales)
# try a range of examples and see how slowly they converge