from pathlib import Path
import yaml
from .RotorDefinition import RotorDefinition

turbine_model_dir = Path(__file__).parent / "ReferenceTurbines"
fn_IEA15MW = turbine_model_dir / "IEA-15-240-RWT.yaml"
fn_IEA10MW = turbine_model_dir / "IEA-10-198-RWT.yaml"
fn_IEA3_4MW = turbine_model_dir / "IEA-3.4-130-RWT.yaml"
fn_NREL_5MW = turbine_model_dir / "NREL-5-126-RWT.yaml"


__all__ = ["IEA15MW", "IEA10MW", "IEA3_4MW", "NREL5MW"]


def IEA15MW() -> RotorDefinition:
    with open(fn_IEA15MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)


def IEA10MW() -> RotorDefinition:
    with open(fn_IEA10MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)


def IEA3_4MW() -> RotorDefinition:
    with open(fn_IEA3_4MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)

def NREL5MW() -> RotorDefinition:
    with open(fn_NREL_5MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)
