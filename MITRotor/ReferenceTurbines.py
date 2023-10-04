from pathlib import Path
import yaml
from .RotorDefinition import RotorDefinition

fn_IEA15MW = Path(__file__).parent / "IEA-15-240-RWT.yaml"
fn_IEA10MW = Path(__file__).parent / "IEA-10-198-RWT.yaml"
fn_IEA3_4MW = Path(__file__).parent / "IEA-3.4-130-RWT.yaml"


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
