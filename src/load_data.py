from pathlib import Path
import pandas as pd

# Finds the data file no matter where you run the code from
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "data.txt"

COLS = [
    "lever_position",
    "ship_speed_knots",
    "gas_turbine_shaft_torque_kNm",
    "gas_turbine_rpm",
    "gas_generator_rpm",
    "starboard_propeller_torque_kN",
    "port_propeller_torque_kN",
    "hp_turbine_exit_temp_C",
    "gt_compressor_inlet_air_temp_C",
    "gt_compressor_outlet_air_temp_C",
    "hp_turbine_exit_pressure_bar",
    "gt_compressor_inlet_air_pressure_bar",
    "gt_compressor_outlet_air_pressure_bar",
    "gas_turbine_exhaust_gas_pressure_bar",
    "turbine_injection_control_pct",
    "fuel_flow_kg_per_s",
    "gt_compressor_decay_state",
    "gt_turbine_decay_state",
]


def load_raw(path=RAW_PATH):
    """Read the raw data file and name the columns."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)