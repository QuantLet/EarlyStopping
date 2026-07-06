from pathlib import Path


SIMULATION_DIR = Path(__file__).resolve().parent


def output_path(filename):
    return SIMULATION_DIR / filename
