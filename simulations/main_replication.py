import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    (
        "general_error_decomposition_plots.py",
        "Replicating the decompositions for two different signals from Figure 1 (a) and (b)",
    ),
    (
        "visualise_error_decomposition.py",
        "Replicating the weak and strong error decompositions from Figure 2 (a) and (b)",
    ),
    ("signals.py", "Replicating the signals from Figure 2 (c)"),
    (
        "TruncatedSVD_Replication.py",
        "Replicating the relative efficiencies from Figure 2 (d) [Truncated SVD]",
    ),
    (
        "Landweber_Replication.py",
        "Replicating the relative efficiencies from Figure 3 (a) and (b) [Landweber]",
    ),
    (
        "ConjugateGradients_Replication.py",
        "Replicating the relative efficiencies from Figure 4 (a) and (b) [Conjugate gradients]",
    ),
    ("L2Boost_signals.py", "Replicating the signals from Figure 5 (a) and (b) [L2Boost - signals]"),
    (
        "L2Boost_Replication.py",
        "Replicating the relative efficiencies from Figure 6 [L2Boost - Replication]",
    ),
    ("signal_estimation_comparison.py", "Replicating the signal estimation from Figure 9 (a) and (b)"),
    ("phillips_data.py", "Replicating the signal estimation from Figure 10 (a) and (b)"),
    ("ComparisonStudy.py", "Replicating the stopping times and errors from Figure 11 (a) and (b)"),
    ("timing_es.py", "Replicating the stopping times and errors from Table 1"),
    (
        "Simulation_counterexample_landweber.py",
        "Replicating the error decomposition from Figure 12 (b) and (d) [Landweber]",
    ),
    (
        "Simulation_counterexample_tSVD.py",
        "Replicating the error decomposition from Figure 12 (a) and (c) [Truncated SVD]",
    ),
]


def info(message, color="blue"):
    colors = {
        "green": "\033[92m",
        "red": "\033[31m",
        "blue": "\033[94m",
    }
    print(f"{colors.get(color, '')}{message}\033[0m")


def parse_args():
    parser = argparse.ArgumentParser(description="Run EarlyStopping replication scripts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scripts that would run without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for script_number, (script_name, description) in enumerate(SCRIPTS, start=1):
        script_path = SCRIPT_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing replication script: {script_path}")

        info(f"Script number {script_number}: {description}")
        command = [sys.executable, str(script_path)]
        if args.dry_run:
            print(" ".join(command))
            continue

        subprocess.run(command, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
