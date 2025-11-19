import sys
from pathlib import Path

"""directories used often"""

# paths
OUTPUT_PATH = "/data/oceans_output/shelf/katner33/melt_experiments/"


def get_filepath(scenario, ens_member):
    """Build the correct file path for the given scenario and ensemble."""
    scenario_map = {
        "LENS": f"old_PAS/PAS_LENS00{ens_member}_O/",
        "MELT_noS": f"PAS_MELT00{ens_member}_noS/",
    }
    if scenario not in scenario_map:
        sys.exit(f"Invalid scenario: {scenario}")

    path = Path(OUTPUT_PATH) / scenario_map[scenario]
    if not path.exists():
        sys.exit(f"Stopped - Could not find directory {path}")

    return path