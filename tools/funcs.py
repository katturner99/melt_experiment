import re
import argparse
from pathlib import Path
import xarray as xr


def get_available_months(base_dir):
    """
    Return a sorted list of YYYYMM directories that contain a MITgcm `output.nc` file.

    This scans `<base_dir>/output/` and identifies all directories matching the
    six-digit YYYYMM pattern (e.g., ``199201``). A directory is included only if
    it contains ``MITgcm/output.nc`` inside it.

    Parameters
    ----------
    base_dir : str or Path
        Path to the simulation root directory containing an ``output/`` folder.

    Returns
    -------
    list of str
        Sorted list of valid YYYYMM directory names that include a MITgcm output
        dataset.
    """
    base_dir = Path(base_dir)
    output_root = base_dir / "output"

    if not output_root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    months = []

    for entry in output_root.iterdir():
        if entry.is_dir() and re.fullmatch(r"\d{6}", entry.name):
            nc_file = entry / "MITgcm" / "output.nc"
            if nc_file.is_file():
                months.append(entry.name)

    return sorted(months)


def parse_args():
    """
    Parse command-line arguments for generating UaMITgcm time series.

    This argument parser supports selecting a scenario, ensemble member, and
    variable type for time-series extraction.

    Returns
    -------
    argparse.Namespace
        A namespace containing parsed arguments: ``scenario``,
        ``ens_member``, and ``variable``.

    Notes
    -----
    Valid variables are:
    ``temperature``, ``salt``, ``etan``, ``melt``, ``undercurrent``.

    Examples
    --------
    From the command line:

    >>> python script.py LENS 3 temperature
    """
    parser = argparse.ArgumentParser(
        description="Create time series for UaMITgcm coupled output."
    )
    parser.add_argument("scenario", help="Scenario (LENS, month, 1year, 5year)")
    parser.add_argument("ens_member", help="Ensemble member (1–9)")
    parser.add_argument(
        "variable", choices=["temperature", "salt", "etan", "melt", "undercurrent", "transport","si_freezing", "si_melting", "fw_total"], help="Variable to process"
    )
    return parser.parse_args()


def load_month_dataset(base_dir, month):
    """
    Load a MITgcm ``output.nc`` dataset for a specific YYYYMM month.

    Parameters
    ----------
    base_dir : str or Path
        Path to the simulation root directory containing an ``output/`` folder.

    month : str
        A six-character YYYYMM string identifying the output month.

    Returns
    -------
    xarray.Dataset
        The monthly MITgcm dataset loaded from ``output/YYYMM/MITgcm/output.nc``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the expected location.

    Examples
    --------
    >>> ds = load_month_dataset("/sim/AMUND_LENS001_O", "199201")
    >>> ds
    <xarray.Dataset> ...
    """
    path = Path(base_dir) / month / "MITgcm" / "output.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_dataset(path)
