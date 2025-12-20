"""
SPARC rotation curve data acquisition and parsing.

SPARC (Spitzer Photometry and Accurate Rotation Curves) provides:
- Rotation curves for 175 disk galaxies
- Decomposed baryonic components (stellar disk, gas, bulge)
- Galaxy properties (distance, inclination, luminosity, quality flag)

Sources:
- Zenodo archive: https://zenodo.org/records/16284118
- Official: http://astroweb.case.edu/SPARC/
- Paper: Lelli et al. (2016), AJ 152, 157

Data files:
- Rotmod_LTG.zip: Individual rotation curve files
- SPARC_Lelli2016c.mrt: Galaxy properties table
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Data paths
DATA_DIR = Path(__file__).parent
SPARC_DIR = DATA_DIR / "sparc_cache"
SPARC_ZIP = SPARC_DIR / "Rotmod_LTG.zip"

# URLs - Zenodo stable archive first (verified working Dec 2025)
ZENODO_BASE = "https://zenodo.org/records/16284118/files"
SPARC_URLS = [
    f"{ZENODO_BASE}/Rotmod_LTG.zip?download=1",
    "http://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
]
SPARC_TABLE_URLS = [
    f"{ZENODO_BASE}/SPARC_Lelli2016c.mrt?download=1",
    "http://astroweb.case.edu/SPARC/SPARC_Lelli2016c.mrt",
]


class SPARCDownloadError(Exception):
    """Raised when SPARC data cannot be downloaded from any source."""
    pass


@dataclass
class GalaxyData:
    """
    Container for a single galaxy's rotation curve data.

    All velocities are in km/s, radii in kpc.
    """
    name: str
    distance: float           # Mpc
    inclination: float        # degrees
    luminosity: float         # 10^9 L_sun (3.6 micron)
    eff_radius: float         # kpc (effective radius)
    disk_scale: float         # kpc (disk scale length)
    MHI: float                # 10^9 M_sun (HI mass)
    quality: int              # Quality flag (1=best, 2=good, 3=marginal)

    # Rotation curve arrays
    radius: np.ndarray        # kpc
    v_obs: np.ndarray         # km/s (observed rotation velocity)
    v_err: np.ndarray         # km/s (error)
    v_gas: np.ndarray         # km/s (gas contribution)
    v_disk: np.ndarray        # km/s (stellar disk contribution at M/L=1)
    v_bulge: np.ndarray       # km/s (bulge contribution at M/L=1)

    def v_baryon(self, ml_disk: float = 0.5, ml_bulge: float = 0.7) -> np.ndarray:
        """
        Total baryonic velocity contribution.

        V_bar^2 = V_gas^2 + (M/L_disk) * V_disk^2 + (M/L_bulge) * V_bulge^2
        """
        return np.sqrt(
            self.v_gas**2 +
            ml_disk * self.v_disk**2 +
            ml_bulge * self.v_bulge**2
        )

    def v_dm_needed(self, ml_disk: float = 0.5, ml_bulge: float = 0.7) -> np.ndarray:
        """
        Dark matter velocity contribution needed to match observed.

        V_DM^2 = V_obs^2 - V_bar^2
        """
        v_bar = self.v_baryon(ml_disk, ml_bulge)
        v_dm_sq = self.v_obs**2 - v_bar**2
        return np.sqrt(np.maximum(v_dm_sq, 0))

    @property
    def n_points(self) -> int:
        """Number of rotation curve points."""
        return len(self.radius)

    @property
    def r_max(self) -> float:
        """Maximum radius in kpc."""
        return float(self.radius[-1]) if len(self.radius) > 0 else 0.0

    @property
    def v_max(self) -> float:
        """Maximum observed velocity in km/s."""
        return float(np.max(self.v_obs)) if len(self.v_obs) > 0 else 0.0

    def is_dwarf(self, v_max_cut: float = 80.0) -> bool:
        """Check if galaxy is a dwarf (v_max < cut)."""
        return self.v_max < v_max_cut

    def is_lsb(self, mu_0_cut: float = 22.0) -> bool:
        """
        Check if galaxy is low surface brightness.
        Uses central surface brightness proxy from luminosity and scale length.
        """
        if self.disk_scale > 0 and self.luminosity > 0:
            L_per_area = self.luminosity / (2 * np.pi * self.disk_scale**2)
            mu_0 = -2.5 * np.log10(L_per_area * 1e9) + 21.57
            return mu_0 > mu_0_cut
        return False


def download_sparc(force: bool = False, verbose: bool = True) -> bool:
    """
    Download SPARC data from Zenodo or official source.

    IMPORTANT: This function will FAIL LOUDLY if data cannot be downloaded.
    No synthetic fallback is provided.

    Parameters:
        force: Re-download even if files exist
        verbose: Print progress

    Returns:
        True if successful

    Raises:
        SPARCDownloadError: If download fails from all sources
    """
    SPARC_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded - check both possible locations
    rotmod_dir = SPARC_DIR / "Rotmod_LTG"
    if not rotmod_dir.exists():
        rotmod_dir = SPARC_DIR
    if not force and len(list(rotmod_dir.glob("*_rotmod.dat"))) > 100:
        if verbose:
            print(f"SPARC data already downloaded ({len(list(rotmod_dir.glob('*.dat')))} galaxies).")
        return True

    if verbose:
        print("Downloading SPARC rotation curve data...")

    # Try to download rotation curve files
    zip_success = False
    errors = []

    for url in SPARC_URLS:
        try:
            if verbose:
                print(f"  Trying: {url[:60]}...")

            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Verify it's a zip file
            if len(response.content) < 1000:
                errors.append(f"{url}: Response too small ({len(response.content)} bytes)")
                continue

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(SPARC_DIR)

            if verbose:
                print(f"    Success!")
            zip_success = True
            break

        except requests.exceptions.RequestException as e:
            errors.append(f"{url}: {e}")
            if verbose:
                print(f"    Failed: {e}")
        except zipfile.BadZipFile as e:
            errors.append(f"{url}: Bad zip file - {e}")
            if verbose:
                print(f"    Failed: Bad zip file")

    if not zip_success:
        error_msg = "Failed to download SPARC rotation curves from any source:\n"
        error_msg += "\n".join(f"  - {e}" for e in errors)
        error_msg += "\n\nPlease check your internet connection or manually download from:"
        error_msg += "\n  https://zenodo.org/records/16284118"
        raise SPARCDownloadError(error_msg)

    # Try to download properties table (non-fatal if fails)
    table_downloaded = False
    for url in SPARC_TABLE_URLS:
        try:
            if verbose:
                print(f"  Downloading properties table...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            (SPARC_DIR / "SPARC_Lelli2016c.mrt").write_bytes(response.content)
            table_downloaded = True
            break
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not download properties table: {e}")

    # Verify download - check both possible extraction locations
    rotmod_dir = SPARC_DIR / "Rotmod_LTG"
    if not rotmod_dir.exists():
        # Files may have extracted directly to SPARC_DIR
        if len(list(SPARC_DIR.glob("*_rotmod.dat"))) > 0:
            rotmod_dir = SPARC_DIR
        else:
            raise SPARCDownloadError(f"No rotation curve files found after extraction")

    n_files = len(list(rotmod_dir.glob("*.dat")))
    if n_files < 100:
        raise SPARCDownloadError(f"Only {n_files} galaxy files found (expected ~175)")

    if verbose:
        print(f"SPARC download complete: {n_files} galaxies")
        if table_downloaded:
            print(f"  Properties table: downloaded")
        else:
            print(f"  Properties table: not available (will parse from headers)")

    return True


def _parse_mrt_table(filepath: Path) -> pd.DataFrame:
    """
    Parse SPARC MRT (Machine-Readable Table) format.

    The SPARC_Lelli2016c.mrt file contains galaxy properties.
    Format: Fixed-width columns as described in the header.
    Data lines are space-indented and start after the header.

    Column byte positions (0-indexed, after stripping leading spaces):
        0-10: Galaxy name
        11-12: Hubble Type
        13-18: Distance (Mpc)
        19-23: e_D
        24-25: f_D (distance method)
        26-29: Inc (deg)
        30-33: e_Inc
        34-40: L[3.6] (10^9 Lsun)
        ...
        86-90: Vflat (km/s)
        96-98: Q (quality)
    """
    content = filepath.read_text()
    lines = content.split('\n')

    data = []
    for line in lines:
        # Data lines have galaxy names like CamB, DDO154, NGC1234, etc.
        # They are indented with spaces
        stripped = line.strip()
        if len(stripped) < 50:
            continue

        # Check if first word looks like a galaxy name
        first_word = stripped.split()[0] if stripped else ''
        if not any(first_word.startswith(prefix) for prefix in
                   ('CamB', 'D5', 'D6', 'DDO', 'ESO', 'F5', 'F6', 'F7', 'F8',
                    'IC', 'KK', 'NGC', 'PGC', 'UGC', 'UGCA')):
            continue

        try:
            # Split by whitespace - more robust than fixed positions
            parts = stripped.split()
            if len(parts) < 17:
                continue

            name = parts[0]
            row = {
                'name': name,
                'hubble_type': int(parts[1]) if parts[1].lstrip('-').isdigit() else 0,
                'distance': float(parts[2]),
                'inclination': float(parts[5]),
                'luminosity': float(parts[7]),
                'eff_radius': float(parts[9]),
                'disk_scale': float(parts[11]),
                'MHI': float(parts[13]),
                'v_flat': float(parts[15]),
                'quality': int(parts[17]),
            }
            data.append(row)
        except (ValueError, IndexError):
            continue

    return pd.DataFrame(data)


def _parse_rotmod_file(filepath: Path, catalog_row: Optional[Dict] = None) -> Dict:
    """
    Parse a single SPARC rotation curve file.

    Format (Rotmod_LTG files):
        Column 1: Radius (kpc)
        Column 2: Vobs (km/s) - observed rotation velocity
        Column 3: errV (km/s) - error on Vobs
        Column 4: Vgas (km/s) - gas contribution
        Column 5: Vdisk (km/s) - stellar disk contribution (at M/L=1)
        Column 6: Vbul (km/s) - bulge contribution (at M/L=1)

    Properties are parsed from catalog if available, otherwise defaults used.
    """
    content = filepath.read_text()
    lines = content.strip().split('\n')

    # Default properties - clean the name
    props = {
        'name': filepath.stem.replace('_rotmod', ''),
        'distance': 10.0,
        'inclination': 60.0,
        'luminosity': 1.0,
        'eff_radius': 2.0,
        'disk_scale': 1.0,
        'MHI': 0.5,
        'quality': 2,
    }

    # Override with catalog data if available
    if catalog_row is not None:
        for key in ['distance', 'inclination', 'luminosity', 'eff_radius',
                    'disk_scale', 'MHI', 'quality']:
            if key in catalog_row and not pd.isna(catalog_row.get(key)):
                props[key] = catalog_row[key]

    # Parse data lines
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'):
            continue

        parts = line.split()
        if len(parts) >= 4:
            try:
                vals = [float(p) for p in parts[:6]]
                # Pad with zeros if fewer than 6 columns
                while len(vals) < 6:
                    vals.append(0.0)
                data_lines.append(vals)
            except ValueError:
                continue

    if not data_lines:
        raise ValueError(f"No valid data found in {filepath}")

    data = np.array(data_lines)

    props['radius'] = data[:, 0]
    props['v_obs'] = np.abs(data[:, 1])  # Ensure positive
    props['v_err'] = np.maximum(data[:, 2], 1.0)  # Minimum 1 km/s error
    props['v_gas'] = np.abs(data[:, 3])
    props['v_disk'] = np.abs(data[:, 4]) if data.shape[1] > 4 else np.zeros_like(data[:, 0])
    props['v_bulge'] = np.abs(data[:, 5]) if data.shape[1] > 5 else np.zeros_like(data[:, 0])

    return props


def _get_rotmod_dir() -> Path:
    """Get the directory containing rotation curve files."""
    rotmod_dir = SPARC_DIR / "Rotmod_LTG"
    if rotmod_dir.exists() and len(list(rotmod_dir.glob("*.dat"))) > 0:
        return rotmod_dir
    # Files may be directly in SPARC_DIR
    if len(list(SPARC_DIR.glob("*_rotmod.dat"))) > 0:
        return SPARC_DIR
    raise SPARCDownloadError("No rotation curve files found. Run download_sparc() first.")


def load_sparc_catalog(verbose: bool = True) -> pd.DataFrame:
    """
    Load SPARC galaxy catalog with properties.

    Raises SPARCDownloadError if data not available.
    """
    download_sparc(verbose=verbose)

    mrt_path = SPARC_DIR / "SPARC_Lelli2016c.mrt"

    if mrt_path.exists():
        try:
            df = _parse_mrt_table(mrt_path)
            if verbose and len(df) > 0:
                print(f"Loaded {len(df)} galaxies from catalog.")
            return df
        except Exception as e:
            if verbose:
                print(f"Warning: Could not parse MRT table: {e}")

    # Fallback: scan rotmod directory for galaxy files
    rotmod_dir = _get_rotmod_dir()

    galaxies = []
    for f in sorted(rotmod_dir.glob("*.dat")):
        galaxies.append({'name': f.stem})

    df = pd.DataFrame(galaxies)
    if verbose:
        print(f"Found {len(df)} galaxy files (no catalog metadata).")
    return df


def load_rotation_curve(galaxy_name: str,
                       catalog: Optional[pd.DataFrame] = None,
                       verbose: bool = False) -> GalaxyData:
    """
    Load rotation curve data for a specific galaxy.

    Parameters:
        galaxy_name: Name of the galaxy (e.g., 'DDO154')
        catalog: Pre-loaded catalog DataFrame (optional)
        verbose: Print status

    Returns:
        GalaxyData object

    Raises:
        FileNotFoundError: If galaxy not found
        SPARCDownloadError: If SPARC data not downloaded
    """
    download_sparc(verbose=verbose)

    rotmod_dir = _get_rotmod_dir()

    # Find the file
    filepath = rotmod_dir / f"{galaxy_name}_rotmod.dat"
    if not filepath.exists():
        filepath = rotmod_dir / f"{galaxy_name}.dat"
    if not filepath.exists():
        # Case-insensitive search
        for f in rotmod_dir.glob("*.dat"):
            if f.stem.lower().replace('_rotmod', '') == galaxy_name.lower():
                filepath = f
                break
        else:
            raise FileNotFoundError(f"Galaxy {galaxy_name} not found in SPARC data")

    # Get catalog row if available
    catalog_row = None
    if catalog is not None and 'name' in catalog.columns:
        match = catalog[catalog['name'].str.lower() == galaxy_name.lower()]
        if len(match) > 0:
            catalog_row = match.iloc[0].to_dict()

    props = _parse_rotmod_file(filepath, catalog_row)
    return GalaxyData(**props)


def load_all_galaxies(verbose: bool = True) -> List[GalaxyData]:
    """
    Load all available SPARC galaxies.

    Returns:
        List of GalaxyData objects

    Raises:
        SPARCDownloadError: If SPARC data not available
    """
    download_sparc(verbose=verbose)

    catalog = None
    try:
        catalog = load_sparc_catalog(verbose=False)
    except Exception:
        pass

    rotmod_dir = _get_rotmod_dir()
    galaxies = []

    for filepath in sorted(rotmod_dir.glob("*.dat")):
        name = filepath.stem.replace('_rotmod', '')
        try:
            gal = load_rotation_curve(name, catalog=catalog, verbose=False)
            galaxies.append(gal)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {name}: {e}")

    if verbose:
        print(f"Loaded {len(galaxies)} galaxies from SPARC")

    return galaxies


def get_dwarf_lsb_sample(v_max_cut: float = 80.0,
                         min_points: int = 8,
                         max_quality: int = 2,
                         verbose: bool = True) -> Tuple[List[GalaxyData], pd.DataFrame]:
    """
    Get dwarf/LSB galaxies suitable for Msoup testing.

    Selection criteria (as per verdict.md specifications):
        - v_max < v_max_cut (80 km/s default)
        - At least min_points data points (8 default)
        - quality <= max_quality (2 default, where 1=best, 3=marginal)

    Parameters:
        v_max_cut: Maximum observed rotation velocity for dwarf classification
        min_points: Minimum number of rotation curve data points
        max_quality: Maximum quality flag (1=best, 2=good, 3=marginal)
        verbose: Print progress

    Returns:
        Tuple of (list of GalaxyData, DataFrame with selection summary)

    Raises:
        SPARCDownloadError: If SPARC data not available
    """
    all_galaxies = load_all_galaxies(verbose=verbose)

    selection_records = []
    selected_galaxies = []

    for gal in all_galaxies:
        record = {
            'name': gal.name,
            'v_max': gal.v_max,
            'n_points': gal.n_points,
            'quality': gal.quality,
            'distance': gal.distance,
            'luminosity': gal.luminosity,
            'selected': False,
            'reason': ''
        }

        # Apply selection criteria
        if gal.v_max >= v_max_cut:
            record['reason'] = f'v_max={gal.v_max:.1f} >= {v_max_cut}'
        elif gal.n_points < min_points:
            record['reason'] = f'n_points={gal.n_points} < {min_points}'
        elif gal.quality > max_quality:
            record['reason'] = f'quality={gal.quality} > {max_quality}'
        else:
            record['selected'] = True
            record['reason'] = 'passed'
            selected_galaxies.append(gal)

        selection_records.append(record)

    selection_df = pd.DataFrame(selection_records)

    if verbose:
        n_total = len(all_galaxies)
        n_selected = len(selected_galaxies)
        print(f"\nDwarf/LSB Selection Summary:")
        print(f"  Criteria: v_max < {v_max_cut} km/s, n_points >= {min_points}, quality <= {max_quality}")
        print(f"  Total galaxies: {n_total}")
        print(f"  Selected: {n_selected}")
        print(f"  Rejected by v_max: {len(selection_df[selection_df['reason'].str.contains('v_max')])}")
        print(f"  Rejected by n_points: {len(selection_df[selection_df['reason'].str.contains('n_points')])}")
        print(f"  Rejected by quality: {len(selection_df[selection_df['reason'].str.contains('quality')])}")

    return selected_galaxies, selection_df


# NO SYNTHETIC FALLBACK - the validator must use real data or fail
def create_synthetic_sparc_data(*args, **kwargs):
    """
    DEPRECATED: Synthetic data generation is disabled.

    The validator must use real SPARC data. If downloads fail, the analysis
    should be marked as incomplete rather than using synthetic data.

    Raises:
        NotImplementedError: Always, to prevent accidental use
    """
    raise NotImplementedError(
        "Synthetic SPARC data is disabled. "
        "The validator requires real SPARC data from Zenodo or the official source. "
        "If download fails, check your internet connection or manually download from: "
        "https://zenodo.org/records/16284118"
    )
