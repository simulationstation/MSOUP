"""
SPARC rotation curve data acquisition and parsing.

SPARC (Spitzer Photometry and Accurate Rotation Curves) provides:
- Rotation curves for 175 disk galaxies
- Decomposed baryonic components (stellar disk, gas)
- Galaxy properties (distance, inclination, luminosity)

Sources:
- Official: http://astroweb.case.edu/SPARC/
- Paper: Lelli et al. (2016), AJ 152, 157
- Data: Rotmod_LTG.zip containing individual galaxy files
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

# URLs (primary and fallback)
# Prefer stable archived source first to ensure reproducibility
STABLE_SPARC_ARCHIVE = "https://zenodo.org/record/4274930/files/Rotmod_LTG.zip?download=1"
SPARC_URLS = [
    STABLE_SPARC_ARCHIVE,
    "http://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
]

# Table with galaxy properties
SPARC_TABLE_URL = "http://astroweb.case.edu/SPARC/SPARC_Lelli2016c.mrt"


@dataclass
class GalaxyData:
    """
    Container for a single galaxy's rotation curve data.
    """
    name: str
    distance: float           # Mpc
    inclination: float        # degrees
    luminosity: float         # 10^9 L_sun
    eff_radius: float         # kpc (effective radius)
    disk_scale: float         # kpc (disk scale length)
    MHI: float                # 10^9 M_sun (HI mass)
    quality: int              # Quality flag (1=best, 3=worst)

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
        # Handle negative values (can happen due to errors)
        return np.sqrt(np.maximum(v_dm_sq, 0))

    @property
    def n_points(self) -> int:
        """Number of rotation curve points."""
        return len(self.radius)

    @property
    def r_max(self) -> float:
        """Maximum radius in kpc."""
        return self.radius[-1] if len(self.radius) > 0 else 0.0

    @property
    def v_max(self) -> float:
        """Maximum observed velocity in km/s."""
        return np.max(self.v_obs) if len(self.v_obs) > 0 else 0.0

    def is_dwarf(self, v_max_cut: float = 80.0) -> bool:
        """Check if galaxy is a dwarf (v_max < cut)."""
        return self.v_max < v_max_cut

    def is_lsb(self, mu_0_cut: float = 22.0) -> bool:
        """
        Check if galaxy is low surface brightness.

        Uses central surface brightness proxy from luminosity and scale length.
        """
        # Approximate central surface brightness
        # mu_0 ~ -2.5 log10(L / (2 pi h^2)) + const
        if self.disk_scale > 0 and self.luminosity > 0:
            L_per_area = self.luminosity / (2 * np.pi * self.disk_scale**2)
            mu_0 = -2.5 * np.log10(L_per_area * 1e9) + 21.57  # Approximate
            return mu_0 > mu_0_cut
        return False


def download_sparc(force: bool = False,
                   verbose: bool = True,
                   prefer_archive: bool = True) -> bool:
    """
    Download SPARC data from official source.

    Parameters:
        force: Re-download even if files exist
        verbose: Print progress
        prefer_archive: Try archived mirror first (Zenodo) for reproducibility

    Returns:
        True if successful
    """
    SPARC_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if not force and (SPARC_DIR / "rotmod").exists():
        if verbose:
            print("SPARC data already downloaded.")
        return True

    if verbose:
        source_pref = "archived mirror" if prefer_archive else "official site first"
        print(f"Downloading SPARC rotation curve data ({source_pref})...")

    # Try primary URL first
    success = False
    url_candidates = SPARC_URLS if prefer_archive else list(reversed(SPARC_URLS))

    for url in url_candidates:
        try:
            if verbose:
                print(f"  Trying: {url}")

            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Check if it's a zip file
            content_type = response.headers.get("Content-Type", "").lower()
            is_zip = url.lower().endswith('.zip') or 'zip' in content_type
            if is_zip:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    zf.extractall(SPARC_DIR)
                success = True
                break
            else:
                # MRT format - save as text
                (SPARC_DIR / "SPARC_table.mrt").write_bytes(response.content)
                success = True
                break

        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            continue

    # Also try to get the summary table
    try:
        if verbose:
            print("  Downloading galaxy properties table...")
        response = requests.get(SPARC_TABLE_URL, timeout=30)
        response.raise_for_status()
        (SPARC_DIR / "SPARC_Lelli2016c.mrt").write_bytes(response.content)
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not download properties table: {e}")

    if success and verbose:
        print("SPARC download complete.")

    return success


def _parse_mrt_table(filepath: Path) -> pd.DataFrame:
    """
    Parse SPARC MRT (Machine-Readable Table) format.

    The MRT format has a header section followed by data.
    """
    lines = filepath.read_text().split('\n')

    # Find data start (after header)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            data_start = i + 1
            break

    # Parse data lines
    data = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 10:
                data.append(parts)

    # Create DataFrame with expected columns
    columns = [
        'Name', 'T', 'D', 'e_D', 'Inc', 'e_Inc',
        'L36', 'e_L36', 'Reff', 'SBeff', 'Rdisk',
        'SBdisk', 'MHI', 'RHI', 'Vflat', 'e_Vflat',
        'Q', 'Ref'
    ]

    df = pd.DataFrame(data)
    if len(df.columns) >= len(columns):
        df = df.iloc[:, :len(columns)]
        df.columns = columns
    else:
        # Fallback: create with available columns
        df.columns = [f'col{i}' for i in range(len(df.columns))]

    return df


def load_sparc_catalog(verbose: bool = True) -> pd.DataFrame:
    """
    Load SPARC galaxy catalog with properties.

    Returns DataFrame with columns:
        - name: Galaxy name
        - distance: Distance in Mpc
        - inclination: Inclination in degrees
        - luminosity: 3.6um luminosity in 10^9 L_sun
        - eff_radius: Effective radius in kpc
        - disk_scale: Disk scale length in kpc
        - MHI: HI mass in 10^9 M_sun
        - quality: Quality flag
        - v_flat: Flat rotation velocity in km/s
    """
    # Ensure data is downloaded
    download_sparc(verbose=verbose)

    # Check for MRT table
    mrt_path = SPARC_DIR / "SPARC_Lelli2016c.mrt"

    if mrt_path.exists():
        try:
            df = _parse_mrt_table(mrt_path)
            if verbose:
                print(f"Loaded {len(df)} galaxies from catalog.")
            return df
        except Exception as e:
            if verbose:
                print(f"Warning: Could not parse MRT table: {e}")

    # Fallback: scan rotmod directory for galaxy files
    rotmod_dir = SPARC_DIR / "Rotmod_LTG"
    if not rotmod_dir.exists():
        rotmod_dir = SPARC_DIR / "rotmod"

    if rotmod_dir.exists():
        galaxies = []
        for f in rotmod_dir.glob("*.dat"):
            name = f.stem
            galaxies.append({'name': name})

        df = pd.DataFrame(galaxies)
        if verbose:
            print(f"Found {len(df)} galaxy files.")
        return df

    raise FileNotFoundError("SPARC data not found. Run download_sparc() first.")


def _parse_rotmod_file(filepath: Path) -> Dict:
    """
    Parse a single SPARC rotation curve file.

    Format:
        Header lines starting with comments
        Data columns: Rad Vobs errV Vgas Vdisk Vbul
    """
    lines = filepath.read_text().split('\n')

    # Parse header for galaxy properties
    props = {
        'name': filepath.stem,
        'distance': 10.0,
        'inclination': 60.0,
        'luminosity': 1.0,
        'eff_radius': 1.0,
        'disk_scale': 1.0,
        'MHI': 1.0,
        'quality': 2,
    }

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for header info
        if line.startswith('#') or line.startswith('!'):
            # Try to parse properties from comments
            if 'Distance' in line or 'D=' in line:
                try:
                    val = float(line.split('=')[-1].split()[0])
                    props['distance'] = val
                except:
                    pass
            if 'Inc' in line or 'i=' in line:
                try:
                    val = float(line.split('=')[-1].split()[0])
                    props['inclination'] = val
                except:
                    pass
            continue

        # Data line
        parts = line.split()
        if len(parts) >= 4:
            try:
                vals = [float(p) for p in parts[:6]]
                # Pad with zeros if fewer columns
                while len(vals) < 6:
                    vals.append(0.0)
                data_lines.append(vals)
            except ValueError:
                continue

    if not data_lines:
        raise ValueError(f"No data found in {filepath}")

    data = np.array(data_lines)

    props['radius'] = data[:, 0]
    props['v_obs'] = data[:, 1]
    props['v_err'] = data[:, 2]
    props['v_gas'] = data[:, 3]
    props['v_disk'] = data[:, 4] if data.shape[1] > 4 else np.zeros_like(data[:, 0])
    props['v_bulge'] = data[:, 5] if data.shape[1] > 5 else np.zeros_like(data[:, 0])

    return props


def load_rotation_curve(galaxy_name: str, verbose: bool = False) -> GalaxyData:
    """
    Load rotation curve data for a specific galaxy.

    Parameters:
        galaxy_name: Name of the galaxy (e.g., 'DDO154')
        verbose: Print status

    Returns:
        GalaxyData object
    """
    # Ensure data is downloaded
    download_sparc(verbose=verbose)

    # Find the file
    rotmod_dir = SPARC_DIR / "Rotmod_LTG"
    if not rotmod_dir.exists():
        rotmod_dir = SPARC_DIR / "rotmod"

    filepath = rotmod_dir / f"{galaxy_name}.dat"
    if not filepath.exists():
        # Try case-insensitive search
        for f in rotmod_dir.glob("*.dat"):
            if f.stem.lower() == galaxy_name.lower():
                filepath = f
                break
        else:
            raise FileNotFoundError(f"Galaxy {galaxy_name} not found in SPARC data")

    props = _parse_rotmod_file(filepath)

    return GalaxyData(**props)


def get_dwarf_lsb_sample(v_max_cut: float = 80.0,
                         min_points: int = 5,
                         verbose: bool = True) -> List[str]:
    """
    Get list of dwarf/LSB galaxies suitable for fitting.

    Selection criteria:
        - v_max < v_max_cut (dwarf criterion)
        - At least min_points data points
        - Quality flag <= 2

    Parameters:
        v_max_cut: Maximum rotation velocity for dwarf classification
        min_points: Minimum number of data points required
        verbose: Print progress

    Returns:
        List of galaxy names
    """
    # Ensure data is downloaded
    download_sparc(verbose=verbose)

    rotmod_dir = SPARC_DIR / "Rotmod_LTG"
    if not rotmod_dir.exists():
        rotmod_dir = SPARC_DIR / "rotmod"

    selected = []
    for filepath in sorted(rotmod_dir.glob("*.dat")):
        try:
            galaxy = load_rotation_curve(filepath.stem, verbose=False)

            # Apply selection
            if (galaxy.v_max < v_max_cut and
                galaxy.n_points >= min_points and
                galaxy.quality <= 2):
                selected.append(galaxy.name)

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {filepath.stem}: {e}")

    if verbose:
        print(f"Selected {len(selected)} dwarf/LSB galaxies for fitting")

    return selected


def create_synthetic_sparc_data(n_galaxies: int = 20,
                                seed: int = 42) -> List[GalaxyData]:
    """
    Create synthetic SPARC-like rotation curve data for testing.

    This is used when the real SPARC data cannot be downloaded.

    Parameters:
        n_galaxies: Number of synthetic galaxies
        seed: Random seed

    Returns:
        List of GalaxyData objects
    """
    np.random.seed(seed)

    galaxies = []
    for i in range(n_galaxies):
        # Random galaxy properties (dwarf/LSB-like)
        v_max = np.random.uniform(30, 80)  # km/s
        r_max = np.random.uniform(3, 10)   # kpc
        n_points = np.random.randint(8, 25)

        # Generate rotation curve
        r = np.linspace(0.5, r_max, n_points)

        # Simple model: rising then flat with some scatter
        v_flat = v_max
        r_turn = r_max * 0.3
        v_obs = v_flat * (1 - np.exp(-r / r_turn))

        # Add noise
        v_err = np.random.uniform(3, 8, n_points)
        v_obs += np.random.normal(0, v_err)

        # Baryonic components (simplified)
        v_disk = 0.5 * v_obs * np.exp(-r / (2 * r_turn))
        v_gas = 0.3 * v_obs * (r / r_max)
        v_bulge = np.zeros_like(r)

        galaxy = GalaxyData(
            name=f"SynGal{i+1:03d}",
            distance=np.random.uniform(3, 15),
            inclination=np.random.uniform(40, 80),
            luminosity=np.random.uniform(0.01, 1.0),
            eff_radius=r_max * 0.3,
            disk_scale=r_max * 0.2,
            MHI=np.random.uniform(0.1, 2.0),
            quality=1,
            radius=r,
            v_obs=np.maximum(v_obs, 5),  # Ensure positive
            v_err=v_err,
            v_gas=np.maximum(v_gas, 0),
            v_disk=np.maximum(v_disk, 0),
            v_bulge=v_bulge,
        )
        galaxies.append(galaxy)

    return galaxies
