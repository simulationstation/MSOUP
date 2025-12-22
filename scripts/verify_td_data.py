#!/usr/bin/env python3
"""
Verify TDCOSMO D_dt posterior data integrity.

This script:
- Lists which lenses were found
- Loads each posterior and prints N_samples, mean, std, median, 68% CI
- Confirms units (Mpc) or clearly records units if not Mpc
- Exits nonzero if fewer than 7 lenses are successfully parsed

Usage:
    python scripts/verify_td_data.py
"""

import sys
from pathlib import Path
import numpy as np

# Expected lenses
EXPECTED_LENSES = [
    "B1608+656",
    "RXJ1131-1231",
    "HE0435-1223",
    "SDSS1206+4332",
    "WFI2033-4723",
    "PG1115+080",
    "DES0408-5354",
]

DATA_DIR = Path(__file__).parent.parent / "data_ignored" / "tdcosmo_posteriors"


def verify_lens(npz_path: Path) -> dict:
    """Load and verify a single lens NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    samples = data["D_dt"]
    z_lens = float(data["z_lens"])
    z_source = float(data["z_source"])
    lens_id = str(data["lens_id"])
    units = str(data["units"])

    n_samples = len(samples)
    mean = np.mean(samples)
    std = np.std(samples)
    median = np.median(samples)
    p16 = np.percentile(samples, 16)
    p84 = np.percentile(samples, 84)

    return {
        "lens_id": lens_id,
        "z_lens": z_lens,
        "z_source": z_source,
        "N_samples": n_samples,
        "mean": mean,
        "std": std,
        "median": median,
        "CI68_low": p16,
        "CI68_high": p84,
        "units": units,
        "valid": n_samples > 0 and np.isfinite(mean) and np.isfinite(std),
    }


def main():
    print("=" * 70)
    print("TDCOSMO D_dt Posterior Data Verification")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        return 1

    # Find all NPZ files
    npz_files = list(DATA_DIR.glob("*_Ddt.npz"))
    print(f"Found {len(npz_files)} NPZ files\n")

    results = []
    errors = []

    for lens_name in EXPECTED_LENSES:
        npz_path = DATA_DIR / f"{lens_name}_Ddt.npz"

        if not npz_path.exists():
            print(f"MISSING: {lens_name}")
            errors.append(f"Missing: {lens_name}")
            continue

        try:
            result = verify_lens(npz_path)
            results.append(result)

            status = "OK" if result["valid"] else "INVALID"
            units_note = "" if result["units"] == "Mpc" else f" (UNITS: {result['units']})"

            print(f"{result['lens_id']:15s} | z_l={result['z_lens']:.3f} z_s={result['z_source']:.3f} | "
                  f"N={result['N_samples']:>9,} | "
                  f"D_dt = {result['mean']:7.1f} ± {result['std']:6.1f} Mpc | "
                  f"median = {result['median']:7.1f} | "
                  f"68% CI = [{result['CI68_low']:.1f}, {result['CI68_high']:.1f}]{units_note} | {status}")

        except Exception as e:
            print(f"ERROR: {lens_name}: {e}")
            errors.append(f"Error loading {lens_name}: {e}")

    # Summary
    print("\n" + "=" * 70)
    n_valid = sum(1 for r in results if r["valid"])
    n_expected = len(EXPECTED_LENSES)

    print(f"Summary: {n_valid}/{n_expected} lenses successfully loaded and verified")

    # Check units
    all_mpc = all(r["units"] == "Mpc" for r in results)
    if all_mpc:
        print("Units: All D_dt values are in Mpc (confirmed)")
    else:
        non_mpc = [r["lens_id"] for r in results if r["units"] != "Mpc"]
        print(f"Units WARNING: Non-Mpc units found for: {non_mpc}")

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  - {err}")

    # Exit code
    if n_valid < n_expected:
        print(f"\nFAILED: Only {n_valid}/{n_expected} lenses verified")
        return 1

    print("\nPASSED: All 7 lenses verified successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
