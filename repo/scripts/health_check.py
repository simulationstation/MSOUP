#!/usr/bin/env python3
"""Mid-run health check for BAO environment overlap pipeline."""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

def find_nearest_idx(arr, val):
    """Find index of value nearest to val in sorted array."""
    return np.abs(arr - val).argmin()

def run_health_check(run_dir: Path, jk_iteration: int, jk_vectors: np.ndarray = None):
    """
    Run health check on pipeline outputs.

    Parameters
    ----------
    run_dir : Path
        Run output directory
    jk_iteration : int
        Current JK iteration number
    jk_vectors : np.ndarray, optional
        Partial JK vectors if available, shape (n_jk_so_far, n_total)

    Returns
    -------
    dict
        Health check results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "jk_iteration": jk_iteration,
        "bins": {},
        "overall_status": "PASS",
        "issues": []
    }

    # Load xi wedge data
    xi_path = run_dir / "xi_wedge_by_Ebin.npz"
    if not xi_path.exists():
        results["overall_status"] = "FAIL"
        results["issues"].append("xi_wedge_by_Ebin.npz not found")
        return results

    xi_data = np.load(xi_path, allow_pickle=True)
    xi_tangential = xi_data["xi_tangential"]  # shape (n_bins, n_s)
    s_centers = xi_data["s_centers"]
    n_bins, n_s = xi_tangential.shape

    # Find indices for BAO bump analysis
    # BAO peak ~105 Mpc/h, reference at ~80 and ~130
    idx_105 = find_nearest_idx(s_centers, 105)
    idx_80 = find_nearest_idx(s_centers, 80)
    idx_130 = find_nearest_idx(s_centers, 130)

    results["s_centers_used"] = {
        "s_105": float(s_centers[idx_105]),
        "s_80": float(s_centers[idx_80]),
        "s_130": float(s_centers[idx_130])
    }

    bump_contrasts = []

    for b in range(n_bins):
        xi_b = xi_tangential[b]
        bin_result = {
            "n_s": int(n_s),
            "nan_count": int(np.isnan(xi_b).sum()),
            "inf_count": int(np.isinf(xi_b).sum()),
            "xi_min": float(np.nanmin(xi_b)),
            "xi_max": float(np.nanmax(xi_b)),
            "xi_mean": float(np.nanmean(xi_b)),
        }

        # BAO bump contrast
        xi_peak = xi_b[idx_105]
        xi_ref = 0.5 * (xi_b[idx_80] + xi_b[idx_130])
        bump_contrast = xi_peak - xi_ref
        bin_result["bao_bump_contrast"] = float(bump_contrast)
        bin_result["xi_at_105"] = float(xi_peak)
        bin_result["xi_ref_mean"] = float(xi_ref)
        bump_contrasts.append(bump_contrast)

        # Check for issues
        if bin_result["nan_count"] > 0:
            results["overall_status"] = "FAIL"
            results["issues"].append(f"Bin {b}: {bin_result['nan_count']} NaN values")
        if bin_result["inf_count"] > 0:
            results["overall_status"] = "FAIL"
            results["issues"].append(f"Bin {b}: {bin_result['inf_count']} Inf values")

        # Partial covariance analysis if JK vectors available
        if jk_vectors is not None and jk_vectors.shape[0] >= 3:
            n_jk = jk_vectors.shape[0]
            # Extract this bin's portion (assuming concatenated bins of n_s each)
            start_idx = b * n_s
            end_idx = (b + 1) * n_s
            jk_bin = jk_vectors[:, start_idx:end_idx]

            # Compute partial covariance
            jk_mean = jk_bin.mean(axis=0)
            delta = jk_bin - jk_mean
            partial_cov = ((n_jk - 1) / n_jk) * (delta.T @ delta)

            # Condition number and eigenvalues
            try:
                eigenvalues = np.linalg.eigvalsh(partial_cov)
                min_eig = float(eigenvalues.min())
                max_eig = float(eigenvalues.max())
                cond_num = max_eig / max(abs(min_eig), 1e-15)

                bin_result["partial_cov_min_eigenvalue"] = min_eig
                bin_result["partial_cov_max_eigenvalue"] = max_eig
                bin_result["partial_cov_condition_number"] = cond_num
                bin_result["partial_cov_n_jk"] = int(n_jk)

                if cond_num > 1e8:
                    results["overall_status"] = "FAIL"
                    results["issues"].append(f"Bin {b}: condition number {cond_num:.2e} > 1e8")
                if min_eig < -1e-10 * max_eig:  # Substantially negative
                    results["overall_status"] = "FAIL"
                    results["issues"].append(f"Bin {b}: min eigenvalue {min_eig:.2e} substantially negative")
            except Exception as e:
                bin_result["partial_cov_error"] = str(e)
                results["issues"].append(f"Bin {b}: covariance computation error: {e}")

        results["bins"][f"bin_{b}"] = bin_result

    # Check for wildly inconsistent bump contrasts (numerical artifact)
    bump_contrasts = np.array(bump_contrasts)
    if np.any(np.isnan(bump_contrasts)):
        results["issues"].append("Some bump contrasts are NaN")
    else:
        bump_std = np.std(bump_contrasts)
        bump_mean = np.mean(bump_contrasts)
        # If spread is huge relative to mean, might be artifact
        if abs(bump_mean) > 1e-10 and bump_std / abs(bump_mean) > 10:
            results["issues"].append(f"Bump contrast spread ({bump_std:.4f}) >> mean ({bump_mean:.4f})")
        results["bump_contrast_summary"] = {
            "mean": float(bump_mean),
            "std": float(bump_std),
            "values": [float(x) for x in bump_contrasts]
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jk-iter", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = run_health_check(args.run_dir, args.jk_iter)

    # Print summary
    print(f"\n{'='*60}")
    print(f"HEALTH CHECK @ JK {args.jk_iter}")
    print(f"{'='*60}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Status: {results['overall_status']}")

    print(f"\nPer-bin summary:")
    print(f"{'Bin':<6} {'n_s':<5} {'NaN':<5} {'Inf':<5} {'xi_mean':<10} {'bump_contrast':<14}")
    print("-" * 55)
    for b in range(len(results["bins"])):
        br = results["bins"][f"bin_{b}"]
        print(f"{b:<6} {br['n_s']:<5} {br['nan_count']:<5} {br['inf_count']:<5} "
              f"{br['xi_mean']:<10.4f} {br['bao_bump_contrast']:<14.4f}")

    if results.get("bump_contrast_summary"):
        bcs = results["bump_contrast_summary"]
        print(f"\nBump contrast: mean={bcs['mean']:.4f}, std={bcs['std']:.4f}")

    if results["issues"]:
        print(f"\nIssues detected:")
        for issue in results["issues"]:
            print(f"  - {issue}")

    print(f"\n{'='*60}\n")

    # Save to file
    output_path = args.output or (args.run_dir / "health_check.json")

    # Load existing or create new
    if output_path.exists():
        with open(output_path) as f:
            all_checks = json.load(f)
    else:
        all_checks = {"checks": []}

    all_checks["checks"].append(results)
    all_checks["latest_status"] = results["overall_status"]
    all_checks["latest_jk"] = args.jk_iter

    with open(output_path, "w") as f:
        json.dump(all_checks, f, indent=2)

    print(f"Results saved to {output_path}")

    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
