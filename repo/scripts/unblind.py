#!/usr/bin/env python3
"""Unblind BAO environment-overlap analysis results.

This script should ONLY be run after all robustness checks are complete.
It requires explicit confirmation and logs all actions for audit trail.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bao_overlap.blinding import (
    BlindState,
    load_blinded_results,
    unblind_results,
    compute_prereg_hash,
)


def check_robustness_complete(results_dir: Path) -> tuple[bool, dict]:
    """Check if all robustness checks are complete and passed."""
    summary_file = results_dir / "robustness_summary.json"

    if not summary_file.exists():
        return False, {"error": "robustness_summary.json not found"}

    with open(summary_file) as f:
        summary = json.load(f)

    total = len(summary.get("variants", []))
    passed = summary.get("pass_count", 0)
    failed = summary.get("fail_count", 0)

    # Require >50% pass rate (as per preregistration stopping rules)
    pass_rate = passed / total if total > 0 else 0
    is_complete = total > 0 and pass_rate >= 0.5

    return is_complete, {
        "total_variants": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
    }


def verify_prereg_hash(results_dir: Path, prereg_path: Path) -> tuple[bool, str, str]:
    """Verify preregistration hash matches stored hash."""
    stored_hash_file = results_dir / "prereg_hash.txt"

    if not stored_hash_file.exists():
        return False, "", "stored hash not found"

    stored_hash = stored_hash_file.read_text().strip()
    current_hash = compute_prereg_hash(prereg_path)

    if stored_hash != current_hash:
        return False, stored_hash, f"hash mismatch: stored={stored_hash[:16]}... current={current_hash[:16]}..."

    return True, stored_hash, "verified"


def main():
    parser = argparse.ArgumentParser(
        description="Unblind BAO analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This script should only be run after ALL of the following:
  1. All robustness checks are complete
  2. Mock analysis is complete
  3. You have reviewed the blinded diagnostics
  4. You are ready to interpret the final results

The unblinding action is logged and cannot be undone.
        """,
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to blinded_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for unblinded_results.json",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="Path to audit log file",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "preregistration.yaml",
        help="Path to preregistration.yaml",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm that you want to unblind (required)",
    )
    parser.add_argument(
        "--skip-robustness-check",
        action="store_true",
        help="Skip robustness completion check (NOT RECOMMENDED)",
    )

    args = parser.parse_args()

    # Safety checks
    if not args.confirm:
        print("ERROR: Must pass --confirm to proceed with unblinding.")
        print("Ensure all robustness checks are complete.")
        sys.exit(1)

    if not args.results.exists():
        print(f"ERROR: Blinded results not found: {args.results}")
        sys.exit(1)

    results_dir = args.results.parent

    # Verify preregistration hash
    hash_ok, prereg_hash, hash_msg = verify_prereg_hash(results_dir, args.prereg)
    if not hash_ok:
        print(f"ERROR: Preregistration hash verification failed: {hash_msg}")
        print("The preregistration.yaml may have been modified after analysis started.")
        sys.exit(1)
    print(f"Preregistration hash verified: {prereg_hash[:16]}...")

    # Check robustness completion
    if not args.skip_robustness_check:
        robustness_ok, robustness_info = check_robustness_complete(results_dir)
        if not robustness_ok:
            print("ERROR: Robustness checks not complete or failed.")
            print(f"Details: {json.dumps(robustness_info, indent=2)}")
            print("Run robustness stage or use --skip-robustness-check (not recommended).")
            sys.exit(1)
        print(f"Robustness checks: {robustness_info['passed']}/{robustness_info['total_variants']} passed")
    else:
        print("WARNING: Skipping robustness check (not recommended)")

    # Load blinded results
    print(f"Loading blinded results from: {args.results}")
    blinded = load_blinded_results(args.results)

    if not blinded.is_blinded:
        print("Results are already unblinded.")
        sys.exit(0)

    # Create unblind state
    state = BlindState(unblind=True)

    # Perform unblinding
    print("\n" + "=" * 60)
    print("UNBLINDING ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Preregistration hash: {prereg_hash}")
    print("=" * 60 + "\n")

    unblinded = unblind_results(
        blinded=blinded,
        state=state,
        confirm=True,
        audit_log_path=args.audit_log,
    )

    # Display results
    print("UNBLINDED RESULTS:")
    print(f"  beta = {unblinded.beta:.6f}")
    print(f"  sigma_beta = {unblinded.sigma_beta:.6f}")
    print(f"  significance = {unblinded.significance:.2f} sigma")
    print(f"  blinding offset (kappa) = {unblinded.kappa:.6f}")

    # Decision based on preregistration criteria
    print("\n" + "-" * 60)
    print("DECISION CRITERIA EVALUATION:")

    if unblinded.significance >= 3.0:
        print(f"  [!] Statistical significance: {unblinded.significance:.2f}σ >= 3.0σ threshold")
        print("  --> MEETS detection criterion #1")
    elif unblinded.significance >= 2.0:
        print(f"  [?] Statistical significance: {unblinded.significance:.2f}σ (inconclusive)")
    else:
        print(f"  [✓] Statistical significance: {unblinded.significance:.2f}σ < 2.0σ")
        print("  --> Consistent with null hypothesis")

    print("-" * 60)
    print("NOTE: Mock calibration and reconstruction checks required for full decision.")
    print("See preregistration.md Section 7 for complete decision criteria.")

    # Save unblinded results
    output_data = {
        "beta": unblinded.beta,
        "sigma_beta": unblinded.sigma_beta,
        "significance": unblinded.significance,
        "kappa": unblinded.kappa,
        "prereg_hash": unblinded.prereg_hash,
        "unblind_timestamp": unblinded.unblind_timestamp,
        "is_blinded": False,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nUnblinded results saved to: {args.output}")
    print(f"Audit log updated: {args.audit_log}")


if __name__ == "__main__":
    main()
