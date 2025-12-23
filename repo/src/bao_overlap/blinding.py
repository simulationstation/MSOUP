"""Blinding utilities for BAO environment-overlap analysis.

This module implements a strict blinding protocol to prevent peeking at
the primary endpoint (beta significance) before all robustness checks
are complete.

Blinding Method:
1. Beta offset: Add random offset kappa ~ U(-0.5, 0.5) to beta
2. Encryption: Encrypt true beta, sigma_beta, and significance using Fernet
3. Key isolation: Store encryption key in ~/.bao_blind_key with 0600 permissions
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Fernet requires cryptography package
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


@dataclass
class BlindState:
    """Track blinding status for the analysis."""
    unblind: bool = False
    key_file: Path = field(default_factory=lambda: Path.home() / ".bao_blind_key")
    offset_applied: bool = False
    kappa: float = 0.0

    def __post_init__(self):
        if isinstance(self.key_file, str):
            self.key_file = Path(self.key_file).expanduser()


@dataclass
class BlindedResult:
    """Container for blinded analysis results."""
    beta_blinded: float
    beta_encrypted: bytes
    sigma_beta_encrypted: bytes
    significance_encrypted: bytes
    kappa_encrypted: bytes
    prereg_hash: str
    timestamp: str
    is_blinded: bool = True


def require_unblind(state: BlindState) -> None:
    """Raise error if attempting to access unblinded results while blinded."""
    if not state.unblind:
        raise RuntimeError(
            "BLINDING VIOLATION: Attempting to access unblinded results.\n"
            "Set blinding.unblind=true and use --confirm-unblind flag to proceed.\n"
            "Ensure all robustness checks are complete before unblinding."
        )


def generate_blinding_key(key_file: Path) -> bytes:
    """Generate and store a new blinding key.

    The key is stored with restrictive permissions (0600) to prevent
    unauthorized access.
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography package required for blinding. Install with: pip install cryptography")

    key_file = Path(key_file).expanduser()

    if key_file.exists():
        raise FileExistsError(
            f"Blinding key already exists at {key_file}. "
            "Delete it manually if you want to regenerate (this will invalidate all blinded results)."
        )

    key = Fernet.generate_key()

    # Write with restrictive permissions
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(key)
    os.chmod(key_file, 0o600)

    return key


def load_blinding_key(key_file: Path) -> bytes:
    """Load existing blinding key."""
    key_file = Path(key_file).expanduser()

    if not key_file.exists():
        raise FileNotFoundError(
            f"Blinding key not found at {key_file}. "
            "Run initialize_blinding() first."
        )

    # Check permissions
    mode = key_file.stat().st_mode & 0o777
    if mode != 0o600:
        raise PermissionError(
            f"Blinding key has insecure permissions ({oct(mode)}). "
            "Expected 0600. Fix with: chmod 600 {key_file}"
        )

    return key_file.read_bytes()


def initialize_blinding(state: BlindState, rng: np.random.Generator) -> BlindState:
    """Initialize blinding for a new analysis run.

    Generates the random offset kappa and ensures the encryption key exists.
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography package required for blinding")

    # Generate or load key
    try:
        key = load_blinding_key(state.key_file)
    except FileNotFoundError:
        key = generate_blinding_key(state.key_file)

    # Generate random offset
    kappa = rng.uniform(-0.5, 0.5)

    return BlindState(
        unblind=False,
        key_file=state.key_file,
        offset_applied=True,
        kappa=kappa
    )


def blind_results(
    beta: float,
    sigma_beta: float,
    state: BlindState,
    prereg_hash: str,
) -> BlindedResult:
    """Apply blinding to analysis results.

    Parameters
    ----------
    beta : float
        True beta value from analysis
    sigma_beta : float
        Uncertainty on beta
    state : BlindState
        Current blinding state with kappa offset
    prereg_hash : str
        SHA256 hash of preregistration.yaml for audit trail

    Returns
    -------
    BlindedResult
        Blinded results with encrypted true values
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography package required for blinding")

    if not state.offset_applied:
        raise RuntimeError("Blinding not initialized. Call initialize_blinding() first.")

    # Load encryption key
    key = load_blinding_key(state.key_file)
    fernet = Fernet(key)

    # Compute derived quantities
    significance = abs(beta) / sigma_beta if sigma_beta > 0 else 0.0

    # Apply offset to beta
    beta_blinded = beta + state.kappa

    # Encrypt true values
    beta_encrypted = fernet.encrypt(str(beta).encode())
    sigma_beta_encrypted = fernet.encrypt(str(sigma_beta).encode())
    significance_encrypted = fernet.encrypt(str(significance).encode())
    kappa_encrypted = fernet.encrypt(str(state.kappa).encode())

    return BlindedResult(
        beta_blinded=beta_blinded,
        beta_encrypted=beta_encrypted,
        sigma_beta_encrypted=sigma_beta_encrypted,
        significance_encrypted=significance_encrypted,
        kappa_encrypted=kappa_encrypted,
        prereg_hash=prereg_hash,
        timestamp=datetime.utcnow().isoformat(),
        is_blinded=True
    )


@dataclass
class UnblindedResult:
    """Container for unblinded analysis results."""
    beta: float
    sigma_beta: float
    significance: float
    kappa: float
    prereg_hash: str
    unblind_timestamp: str
    audit_entry: dict


def unblind_results(
    blinded: BlindedResult,
    state: BlindState,
    confirm: bool = False,
    audit_log_path: Path | None = None,
) -> UnblindedResult:
    """Unblind analysis results.

    IMPORTANT: This should only be called after all robustness checks
    are complete and the analysis is ready for final interpretation.

    Parameters
    ----------
    blinded : BlindedResult
        Blinded results to decrypt
    state : BlindState
        Blinding state (must have unblind=True)
    confirm : bool
        Must be True to proceed (safety check)
    audit_log_path : Path, optional
        Path to audit log file for recording unblinding event

    Returns
    -------
    UnblindedResult
        Decrypted true values
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography package required for blinding")

    if not confirm:
        raise RuntimeError(
            "UNBLINDING BLOCKED: Must pass confirm=True to unblind.\n"
            "Ensure all robustness checks are complete."
        )

    if not state.unblind:
        raise RuntimeError(
            "UNBLINDING BLOCKED: BlindState.unblind must be True.\n"
            "Set blinding.unblind=true in config to enable."
        )

    # Load encryption key
    key = load_blinding_key(state.key_file)
    fernet = Fernet(key)

    # Decrypt values
    beta = float(fernet.decrypt(blinded.beta_encrypted).decode())
    sigma_beta = float(fernet.decrypt(blinded.sigma_beta_encrypted).decode())
    significance = float(fernet.decrypt(blinded.significance_encrypted).decode())
    kappa = float(fernet.decrypt(blinded.kappa_encrypted).decode())

    # Create audit entry
    audit_entry = {
        "event": "unblind",
        "timestamp": datetime.utcnow().isoformat(),
        "user": os.environ.get("USER", "unknown"),
        "prereg_hash": blinded.prereg_hash,
        "blinded_timestamp": blinded.timestamp,
        "beta": beta,
        "sigma_beta": sigma_beta,
        "significance": significance,
    }

    # Write audit log
    if audit_log_path is not None:
        audit_log_path = Path(audit_log_path)
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing log or create new
        existing = []
        if audit_log_path.exists():
            with open(audit_log_path) as f:
                existing = json.load(f)

        existing.append(audit_entry)

        with open(audit_log_path, "w") as f:
            json.dump(existing, f, indent=2)

    return UnblindedResult(
        beta=beta,
        sigma_beta=sigma_beta,
        significance=significance,
        kappa=kappa,
        prereg_hash=blinded.prereg_hash,
        unblind_timestamp=audit_entry["timestamp"],
        audit_entry=audit_entry
    )


def save_blinded_results(blinded: BlindedResult, path: Path) -> None:
    """Save blinded results to JSON file."""
    path = Path(path)

    data = {
        "beta_blinded": blinded.beta_blinded,
        "beta_encrypted": blinded.beta_encrypted.decode() if isinstance(blinded.beta_encrypted, bytes) else blinded.beta_encrypted,
        "sigma_beta_encrypted": blinded.sigma_beta_encrypted.decode() if isinstance(blinded.sigma_beta_encrypted, bytes) else blinded.sigma_beta_encrypted,
        "significance_encrypted": blinded.significance_encrypted.decode() if isinstance(blinded.significance_encrypted, bytes) else blinded.significance_encrypted,
        "kappa_encrypted": blinded.kappa_encrypted.decode() if isinstance(blinded.kappa_encrypted, bytes) else blinded.kappa_encrypted,
        "prereg_hash": blinded.prereg_hash,
        "timestamp": blinded.timestamp,
        "is_blinded": blinded.is_blinded,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_blinded_results(path: Path) -> BlindedResult:
    """Load blinded results from JSON file."""
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    return BlindedResult(
        beta_blinded=data["beta_blinded"],
        beta_encrypted=data["beta_encrypted"].encode() if isinstance(data["beta_encrypted"], str) else data["beta_encrypted"],
        sigma_beta_encrypted=data["sigma_beta_encrypted"].encode() if isinstance(data["sigma_beta_encrypted"], str) else data["sigma_beta_encrypted"],
        significance_encrypted=data["significance_encrypted"].encode() if isinstance(data["significance_encrypted"], str) else data["significance_encrypted"],
        kappa_encrypted=data["kappa_encrypted"].encode() if isinstance(data["kappa_encrypted"], str) else data["kappa_encrypted"],
        prereg_hash=data["prereg_hash"],
        timestamp=data["timestamp"],
        is_blinded=data["is_blinded"],
    )


def compute_prereg_hash(prereg_path: Path) -> str:
    """Compute SHA256 hash of preregistration.yaml for audit trail."""
    prereg_path = Path(prereg_path)
    content = prereg_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def destroy_blinding_key(key_file: Path, confirm: bool = False) -> None:
    """Securely delete the blinding key after analysis is complete.

    WARNING: This is irreversible. All encrypted results will become
    permanently inaccessible.
    """
    if not confirm:
        raise RuntimeError(
            "KEY DESTRUCTION BLOCKED: Must pass confirm=True.\n"
            "WARNING: This will permanently invalidate all blinded results."
        )

    key_file = Path(key_file).expanduser()

    if not key_file.exists():
        return

    # Overwrite with random data before deletion
    size = key_file.stat().st_size
    key_file.write_bytes(secrets.token_bytes(size))
    key_file.unlink()
