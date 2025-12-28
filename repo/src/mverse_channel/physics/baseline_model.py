"""Baseline open quantum system simulation."""

from __future__ import annotations

import importlib
from typing import Dict

import numpy as np

from mverse_channel.config import SimulationConfig


def _load_qutip():
    if importlib.util.find_spec("qutip") is None:
        raise RuntimeError("qutip is required for baseline simulations")
    import qutip  # noqa: PLC0415

    return qutip


def simulate_baseline(config: SimulationConfig) -> Dict[str, np.ndarray]:
    """Simulate baseline dynamics and return quadrature time series."""
    qutip = _load_qutip()
    tlist, n_steps = config.time_grid()

    a = qutip.tensor(qutip.destroy(config.n_levels), qutip.qeye(config.n_levels))
    b = qutip.tensor(qutip.qeye(config.n_levels), qutip.destroy(config.n_levels))

    n_a = a.dag() * a
    n_b = b.dag() * b

    h0 = config.omega_a * n_a + config.omega_b * n_b
    h0 = h0 + config.coupling_j * (a.dag() * b + a * b.dag())

    if config.kerr_a:
        h0 = h0 + config.kerr_a * n_a * n_a
    if config.kerr_b:
        h0 = h0 + config.kerr_b * n_b * n_b

    def mod_coeff(t, args):
        return config.modulation.delta_omega * np.sin(config.modulation.omega_mod * t)

    def drive_coeff_a(t, args):
        return config.drive_amp_a * np.cos(config.drive_freq * t + config.drive_phase)

    def drive_coeff_b(t, args):
        return config.drive_amp_b * np.cos(config.drive_freq * t + config.drive_phase)

    hamiltonian = [h0]
    if config.modulation.enabled:
        hamiltonian.append([n_a, mod_coeff])
    hamiltonian.append([a + a.dag(), drive_coeff_a])
    hamiltonian.append([b + b.dag(), drive_coeff_b])

    c_ops = [
        np.sqrt(config.kappa_a * (1 + config.nth_a)) * a,
        np.sqrt(config.kappa_a * config.nth_a) * a.dag(),
        np.sqrt(config.kappa_b * (1 + config.nth_b)) * b,
        np.sqrt(config.kappa_b * config.nth_b) * b.dag(),
    ]

    result = qutip.mesolve(
        hamiltonian,
        qutip.tensor(qutip.basis(config.n_levels, 0), qutip.basis(config.n_levels, 0)),
        tlist,
        c_ops,
        e_ops=[a, a.dag(), b, b.dag()],
    )

    a_exp = np.array(result.expect[0])
    a_dag_exp = np.array(result.expect[1])
    b_exp = np.array(result.expect[2])
    b_dag_exp = np.array(result.expect[3])

    ia = (a_exp + a_dag_exp).real
    qa = (-1j * (a_exp - a_dag_exp)).real
    ib = (b_exp + b_dag_exp).real
    qb = (-1j * (b_exp - b_dag_exp)).real

    return {
        "t": np.array(tlist),
        "ia": ia,
        "qa": qa,
        "ib": ib,
        "qb": qb,
        "n_steps": np.array([n_steps]),
    }
