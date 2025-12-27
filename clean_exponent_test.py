#!/usr/bin/env python3
"""
Clean Exponent Test for Two-Gate Model

Sweeps κ with fixed geometric gate and matches neutrality permeability
so that p_neu ≈ p_geo at each κ.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import Tuple, List, Dict
import csv
import time
import os
import multiprocessing
import contextlib
import functools


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


@dataclass
class SimulationParams:
    """All simulation parameters."""
    L: int = 24
    T: float = 1.0
    J: float = 0.6
    h: float = 0.0
    kappa: float = 50.0
    g: float = 0.5
    mu: float = 1.0
    gamma: float = 0.5
    sigma: float = 1.0
    dtheta: float = 0.3
    delta: float = 0.4
    m_tol: float = 0.10
    q_level: float = 0.95
    n_burnin: int = 400
    n_sample: int = 1600
    n_batches: int = 20

    @property
    def N(self) -> int:
        return self.L * self.L


class TwoGateLattice:
    """2D periodic square lattice with spins, edge phases, and credit fields."""

    def __init__(self, params: SimulationParams, rng: np.random.Generator = None):
        self.params = params
        self.L = params.L
        self.N = params.N
        self.rng = rng if rng is not None else np.random.default_rng()

        self.spins = np.zeros(self.N, dtype=np.int8)
        self.credits = np.zeros(self.N, dtype=np.float64)

        self._setup_edges()
        self._setup_plaquettes()
        self.initialize_random()

        self._phi_cache = np.zeros(self.n_plaquettes, dtype=np.float64)
        self._q_cache = np.zeros(self.N, dtype=np.float64)
        self._update_all_caches()

    def _setup_edges(self):
        L = self.L
        self.n_edges = 2 * self.N
        self.theta = np.zeros(self.n_edges, dtype=np.float64)
        self.edge_endpoints = np.zeros((self.n_edges, 2), dtype=np.int32)

        for y in range(L):
            for x in range(L):
                i = x + y * L
                x_right = (x + 1) % L
                i_right = x_right + y * L
                h_edge = i
                self.edge_endpoints[h_edge] = [i, i_right]

                y_up = (y + 1) % L
                i_up = x + y_up * L
                v_edge = self.N + i
                self.edge_endpoints[v_edge] = [i, i_up]

        self.site_edges = [[] for _ in range(self.N)]
        for e_idx in range(self.n_edges):
            i, j = self.edge_endpoints[e_idx]
            self.site_edges[i].append((e_idx, +1))
            self.site_edges[j].append((e_idx, -1))

    def _setup_plaquettes(self):
        L = self.L
        self.n_plaquettes = self.N

        self.plaquette_edges = []
        self.plaquette_vertices = []

        for y in range(L):
            for x in range(L):
                i = x + y * L
                x_right = (x + 1) % L
                y_up = (y + 1) % L

                i_right = x_right + y * L
                i_up = x + y_up * L
                i_diag = x_right + y_up * L

                h_bottom = i
                v_right = self.N + i_right
                h_top = i_up
                v_left = self.N + i

                edges = [
                    (h_bottom, +1),
                    (v_right, +1),
                    (h_top, -1),
                    (v_left, -1)
                ]

                self.plaquette_edges.append(edges)
                self.plaquette_vertices.append([i, i_right, i_diag, i_up])

        self.edge_plaquettes = [[] for _ in range(self.n_edges)]
        for p_idx, edges in enumerate(self.plaquette_edges):
            for e_idx, _ in edges:
                self.edge_plaquettes[e_idx].append(p_idx)

        self.site_plaquettes = [[] for _ in range(self.N)]
        for p_idx, verts in enumerate(self.plaquette_vertices):
            for v in verts:
                self.site_plaquettes[v].append(p_idx)

    def get_theta(self, e_idx: int, orientation: int) -> float:
        return orientation * self.theta[e_idx]

    def compute_phi(self, p_idx: int) -> float:
        total = 0.0
        for e_idx, orient in self.plaquette_edges[p_idx]:
            total += self.get_theta(e_idx, orient)
        return wrap_angle(total)

    def _update_phi_cache(self, p_idx: int):
        self._phi_cache[p_idx] = self.compute_phi(p_idx)

    def _update_q_cache(self, site: int):
        q = 0.0
        for p_idx in self.site_plaquettes[site]:
            q += 1.0 - np.cos(self._phi_cache[p_idx])
        self._q_cache[site] = q

    def _update_all_caches(self):
        for p_idx in range(self.n_plaquettes):
            self._update_phi_cache(p_idx)
        for i in range(self.N):
            self._update_q_cache(i)

    def initialize_random(self):
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(np.int8)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.n_edges)
        self.credits = self.rng.normal(0, self.params.sigma, size=self.N)

    def delta_H_spin_flip(self, site: int) -> float:
        p = self.params
        s_old = self.spins[site]
        s_new = -s_old

        delta_Hs = 0.0
        for e_idx, orient in self.site_edges[site]:
            i, j = self.edge_endpoints[e_idx]
            neighbor = j if i == site else i
            delta_Hs += -p.J * (s_new - s_old) * self.spins[neighbor]

        delta_Hs += -p.h * (s_new - s_old)

        delta_HsT = 0.0
        for e_idx, orient in self.site_edges[site]:
            i, j = self.edge_endpoints[e_idx]
            other = j if i == site else i
            s_other = self.spins[other]

            P_old = ((1 + s_old) / 2) * ((1 + s_other) / 2)
            P_new = ((1 + s_new) / 2) * ((1 + s_other) / 2)

            cos_theta = np.cos(self.theta[e_idx])
            delta_HsT += -p.g * (P_new - P_old) * cos_theta

        c_i = self.credits[site]
        delta_Hc = -p.mu * c_i * (s_new - s_old)

        return delta_Hs + delta_HsT + delta_Hc

    def delta_H_edge_update(self, e_idx: int, theta_new: float) -> float:
        p = self.params
        theta_old = self.theta[e_idx]

        i, j = self.edge_endpoints[e_idx]
        P_ij = ((1 + self.spins[i]) / 2) * ((1 + self.spins[j]) / 2)
        delta_HsT = -p.g * P_ij * (np.cos(theta_new) - np.cos(theta_old))

        affected_plaquettes = self.edge_plaquettes[e_idx]
        old_phis = [self._phi_cache[p_idx] for p_idx in affected_plaquettes]

        self.theta[e_idx] = theta_new
        new_phis = [self.compute_phi(p_idx) for p_idx in affected_plaquettes]
        self.theta[e_idx] = theta_old

        delta_HT = 0.0
        for old_phi, new_phi in zip(old_phis, new_phis):
            delta_HT += p.kappa * ((1 - np.cos(new_phi)) - (1 - np.cos(old_phi)))

        affected_sites = set()
        for p_idx in affected_plaquettes:
            affected_sites.update(self.plaquette_vertices[p_idx])

        delta_Hc = 0.0
        for site in affected_sites:
            c_i = self.credits[site]
            delta_q = 0.0
            for p_idx in self.site_plaquettes[site]:
                if p_idx in affected_plaquettes:
                    idx = affected_plaquettes.index(p_idx)
                    delta_q += (1 - np.cos(new_phis[idx])) - (1 - np.cos(old_phis[idx]))
            delta_Hc += p.gamma * c_i * delta_q

        return delta_HT + delta_HsT + delta_Hc

    def update_spin(self, site: int) -> bool:
        delta_H = self.delta_H_spin_flip(site)
        if delta_H <= 0 or self.rng.random() < np.exp(-delta_H / self.params.T):
            self.spins[site] *= -1
            return True
        return False

    def update_edge(self, e_idx: int) -> bool:
        theta_old = self.theta[e_idx]
        delta = self.rng.uniform(-self.params.dtheta, self.params.dtheta)
        theta_new = wrap_angle(theta_old + delta)

        delta_H = self.delta_H_edge_update(e_idx, theta_new)

        if delta_H <= 0 or self.rng.random() < np.exp(-delta_H / self.params.T):
            self.theta[e_idx] = theta_new

            affected_plaquettes = self.edge_plaquettes[e_idx]
            for p_idx in affected_plaquettes:
                self._update_phi_cache(p_idx)

            affected_sites = set()
            for p_idx in affected_plaquettes:
                affected_sites.update(self.plaquette_vertices[p_idx])
            for site in affected_sites:
                self._update_q_cache(site)

            return True
        return False

    def update_credit_gibbs(self, site: int):
        p = self.params
        s_i = self.spins[site]
        q_i = self._q_cache[site]

        mean = p.sigma**2 * (p.mu * s_i - p.gamma * q_i)
        std = p.sigma * np.sqrt(p.T)

        self.credits[site] = self.rng.normal(mean, std)

    def sweep(self) -> Tuple[int, int]:
        """One full sweep. Returns (spin_accepts, edge_accepts)."""
        spin_accepts = 0
        edge_accepts = 0

        spin_order = self.rng.permutation(self.N)
        for site in spin_order:
            if self.update_spin(site):
                spin_accepts += 1

        edge_order = self.rng.permutation(self.n_edges)
        for e_idx in edge_order:
            if self.update_edge(e_idx):
                edge_accepts += 1

        for site in range(self.N):
            self.update_credit_gibbs(site)

        return spin_accepts, edge_accepts

    def magnetization(self) -> float:
        return np.mean(self.spins)

    def max_holonomy(self) -> float:
        return np.max(np.abs(self._phi_cache))

    def mean_holonomy_magnitude(self) -> float:
        return np.mean(np.abs(self._phi_cache))

    def geometric_gate(self, delta: float) -> bool:
        abs_phi = np.abs(self._phi_cache)
        q = np.quantile(abs_phi, self.params.q_level)
        return q <= delta

    def neutrality_gate(self, m_tol: float) -> bool:
        return np.abs(self.magnetization()) <= m_tol


def run_calibration_point(params: SimulationParams, delta: float, m_tol: float,
                          n_burnin: int = 200, n_sample: int = 600,
                          seed: int = None) -> Tuple[float, float]:
    """Quick run to measure p_geo and p_neu for given thresholds."""
    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    for _ in range(n_burnin):
        lattice.sweep()

    g_geo_count = 0
    g_neu_count = 0

    for _ in range(n_sample):
        lattice.sweep()
        if lattice.geometric_gate(delta):
            g_geo_count += 1
        if lattice.neutrality_gate(m_tol):
            g_neu_count += 1

    return g_geo_count / n_sample, g_neu_count / n_sample


def probe_run_stats(params: SimulationParams, n_burnin: int = 200,
                    n_sample: int = 400, seed: int = None) -> Dict[str, float]:
    """Probe run to measure magnetization and holonomy statistics."""
    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    for _ in range(n_burnin):
        lattice.sweep()

    abs_m = []
    phi_median = []
    phi_q = []
    phi_q99 = []
    phi_max = []

    for _ in range(n_sample):
        lattice.sweep()
        abs_phi = np.abs(lattice._phi_cache)
        abs_m.append(abs(lattice.magnetization()))
        phi_median.append(np.median(abs_phi))
        phi_q.append(np.quantile(abs_phi, params.q_level))
        phi_q99.append(np.quantile(abs_phi, 0.99))
        phi_max.append(np.max(abs_phi))

    return {
        'mean_abs_m': float(np.mean(abs_m)),
        'median_abs_phi': float(np.median(phi_median)),
        'median_q_level': float(np.median(phi_q)),
        'median_q99': float(np.median(phi_q99)),
        'median_max': float(np.median(phi_max)),
        'mean_q_level': float(np.mean(phi_q)),
    }


def _estimate_p_geo(params: SimulationParams, delta: float,
                    n_burnin: int, n_sample: int, seed: int) -> float:
    p_geo, _ = run_calibration_point(params, delta, m_tol=1.0,
                                     n_burnin=n_burnin, n_sample=n_sample, seed=seed)
    return p_geo


def calibrate_delta(params: SimulationParams, target_p: float, median_q: float,
                    n_burnin: int = 200, n_sample: int = 600,
                    max_iter: int = 10, tol: float = 0.02, seed_base: int = 123) -> Tuple[float, bool]:
    """Bisection search to find δ such that p_geo(δ) ≈ target_p."""
    print(f"  Calibrating δ for p_geo ≈ {target_p:.3f} (q_level={params.q_level:.2f})...")

    base = max(median_q, 1e-3)
    delta_low = max(0.5 * base, 1e-4)
    delta_high = min(1.5 * base, np.pi)

    p_low = _estimate_p_geo(params, delta_low, n_burnin, n_sample, seed_base)
    p_high = _estimate_p_geo(params, delta_high, n_burnin, n_sample, seed_base)
    print(f"    Initial bracket: δ={delta_low:.4f} -> p_geo={p_low:.3f}, "
          f"δ={delta_high:.4f} -> p_geo={p_high:.3f}")

    expand_steps = 0
    while not (p_low < target_p < p_high) and expand_steps < 6:
        if p_low >= target_p:
            delta_high = delta_low
            delta_low = max(delta_low * 0.5, 1e-4)
        elif p_high <= target_p:
            delta_low = delta_high
            delta_high = min(delta_high * 1.5, np.pi)
        p_low = _estimate_p_geo(params, delta_low, n_burnin, n_sample, seed_base)
        p_high = _estimate_p_geo(params, delta_high, n_burnin, n_sample, seed_base)
        expand_steps += 1
        print(f"    Expanded bracket {expand_steps}: δ={delta_low:.4f} -> p_geo={p_low:.3f}, "
              f"δ={delta_high:.4f} -> p_geo={p_high:.3f}")

    if not (p_low < target_p < p_high):
        print("    Failed to bracket target p_geo; marking κ infeasible.")
        return delta_high, False

    delta_mid = delta_high
    for it in range(max_iter):
        delta_mid = 0.5 * (delta_low + delta_high)
        p_mid = _estimate_p_geo(params, delta_mid, n_burnin, n_sample, seed_base + it)
        print(f"    Iter {it + 1}: δ={delta_mid:.4f} -> p_geo={p_mid:.3f}")
        if abs(p_mid - target_p) < tol:
            print(f"  Early stop: |p_geo - target| < {tol}")
            return delta_mid, True
        if p_mid < target_p:
            delta_low = delta_mid
        else:
            delta_high = delta_mid

    print(f"  Final δ={delta_mid:.4f} -> p_geo={p_mid:.3f}")
    return delta_mid, True


def calibrate_m_tol(params: SimulationParams, delta: float, target_p: float,
                    n_burnin: int = 200, n_sample: int = 600,
                    max_iter: int = 10, tol: float = 0.02, seed_base: int = 321) -> Tuple[float, bool]:
    """Bisection search to find m_tol such that p_neu(m_tol) ≈ target_p."""
    print(f"  Calibrating m_tol for p_neu ≈ {target_p:.3f}...")

    m_low, m_high = 0.01, 0.99
    _, p_low = run_calibration_point(params, delta, m_low,
                                     n_burnin=n_burnin, n_sample=n_sample, seed=seed_base)
    _, p_high = run_calibration_point(params, delta, m_high,
                                      n_burnin=n_burnin, n_sample=n_sample, seed=seed_base)
    print(f"    Initial bracket: m={m_low:.3f} -> p_neu={p_low:.3f}, "
          f"m={m_high:.3f} -> p_neu={p_high:.3f}")

    if not (p_low < target_p < p_high):
        print("    Failed to bracket target p_neu.")
        return m_high, False

    m_mid = m_high
    for it in range(max_iter):
        m_mid = 0.5 * (m_low + m_high)
        _, p_mid = run_calibration_point(params, delta, m_mid,
                                         n_burnin=n_burnin, n_sample=n_sample,
                                         seed=seed_base + it)
        print(f"    Iter {it + 1}: m={m_mid:.4f} -> p_neu={p_mid:.3f}")
        if abs(p_mid - target_p) < tol:
            print(f"  Early stop: |p_neu - target| < {tol}")
            return m_mid, True
        if p_mid < target_p:
            m_low = m_mid
        else:
            m_high = m_mid

    print(f"  Final m_tol={m_mid:.4f} -> p_neu={p_mid:.3f}")
    return m_mid, True


def run_main_sweep_point(args):
    """Worker function for main measurement run."""
    params, delta, m_tol, seed = args
    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    # Burn-in
    for _ in range(params.n_burnin):
        lattice.sweep()

    # Sampling with batch tracking
    n_batches = params.n_batches
    batch_size = params.n_sample // n_batches

    batch_geo = []
    batch_neu = []
    batch_both = []
    batch_m = []
    batch_phi = []
    batch_phi_median = []
    batch_phi_q95 = []
    batch_phi_max = []
    total_spin_acc = 0
    total_edge_acc = 0

    for b in range(n_batches):
        geo_count = 0
        neu_count = 0
        both_count = 0
        m_sum = 0.0
        phi_sum = 0.0
        phi_median_sum = 0.0
        phi_q95_sum = 0.0
        phi_max_sum = 0.0

        for _ in range(batch_size):
            spin_acc, edge_acc = lattice.sweep()
            total_spin_acc += spin_acc
            total_edge_acc += edge_acc

            g_geo = lattice.geometric_gate(delta)
            g_neu = lattice.neutrality_gate(m_tol)

            if g_geo:
                geo_count += 1
            if g_neu:
                neu_count += 1
            if g_geo and g_neu:
                both_count += 1

            m_sum += abs(lattice.magnetization())
            phi_sum += lattice.mean_holonomy_magnitude()
            abs_phi = np.abs(lattice._phi_cache)
            phi_median_sum += np.median(abs_phi)
            phi_q95_sum += np.quantile(abs_phi, params.q_level)
            phi_max_sum += np.max(abs_phi)

        batch_geo.append(geo_count / batch_size)
        batch_neu.append(neu_count / batch_size)
        batch_both.append(both_count / batch_size)
        batch_m.append(m_sum / batch_size)
        batch_phi.append(phi_sum / batch_size)
        batch_phi_median.append(phi_median_sum / batch_size)
        batch_phi_q95.append(phi_q95_sum / batch_size)
        batch_phi_max.append(phi_max_sum / batch_size)

    # Compute means and stderrs
    p_geo = np.mean(batch_geo)
    p_neu = np.mean(batch_neu)
    p_both = np.mean(batch_both)

    stderr_geo = np.std(batch_geo) / np.sqrt(n_batches)
    stderr_neu = np.std(batch_neu) / np.sqrt(n_batches)
    stderr_both = np.std(batch_both) / np.sqrt(n_batches)

    # Compute rho per batch and then average
    rho_batches = []
    for i in range(n_batches):
        prod = batch_geo[i] * batch_neu[i]
        if prod > 1e-10:
            rho_batches.append(batch_both[i] / prod)

    if len(rho_batches) >= 3:
        rho = np.mean(rho_batches)
        stderr_rho = np.std(rho_batches) / np.sqrt(len(rho_batches))
    elif p_geo * p_neu > 1e-10:
        rho = p_both / (p_geo * p_neu)
        stderr_rho = np.nan
    else:
        rho = np.nan
        stderr_rho = np.nan

    accept_spin = total_spin_acc / (params.n_sample * params.N)
    accept_theta = total_edge_acc / (params.n_sample * 2 * params.N)
    mean_m = np.mean(batch_m)
    mean_phi = np.mean(batch_phi)
    mean_phi_median = np.mean(batch_phi_median)
    mean_phi_q95 = np.mean(batch_phi_q95)
    mean_phi_max = np.mean(batch_phi_max)

    return {
        'p_geo': p_geo,
        'p_neu': p_neu,
        'p_both': p_both,
        'rho': rho,
        'stderr_geo': stderr_geo,
        'stderr_neu': stderr_neu,
        'stderr_both': stderr_both,
        'stderr_rho': stderr_rho,
        'accept_spin': accept_spin,
        'accept_theta': accept_theta,
        'mean_m': mean_m,
        'mean_phi': mean_phi,
        'mean_phi_median': mean_phi_median,
        'mean_phi_q95': mean_phi_q95,
        'mean_phi_max': mean_phi_max,
    }


def run_epsilon_job(eps: float, feasible_kappas: List[float], fixed_delta: float = None,
                    params: SimulationParams = None, base_seed: int = 0, idx: int = 0,
                    kappa_stats: Dict[float, Dict[str, float]] = None,
                    cal_burnin: int = 200, cal_sample: int = 600,
                    log_dir: str = "logs") -> Dict[str, float]:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eps_{idx}_{eps:.2f}.log")

    seed_cal_geo = base_seed + 1000 * idx + 11
    seed_cal_neu = base_seed + 1000 * idx + 23
    seed_final = base_seed + 1000 * idx + 37

    with open(log_path, "w", encoding="utf-8") as log_file, \
            contextlib.redirect_stdout(log_file), \
            contextlib.redirect_stderr(log_file):
        print(f"ε job idx={idx} target={eps:.4f}")
        print(f"Seeds: cal_geo={seed_cal_geo}, cal_neu={seed_cal_neu}, final={seed_final}")
        print(f"Feasible κ list: {feasible_kappas}")

        chosen = None
        for kappa in feasible_kappas:
            params_k = replace(params, kappa=kappa)
            stats = kappa_stats[kappa]
            if fixed_delta is None:
                delta, ok_delta = calibrate_delta(params_k, target_p=eps,
                                                  median_q=stats['median_q_level'],
                                                  n_burnin=cal_burnin, n_sample=cal_sample,
                                                  seed_base=seed_cal_geo)
            else:
                delta = fixed_delta
                ok_delta = True

            if not ok_delta:
                continue

            m_tol, ok_m = calibrate_m_tol(params_k, delta, target_p=eps,
                                          n_burnin=cal_burnin, n_sample=cal_sample,
                                          seed_base=seed_cal_neu)
            if ok_m:
                chosen = (kappa, delta, m_tol)
                break

        if chosen is None:
            raise RuntimeError(f"Failed to find feasible κ for ε={eps:.2f}.")

        kappa, delta, m_tol = chosen
        print(f"Selected κ={kappa:.1f}, δ={delta:.4f}, m_tol={m_tol:.4f}")
        params_final = replace(params, kappa=kappa)
        measurement = run_main_sweep_point((params_final, delta, m_tol, seed_final))

    return {
        'epsilon_target': eps,
        'kappa': kappa,
        'delta': delta,
        'm_tol': m_tol,
        'p_geo': measurement['p_geo'],
        'p_neu': measurement['p_neu'],
        'p_both': measurement['p_both'],
        'rho': measurement['rho'],
        'stderr_geo': measurement['stderr_geo'],
        'stderr_neu': measurement['stderr_neu'],
        'stderr_both': measurement['stderr_both'],
        'stderr_rho': measurement['stderr_rho'],
        'accept_spin': measurement['accept_spin'],
        'accept_theta': measurement['accept_theta'],
        'mean_abs_phi': measurement['mean_phi'],
        'mean_m': measurement['mean_m'],
        'q95_median': kappa_stats[kappa]['median_q_level'],
        'q99_median': kappa_stats[kappa]['median_q99'],
        'qmax_median': kappa_stats[kappa]['median_max'],
        'J': params.J,
        'T': params.T,
        'g': params.g,
        'mu': params.mu,
        'gamma': params.gamma,
        'sigma': params.sigma,
        'seed_cal_geo': seed_cal_geo,
        'seed_cal_neu': seed_cal_neu,
        'seed_final': seed_final,
        'log_path': log_path,
    }


def run_epsilon_job_task(task, job_kwargs):
    idx, eps = task
    return run_epsilon_job(eps, idx=idx, **job_kwargs)


def run_kappa_feasibility_task(args):
    """Worker for parallel κ feasibility scan."""
    kappa, params, cal_burnin, cal_sample, seed_base = args
    params_k = replace(params, kappa=kappa)
    stats = probe_run_stats(params_k, n_burnin=200, n_sample=400, seed=200 + int(kappa))

    delta_try, ok = calibrate_delta(params_k, target_p=0.2,
                                    median_q=stats['median_q_level'],
                                    n_burnin=cal_burnin, n_sample=cal_sample,
                                    seed_base=seed_base + int(kappa))
    return {
        'kappa': kappa,
        'stats': stats,
        'delta': delta_try,
        'feasible': ok
    }


def create_figures(results: List[Dict], output_dir: str = "."):
    """Create the 4 required figures."""

    epsilon_targets = np.array([r['epsilon_target'] for r in results])
    p_geo_arr = np.array([r['p_geo'] for r in results])
    p_neu_arr = np.array([r['p_neu'] for r in results])
    p_both_arr = np.array([r['p_both'] for r in results])
    rho_arr = np.array([r['rho'] for r in results])
    stderr_geo_arr = np.array([r['stderr_geo'] for r in results])
    stderr_neu_arr = np.array([r['stderr_neu'] for r in results])
    stderr_both_arr = np.array([r['stderr_both'] for r in results])
    stderr_rho_arr = np.array([r['stderr_rho'] for r in results])
    kappa_arr = np.array([r['kappa'] for r in results])
    q95_median_arr = np.array([r['q95_median'] for r in results])
    q99_median_arr = np.array([r['q99_median'] for r in results])
    qmax_median_arr = np.array([r['qmax_median'] for r in results])

    # 1.png: p_both vs p_geo*p_neu
    plt.figure(figsize=(8, 8))
    products = p_geo_arr * p_neu_arr
    valid = np.isfinite(products) & np.isfinite(p_both_arr) & (products > 0)

    if np.any(valid):
        plt.scatter(products[valid], p_both_arr[valid], c=epsilon_targets[valid],
                    cmap='viridis', s=140, edgecolors='black', linewidths=1.2)
        plt.colorbar(label='ε target')
        max_val = max(np.max(products[valid]), np.max(p_both_arr[valid])) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
    else:
        max_val = 1.0
        plt.text(0.5, 0.5, 'No valid points', transform=plt.gca().transAxes,
                 ha='center', va='center')

    plt.xlabel('p_geo × p_neu', fontsize=14)
    plt.ylabel('p_both', fontsize=14)
    plt.title('Matched-permeability: p_both vs p_geo·p_neu', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/1.png")

    # 2.png: rho vs epsilon target
    plt.figure(figsize=(10, 6))
    valid_rho = np.isfinite(rho_arr)
    err_plot = np.where(np.isfinite(stderr_rho_arr), stderr_rho_arr, 0)
    plt.errorbar(epsilon_targets[valid_rho], rho_arr[valid_rho],
                 yerr=err_plot[valid_rho], fmt='ko-', markersize=8,
                 linewidth=2, capsize=4, capthick=1.5)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='ρ=1')
    plt.xlabel('ε target', fontsize=14)
    plt.ylabel('ρ = p_both / (p_geo × p_neu)', fontsize=14)
    plt.title('Gate correlation ratio vs ε', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/2.png")

    # 3.png: p_both vs epsilon^2 (epsilon_hat)
    epsilon_hat = (p_geo_arr + p_neu_arr) / 2
    eps_sq = epsilon_hat**2
    plt.figure(figsize=(10, 7))
    plt.errorbar(eps_sq, p_both_arr, yerr=stderr_both_arr, fmt='o',
                 markersize=8, capsize=4, label='data')

    fit_mask = np.isfinite(eps_sq) & np.isfinite(p_both_arr)
    if np.sum(fit_mask) >= 2:
        coeff = np.polyfit(eps_sq[fit_mask], p_both_arr[fit_mask], 1)
        fit_line = np.poly1d(coeff)
        x_fit = np.linspace(0, np.max(eps_sq[fit_mask]) * 1.05, 100)
        plt.plot(x_fit, fit_line(x_fit), 'k--', linewidth=2,
                 label=f'fit: y={coeff[0]:.2f}x+{coeff[1]:.3f}')

    plt.xlabel('ε̂² with ε̂=(p_geo+p_neu)/2', fontsize=14)
    plt.ylabel('p_both', fontsize=14)
    plt.title('Matched-square test', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/3.png")

    # 4.png: chosen κ vs epsilon target with q95 stats inset
    plt.figure(figsize=(10, 7))
    plt.plot(epsilon_targets, kappa_arr, 'o-', linewidth=2, label='chosen κ')
    plt.xlabel('ε target', fontsize=14)
    plt.ylabel('κ', fontsize=14)
    plt.title('Regime selection (κ vs ε)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    inset = plt.axes([0.55, 0.2, 0.35, 0.3])
    inset.plot(epsilon_targets, q95_median_arr, 'o-', label='median q95')
    inset.plot(epsilon_targets, q99_median_arr, 's--', label='median q99')
    inset.plot(epsilon_targets, qmax_median_arr, 'd-.', label='median max')
    inset.set_xlabel('ε', fontsize=9)
    inset.set_ylabel('|Phi| stats', fontsize=9)
    inset.grid(True, alpha=0.3)
    inset.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/4.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/4.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean exponent test for the two-gate model.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test with small L and short sampling.")
    parser.add_argument("--smoke", action="store_true",
                        help="Alias for --smoke-test.")
    args = parser.parse_args()
    smoke_mode = args.smoke_test or args.smoke

    print("=" * 70)
    print("CLEAN EXPONENT TEST FOR TWO-GATE MODEL")
    print("=" * 70)

    start_time = time.time()

    params = SimulationParams(
        L=24, T=1.0, J=0.6, h=0.0,
        kappa=50.0, g=0.5,
        mu=1.0, gamma=0.5, sigma=1.0,
        dtheta=0.6,
        n_burnin=600, n_sample=2400, n_batches=30,
        q_level=0.95
    )

    epsilon_targets = [0.10, 0.15, 0.20, 0.30, 0.40]
    kappa_candidates = [20, 30, 40, 50, 65, 80]
    cal_burnin = 200
    cal_sample = 600
    base_seed = 4000

    if smoke_mode:
        params = replace(params, L=8, n_burnin=120, n_sample=300, n_batches=10)
        epsilon_targets = [0.2, 0.4]
        kappa_candidates = [20, 30, 40, 50]
        cal_burnin = 50
        cal_sample = 100

    print(f"\nBase parameters:")
    print(f"  L={params.L}, T={params.T}, J={params.J}, g={params.g}")
    print(f"  mu={params.mu}, gamma={params.gamma}, sigma={params.sigma}")
    print(f"  q_level={params.q_level:.2f}")

    # ========== PHASE 1: ORDERED-PHASE CHECK ==========
    print("\n" + "=" * 70)
    print("PHASE 1: ORDERED-PHASE CHECK")
    print("=" * 70)

    probe_params = replace(params, kappa=50.0, g=0.5, h=0.0)
    probe_stats = probe_run_stats(probe_params, n_burnin=200, n_sample=400, seed=101)
    print(f"  Probe mean|m|={probe_stats['mean_abs_m']:.3f} at J={probe_params.J}, T={probe_params.T}")

    if probe_stats['mean_abs_m'] > 0.5:
        probe_params = replace(probe_params, T=max(probe_params.T, 1.6))
        probe_stats = probe_run_stats(probe_params, n_burnin=200, n_sample=400, seed=102)
        print(f"  Raised T: mean|m|={probe_stats['mean_abs_m']:.3f} at J={probe_params.J}, T={probe_params.T}")

    if probe_stats['mean_abs_m'] > 0.5:
        probe_params = replace(probe_params, J=0.4)
        probe_stats = probe_run_stats(probe_params, n_burnin=200, n_sample=400, seed=103)
        print(f"  Lowered J: mean|m|={probe_stats['mean_abs_m']:.3f} at J={probe_params.J}, T={probe_params.T}")

    if probe_stats['mean_abs_m'] > 0.3:
        print("  Warning: mean|m| remains > 0.3; neutrality gating may be tight.")

    params = replace(params, J=probe_params.J, T=probe_params.T)
    print(f"  Selected (J,T)=({params.J},{params.T}) with mean|m|={probe_stats['mean_abs_m']:.3f}")

    # ========== PHASE 2: KAPPA FEASIBILITY SCAN ==========
    print("\n" + "=" * 70)
    print("PHASE 2: KAPPA FEASIBILITY SCAN (PARALLEL)")
    print("=" * 70)

    kappa_stats = {}
    feasible_kappas = []

    cpu_total = os.cpu_count() or 1
    phase2_processes = min(len(kappa_candidates), max(1, cpu_total - 1))
    print(f"  Scanning {len(kappa_candidates)} κ values with {phase2_processes} processes...")

    phase2_tasks = [(kappa, params, cal_burnin, cal_sample, 400) for kappa in kappa_candidates]

    with multiprocessing.Pool(processes=phase2_processes) as pool:
        phase2_results = pool.map(run_kappa_feasibility_task, phase2_tasks)

    for res in sorted(phase2_results, key=lambda x: x['kappa']):
        kappa = res['kappa']
        stats = res['stats']
        kappa_stats[kappa] = stats
        print(f"  κ={kappa:>5.1f}: median(|Phi|)={stats['median_abs_phi']:.4f}, "
              f"q95={stats['median_q_level']:.4f}, q99={stats['median_q99']:.4f}, "
              f"max={stats['median_max']:.4f}")
        if res['feasible']:
            feasible_kappas.append(kappa)
            print(f"    Feasible κ={kappa:.1f} with δ≈{res['delta']:.4f}")
        else:
            print(f"    Infeasible κ={kappa:.1f}")

    if not feasible_kappas and params.q_level > 0.90:
        print("\n  No feasible κ found. Relaxing q_level to 0.90 and rescanning (parallel)...")
        params = replace(params, q_level=0.90)
        feasible_kappas = []

        phase2_tasks_retry = [(kappa, params, cal_burnin, cal_sample, 460) for kappa in kappa_candidates]
        with multiprocessing.Pool(processes=phase2_processes) as pool:
            phase2_results_retry = pool.map(run_kappa_feasibility_task, phase2_tasks_retry)

        for res in sorted(phase2_results_retry, key=lambda x: x['kappa']):
            kappa = res['kappa']
            stats = res['stats']
            kappa_stats[kappa] = stats
            print(f"  κ={kappa:>5.1f}: median(|Phi|)={stats['median_abs_phi']:.4f}, "
                  f"q90={stats['median_q_level']:.4f}, q99={stats['median_q99']:.4f}, "
                  f"max={stats['median_max']:.4f}")
            if res['feasible']:
                feasible_kappas.append(kappa)
                print(f"    Feasible κ={kappa:.1f} with δ≈{res['delta']:.4f}")

    if not feasible_kappas:
        raise RuntimeError("No feasible κ found; cannot proceed with matched-permeability test.")

    print(f"\n  Feasible κ values: {feasible_kappas}")

    # ========== PHASE 3: MATCHED-PERMEABILITY TEST ==========
    print("\n" + "=" * 70)
    print("PHASE 3: MATCHED-PERMEABILITY TEST")
    print("=" * 70)

    results = []
    cpu_total = os.cpu_count() or 1
    if smoke_mode:
        processes = min(2, len(epsilon_targets))
    else:
        processes = min(len(epsilon_targets), max(1, cpu_total - 1))

    print(f"  Launching ε jobs with {processes} process(es).")
    tasks = [(idx, epsilon) for idx, epsilon in enumerate(epsilon_targets)]
    job_kwargs = {
        'feasible_kappas': feasible_kappas,
        'fixed_delta': None,
        'params': params,
        'base_seed': base_seed,
        'kappa_stats': kappa_stats,
        'cal_burnin': cal_burnin,
        'cal_sample': cal_sample,
        'log_dir': "logs",
    }
    worker = functools.partial(run_epsilon_job_task, job_kwargs=job_kwargs)

    with multiprocessing.Pool(processes=processes) as pool:
        for result in pool.imap_unordered(worker, tasks):
            results.append(result)
            print(f"  Completed ε={result['epsilon_target']:.2f} "
                  f"(κ={result['kappa']:.1f}, log={result['log_path']})")

    # ========== PHASE 4: SAVING OUTPUT ==========
    print("\n" + "=" * 70)
    print("PHASE 4: SAVING OUTPUT")
    print("=" * 70)

    results = sorted(results, key=lambda r: r['epsilon_target'])

    csv_filename = "robust_matched_results.csv"
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = [
            'epsilon_target', 'kappa', 'delta', 'm_tol', 'p_geo', 'p_neu', 'p_both', 'rho',
            'stderr_p_geo', 'stderr_p_neu', 'stderr_p_both', 'stderr_rho',
            'J', 'T', 'g', 'mu', 'gamma', 'sigma',
            'mean_abs_phi', 'mean_m', 'accept_spin', 'accept_theta'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'epsilon_target': r['epsilon_target'],
                'kappa': r['kappa'],
                'delta': r['delta'],
                'm_tol': r['m_tol'],
                'p_geo': r['p_geo'],
                'p_neu': r['p_neu'],
                'p_both': r['p_both'],
                'rho': r['rho'],
                'stderr_p_geo': r['stderr_geo'],
                'stderr_p_neu': r['stderr_neu'],
                'stderr_p_both': r['stderr_both'],
                'stderr_rho': r['stderr_rho'],
                'J': r['J'],
                'T': r['T'],
                'g': r['g'],
                'mu': r['mu'],
                'gamma': r['gamma'],
                'sigma': r['sigma'],
                'mean_abs_phi': r['mean_abs_phi'],
                'mean_m': r['mean_m'],
                'accept_spin': r['accept_spin'],
                'accept_theta': r['accept_theta'],
            })
    print(f"  Saved: {csv_filename}")

    create_figures(results, output_dir=".")

    # ========== FINAL REPORT ==========
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    rho_arr = np.array([r['rho'] for r in results])
    rho_valid = rho_arr[np.isfinite(rho_arr)]
    if len(rho_valid) > 0:
        mean_rho = np.mean(rho_valid)
        stderr_rho = np.std(rho_valid) / np.sqrt(len(rho_valid))
        print(f"\n  Mean ρ across ε: {mean_rho:.3f} ± {stderr_rho:.3f}")

    print(f"\n  Data points: {len(results)}")
    print(f"\n  Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 70)

    if smoke_mode:
        print("\nSmoke test summary:")
        for res in results:
            rho_str = f"{res['rho']:.3f}" if np.isfinite(res['rho']) else "nan"
            print(f"  ε={res['epsilon_target']:.2f} κ={res['kappa']:.1f}: "
                  f"p_geo={res['p_geo']:.4f}, p_neu={res['p_neu']:.4f}, "
                  f"p_both={res['p_both']:.4f}, rho={rho_str}")


if __name__ == "__main__":
    main()
