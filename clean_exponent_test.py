#!/usr/bin/env python3
"""
Clean Exponent Test for Two-Gate Model

Tests multiplicativity (c ≈ a + b) and gate independence (ρ ≈ 1) in a regime
where BOTH gates have real dynamic range (0.05 ≤ p ≤ 0.5).

Calibration ensures p_geo(t=1) ≈ 0.2 and p_neu(t=1) ≈ 0.2.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import csv
import time
from multiprocessing import Pool, cpu_count


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
        q = np.quantile(abs_phi, 0.95)
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


def calibrate_delta(params: SimulationParams, m_tol: float, target_p: float = 0.2,
                    tol: float = 0.03, max_iter: int = 15,
                    n_burnin: int = 200, n_sample: int = 600) -> float:
    """Bisection search to find δ0 such that p_geo(δ0) ≈ target_p."""
    print(f"  Calibrating δ0 for p_geo ≈ {target_p}...")

    # Start with initial bracket
    delta_low, delta_high = 0.05, 2.0

    # Check bounds
    p_low, _ = run_calibration_point(params, delta_low, m_tol,
                                     n_burnin=n_burnin, n_sample=n_sample, seed=42)
    p_high, _ = run_calibration_point(params, delta_high, m_tol,
                                      n_burnin=n_burnin, n_sample=n_sample, seed=42)
    print(f"    Initial: δ={delta_low:.3f} -> p_geo={p_low:.3f}, δ={delta_high:.3f} -> p_geo={p_high:.3f}")

    if p_high < target_p:
        print(f"    Warning: p_geo({delta_high}) = {p_high:.3f} < target. Increasing delta_high.")
        delta_high = 4.0
        p_high, _ = run_calibration_point(params, delta_high, m_tol,
                                          n_burnin=n_burnin, n_sample=n_sample, seed=42)

    for it in range(max_iter):
        delta_mid = (delta_low + delta_high) / 2
        p_mid, _ = run_calibration_point(params, delta_mid, m_tol,
                                         n_burnin=n_burnin, n_sample=n_sample, seed=42 + it)
        print(f"    Iter {it+1}: δ={delta_mid:.4f} -> p_geo={p_mid:.3f}")

        if abs(p_mid - target_p) < tol:
            print(f"  Converged: δ0 = {delta_mid:.4f} (p_geo = {p_mid:.3f})")
            return delta_mid

        if p_mid < target_p:
            delta_low = delta_mid
        else:
            delta_high = delta_mid

    delta_final = (delta_low + delta_high) / 2
    p_final, _ = run_calibration_point(params, delta_final, m_tol,
                                       n_burnin=n_burnin, n_sample=n_sample, seed=99)
    print(f"  Final: δ0 = {delta_final:.4f} (p_geo = {p_final:.3f})")
    return delta_final


def calibrate_m_tol(params: SimulationParams, delta: float, target_p: float = 0.2,
                    tol: float = 0.03, max_iter: int = 15,
                    n_burnin: int = 200, n_sample: int = 600) -> float:
    """Bisection search to find m0 such that p_neu(m0) ≈ target_p."""
    print(f"  Calibrating m0 for p_neu ≈ {target_p}...")

    m_low, m_high = 0.01, 0.5

    _, p_low = run_calibration_point(params, delta, m_low,
                                     n_burnin=n_burnin, n_sample=n_sample, seed=42)
    _, p_high = run_calibration_point(params, delta, m_high,
                                      n_burnin=n_burnin, n_sample=n_sample, seed=42)
    print(f"    Initial: m={m_low:.3f} -> p_neu={p_low:.3f}, m={m_high:.3f} -> p_neu={p_high:.3f}")

    if p_high < target_p:
        print(f"    Warning: p_neu({m_high}) = {p_high:.3f} < target. Increasing m_high.")
        m_high = 1.0
        _, p_high = run_calibration_point(params, delta, m_high,
                                          n_burnin=n_burnin, n_sample=n_sample, seed=42)

    for it in range(max_iter):
        m_mid = (m_low + m_high) / 2
        _, p_mid = run_calibration_point(params, delta, m_mid,
                                         n_burnin=n_burnin, n_sample=n_sample, seed=42 + it)
        print(f"    Iter {it+1}: m={m_mid:.4f} -> p_neu={p_mid:.3f}")

        if abs(p_mid - target_p) < tol:
            print(f"  Converged: m0 = {m_mid:.4f} (p_neu = {p_mid:.3f})")
            return m_mid

        if p_mid < target_p:
            m_low = m_mid
        else:
            m_high = m_mid

    m_final = (m_low + m_high) / 2
    _, p_final = run_calibration_point(params, delta, m_final,
                                       n_burnin=n_burnin, n_sample=n_sample, seed=99)
    print(f"  Final: m0 = {m_final:.4f} (p_neu = {p_final:.3f})")
    return m_final


def run_main_sweep_point(args):
    """Worker function for parallel sweep."""
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
            phi_q95_sum += np.quantile(abs_phi, 0.95)
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


def fit_log_slope(t_vals: np.ndarray, p_vals: np.ndarray, p_errs: np.ndarray = None) -> Tuple[float, float]:
    """
    Fit log(p) = a * log(t) + b.
    Returns (slope, slope_stderr).
    """
    # Filter out zeros and nans
    valid = (p_vals > 0) & np.isfinite(p_vals) & (t_vals > 0)
    t_valid = t_vals[valid]
    p_valid = p_vals[valid]

    if len(t_valid) < 3:
        return np.nan, np.nan

    log_t = np.log(t_valid)
    log_p = np.log(p_valid)

    # Simple linear regression
    n = len(log_t)
    sum_x = np.sum(log_t)
    sum_y = np.sum(log_p)
    sum_xx = np.sum(log_t**2)
    sum_xy = np.sum(log_t * log_p)

    denom = n * sum_xx - sum_x**2
    if abs(denom) < 1e-12:
        return np.nan, np.nan

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # Residual variance for error estimate
    residuals = log_p - (slope * log_t + intercept)
    s2 = np.sum(residuals**2) / (n - 2) if n > 2 else np.nan
    slope_stderr = np.sqrt(s2 * n / denom) if np.isfinite(s2) else np.nan

    return slope, slope_stderr


def create_figures(results: List[Dict], t_values: List[float], delta0: float, m0: float,
                   slopes: Dict, output_dir: str = "."):
    """Create the 4 required figures."""

    t_arr = np.array(t_values)
    p_geo_arr = np.array([r['p_geo'] for r in results])
    p_neu_arr = np.array([r['p_neu'] for r in results])
    p_both_arr = np.array([r['p_both'] for r in results])
    rho_arr = np.array([r['rho'] for r in results])
    stderr_rho_arr = np.array([r['stderr_rho'] for r in results])

    # Filter for included points (not saturated)
    included = (p_geo_arr > 0.02) & (p_geo_arr < 0.6) & \
               (p_neu_arr > 0.02) & (p_neu_arr < 0.6) & \
               (p_both_arr > 0.02) & (p_both_arr < 0.6)

    t_inc = t_arr[included]
    p_geo_inc = p_geo_arr[included]
    p_neu_inc = p_neu_arr[included]
    p_both_inc = p_both_arr[included]

    # 1.png: Log-log plot
    plt.figure(figsize=(10, 7))

    # Plot all points
    plt.loglog(t_arr, p_geo_arr, 'bo', markersize=8, label='p_geo (all)', alpha=0.5)
    plt.loglog(t_arr, p_neu_arr, 'gs', markersize=8, label='p_neu (all)', alpha=0.5)
    plt.loglog(t_arr, p_both_arr, 'r^', markersize=8, label='p_both (all)', alpha=0.5)

    # Highlight included points
    plt.loglog(t_inc, p_geo_inc, 'bo', markersize=12, markeredgecolor='black', markeredgewidth=2)
    plt.loglog(t_inc, p_neu_inc, 'gs', markersize=12, markeredgecolor='black', markeredgewidth=2)
    plt.loglog(t_inc, p_both_inc, 'r^', markersize=12, markeredgecolor='black', markeredgewidth=2)

    # Fitted lines (for included points only)
    if len(t_inc) >= 3:
        t_fit = np.linspace(t_inc.min(), t_inc.max(), 100)

        a, a_err = slopes['a'], slopes['a_err']
        b, b_err = slopes['b'], slopes['b_err']
        c, c_err = slopes['c'], slopes['c_err']

        if np.isfinite(a):
            # Fit intercept from first included point
            log_p_geo_0 = np.log(p_geo_inc[0]) - a * np.log(t_inc[0])
            p_geo_fit = np.exp(a * np.log(t_fit) + log_p_geo_0)
            plt.loglog(t_fit, p_geo_fit, 'b--', linewidth=2, label=f'a={a:.2f}±{a_err:.2f}')

        if np.isfinite(b):
            log_p_neu_0 = np.log(p_neu_inc[0]) - b * np.log(t_inc[0])
            p_neu_fit = np.exp(b * np.log(t_fit) + log_p_neu_0)
            plt.loglog(t_fit, p_neu_fit, 'g--', linewidth=2, label=f'b={b:.2f}±{b_err:.2f}')

        if np.isfinite(c):
            log_p_both_0 = np.log(p_both_inc[0]) - c * np.log(t_inc[0])
            p_both_fit = np.exp(c * np.log(t_fit) + log_p_both_0)
            plt.loglog(t_fit, p_both_fit, 'r--', linewidth=2, label=f'c={c:.2f}±{c_err:.2f}')

    plt.xlabel('t (threshold scale)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title(f'Log-Log Gate Probabilities (δ₀={delta0:.3f}, m₀={m0:.3f})', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/1.png")

    # 2.png: rho(t) vs t
    plt.figure(figsize=(10, 6))

    valid_rho = np.isfinite(rho_arr)
    valid_err = np.isfinite(stderr_rho_arr)

    t_plot = t_arr[valid_rho]
    rho_plot = rho_arr[valid_rho]
    err_plot = np.where(valid_err[valid_rho], stderr_rho_arr[valid_rho], 0)

    plt.errorbar(t_plot, rho_plot, yerr=err_plot, fmt='ko-', markersize=10,
                 linewidth=2, capsize=5, capthick=2, label='ρ(t)')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='ρ=1 (independence)')
    plt.fill_between([0, 1.1], 0.95, 1.05, alpha=0.2, color='green', label='±5% band')

    plt.xlabel('t (threshold scale)', fontsize=14)
    plt.ylabel('ρ = p_both / (p_geo × p_neu)', fontsize=14)
    plt.title('Gate Correlation Ratio vs Threshold Scale', fontsize=14)
    plt.xlim(0, 1.05)
    plt.ylim(0.5, 1.5)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/2.png")

    # 3.png: Scatter p_both vs p_geo*p_neu
    plt.figure(figsize=(8, 8))

    products = p_geo_arr * p_neu_arr
    valid = np.isfinite(products) & np.isfinite(p_both_arr) & (products > 0)

    if np.any(valid):
        plt.scatter(products[valid], p_both_arr[valid], c=t_arr[valid], cmap='viridis',
                    s=150, edgecolors='black', linewidths=1.5)
        plt.colorbar(label='t')

        max_val = max(np.max(products[valid]), np.max(p_both_arr[valid])) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (independence)')
    else:
        max_val = 1.0
        plt.text(0.5, 0.5, 'No valid points for scatter',
                 transform=plt.gca().transAxes, ha='center', va='center')

    plt.xlabel('p_geo × p_neu', fontsize=14)
    plt.ylabel('p_both', fontsize=14)
    plt.title('Gate Independence Test (color = t)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/3.png")

    # 4.png: Summary panel
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    a, a_err = slopes['a'], slopes['a_err']
    b, b_err = slopes['b'], slopes['b_err']
    c, c_err = slopes['c'], slopes['c_err']

    if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
        diff = c - (a + b)
        diff_err = np.sqrt(a_err**2 + b_err**2 + c_err**2)
    else:
        diff = np.nan
        diff_err = np.nan

    # Mean rho for included points
    rho_inc = rho_arr[included]
    rho_inc = rho_inc[np.isfinite(rho_inc)]
    if len(rho_inc) > 0:
        mean_rho = np.mean(rho_inc)
        stderr_mean_rho = np.std(rho_inc) / np.sqrt(len(rho_inc))
    else:
        mean_rho = np.nan
        stderr_mean_rho = np.nan

    n_included = np.sum(included)

    summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                   CLEAN EXPONENT TEST SUMMARY                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Calibrated Parameters:                                          ║
║    δ₀ = {delta0:.4f}                                              ║
║    m₀ = {m0:.4f}                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Fitted Exponents (log p vs log t):                              ║
║    a (p_geo slope)  = {a:+.3f} ± {a_err:.3f}                       ║
║    b (p_neu slope)  = {b:+.3f} ± {b_err:.3f}                       ║
║    c (p_both slope) = {c:+.3f} ± {c_err:.3f}                       ║
║                                                                  ║
║    c - (a + b) = {diff:+.3f} ± {diff_err:.3f}                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Gate Independence:                                              ║
║    Mean ρ = {mean_rho:.3f} ± {stderr_mean_rho:.3f}                 ║
║    (ρ = 1 indicates independence)                                ║
╠══════════════════════════════════════════════════════════════════╣
║  Data Quality:                                                   ║
║    Total points: {len(t_values)}                                  ║
║    Included in fit: {n_included} (0.02 < p < 0.60 for all gates)  ║
╠══════════════════════════════════════════════════════════════════╣
║  INTERPRETATION:                                                 ║
║    Multiplicativity: {"SUPPORTED" if np.isfinite(diff) and abs(diff) < 2*diff_err else "UNCLEAR"}  (c ≈ a + b?)           ║
║    Independence:     {"SUPPORTED" if np.isfinite(mean_rho) and abs(mean_rho - 1) < 0.1 else "UNCLEAR"}  (ρ ≈ 1?)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/4.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/4.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean exponent test for the two-gate model.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test with small L and short sampling.")
    args = parser.parse_args()

    print("=" * 70)
    print("CLEAN EXPONENT TEST FOR TWO-GATE MODEL")
    print("=" * 70)

    start_time = time.time()

    # Base parameters
    # NOTE: For 2D Ising, T_c = 2J/ln(1+sqrt(2)) ≈ 2.27*J
    # We need T > T_c for magnetization fluctuations around m=0
    # With J=0.4, T_c ≈ 0.91, so T=2.0 gives T/T_c ≈ 2.2 (paramagnetic)
    params = SimulationParams(
        L=16, T=2.5, J=0.35, h=0.0,
        kappa=50.0, g=0.5,
        mu=1.0, gamma=0.2, sigma=1.0,
        dtheta=0.6,
        n_burnin=200, n_sample=800, n_batches=10
    )

    t_values = [0.20, 0.35, 0.50, 0.70, 0.85, 1.00]  # 6 points for quick test

    if args.smoke_test:
        params.L = 8
        params.n_burnin = 50
        params.n_sample = 100
        params.n_batches = 10
        t_values = [0.50, 1.00]

    print(f"\nBase parameters:")
    print(f"  L={params.L}, T={params.T}, J={params.J}, kappa={params.kappa}, g={params.g}")
    print(f"  mu={params.mu}, gamma={params.gamma}, sigma={params.sigma}")

    # Target probability for calibration
    # Use 0.2 for dynamic range in log space during threshold shrinking
    target_p = 0.2

    # ========== CALIBRATION ==========
    print("\n" + "=" * 70)
    print("PHASE 1: CALIBRATION")
    print("=" * 70)

    # Initial guesses
    delta0_init = 0.4
    m0_init = 0.10

    # Calibrate δ0 (keeping m0 fixed at initial)
    delta0 = calibrate_delta(params, m0_init, target_p=target_p)

    # Calibrate m0 (keeping δ0 fixed at calibrated value)
    m0 = calibrate_m_tol(params, delta0, target_p=target_p)

    # Verify calibration
    print("\n  Verifying calibration at t=1...")
    p_geo_cal, p_neu_cal = run_calibration_point(params, delta0, m0, n_burnin=200, n_sample=600, seed=999)
    print(f"  Verified: p_geo = {p_geo_cal:.3f}, p_neu = {p_neu_cal:.3f}")

    # Check if calibration is acceptable
    if p_geo_cal < 0.1 or p_neu_cal < 0.1:
        print("\n  WARNING: Probabilities very low, results may be unreliable...")

    print(f"\n  FINAL CALIBRATION: δ0 = {delta0:.4f}, m0 = {m0:.4f}")

    # ========== MAIN SWEEP ==========
    print("\n" + "=" * 70)
    print("PHASE 2: MAIN SWEEP")
    print("=" * 70)

    print(f"\n  Running {len(t_values)} t-values with {params.n_sample} samples each...")
    print(f"  (burn-in: {params.n_burnin}, batches: {params.n_batches})")

    # Prepare parallel jobs
    jobs = []
    for i, t in enumerate(t_values):
        delta_t = delta0 * t
        m_tol_t = m0 * t
        jobs.append((params, delta_t, m_tol_t, 1000 + i))

    # Run in parallel
    n_workers = min(len(t_values), cpu_count())
    print(f"  Using {n_workers} parallel workers...")

    with Pool(n_workers) as pool:
        results = pool.map(run_main_sweep_point, jobs)

    # Add t, delta, m_tol to results
    for i, (t, result) in enumerate(zip(t_values, results)):
        result['t'] = t
        result['delta'] = delta0 * t
        result['m_tol'] = m0 * t

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYSIS")
    print("=" * 70)

    # Print table
    print("\n  Results table:")
    print("  " + "-" * 90)
    print(f"  {'t':>6} | {'δ':>6} | {'m_tol':>6} | {'p_geo':>7} | {'p_neu':>7} | {'p_both':>7} | {'ρ':>7} | {'included':>8}")
    print("  " + "-" * 90)

    t_arr = np.array(t_values)
    p_geo_arr = np.array([r['p_geo'] for r in results])
    p_neu_arr = np.array([r['p_neu'] for r in results])
    p_both_arr = np.array([r['p_both'] for r in results])

    for r in results:
        inc = "yes" if (0.02 < r['p_geo'] < 0.6 and
                        0.02 < r['p_neu'] < 0.6 and
                        0.02 < r['p_both'] < 0.6) else "no"
        rho_str = f"{r['rho']:.3f}" if np.isfinite(r['rho']) else "nan"
        print(f"  {r['t']:>6.2f} | {r['delta']:>6.4f} | {r['m_tol']:>6.4f} | "
              f"{r['p_geo']:>7.4f} | {r['p_neu']:>7.4f} | {r['p_both']:>7.4f} | "
              f"{rho_str:>7} | {inc:>8}")

    print("  " + "-" * 90)

    # Filter for slope fitting
    included = (p_geo_arr > 0.02) & (p_geo_arr < 0.6) & \
               (p_neu_arr > 0.02) & (p_neu_arr < 0.6) & \
               (p_both_arr > 0.02) & (p_both_arr < 0.6)

    n_included = np.sum(included)
    print(f"\n  Points included in fit: {n_included} / {len(t_values)}")

    if n_included < 7:
        print("  WARNING: Fewer than 7 points included. Results may be unreliable.")

    # Compute slopes
    t_inc = t_arr[included]
    p_geo_inc = p_geo_arr[included]
    p_neu_inc = p_neu_arr[included]
    p_both_inc = p_both_arr[included]

    a, a_err = fit_log_slope(t_inc, p_geo_inc)
    b, b_err = fit_log_slope(t_inc, p_neu_inc)
    c, c_err = fit_log_slope(t_inc, p_both_inc)

    slopes = {'a': a, 'a_err': a_err, 'b': b, 'b_err': b_err, 'c': c, 'c_err': c_err}

    print(f"\n  Fitted exponents:")
    print(f"    a (p_geo slope)  = {a:.3f} ± {a_err:.3f}")
    print(f"    b (p_neu slope)  = {b:.3f} ± {b_err:.3f}")
    print(f"    c (p_both slope) = {c:.3f} ± {c_err:.3f}")

    if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
        diff = c - (a + b)
        diff_err = np.sqrt(a_err**2 + b_err**2 + c_err**2)
        print(f"\n    c - (a + b) = {diff:.3f} ± {diff_err:.3f}")

        if abs(diff) < 2 * diff_err:
            print("    => Consistent with multiplicativity (c ≈ a + b)")
        else:
            print("    => Deviation from multiplicativity")

    # Mean rho
    rho_arr = np.array([r['rho'] for r in results])
    rho_inc = rho_arr[included]
    rho_inc = rho_inc[np.isfinite(rho_inc)]

    if len(rho_inc) > 0:
        mean_rho = np.mean(rho_inc)
        stderr_rho = np.std(rho_inc) / np.sqrt(len(rho_inc))
        print(f"\n  Mean ρ across included points: {mean_rho:.3f} ± {stderr_rho:.3f}")

        if abs(mean_rho - 1.0) < 0.1:
            print("    => Consistent with gate independence (ρ ≈ 1)")
        else:
            print("    => Deviation from independence")

    def log_phi_stats(results_list: List[Dict], t_target: float):
        for res in results_list:
            if np.isclose(res['t'], t_target, atol=1e-8):
                print(f"\n  |Phi| stats at t={t_target:.2f}:")
                print(f"    median = {res['mean_phi_median']:.4f}")
                print(f"    q95    = {res['mean_phi_q95']:.4f}")
                print(f"    max    = {res['mean_phi_max']:.4f}")
                return

    log_phi_stats(results, 1.0)
    log_phi_stats(results, 0.2)

    # ========== OUTPUT FILES ==========
    print("\n" + "=" * 70)
    print("PHASE 4: SAVING OUTPUT")
    print("=" * 70)

    # Save CSV
    csv_filename = "clean_exponent_results.csv"
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = ['t', 'delta', 'm_tol', 'p_geo', 'p_neu', 'p_both', 'rho',
                      'stderr_p_geo', 'stderr_p_neu', 'stderr_p_both', 'stderr_rho',
                      'accept_spin', 'accept_theta', 'mean_abs_phi', 'mean_m']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                't': r['t'],
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
                'accept_spin': r['accept_spin'],
                'accept_theta': r['accept_theta'],
                'mean_abs_phi': r['mean_phi'],
                'mean_m': r['mean_m'],
            })
    print(f"  Saved: {csv_filename}")

    # Create figures
    create_figures(results, t_values, delta0, m0, slopes, output_dir=".")

    # ========== FINAL REPORT ==========
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(f"\n  Calibrated thresholds:")
    print(f"    δ0 = {delta0:.4f}")
    print(f"    m0 = {m0:.4f}")
    print(f"    Achieved at t=1: p_geo = {p_geo_cal:.3f}, p_neu = {p_neu_cal:.3f}")

    print(f"\n  Exponents:")
    print(f"    a = {a:.3f} ± {a_err:.3f}")
    print(f"    b = {b:.3f} ± {b_err:.3f}")
    print(f"    c = {c:.3f} ± {c_err:.3f}")

    if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
        diff = c - (a + b)
        diff_err = np.sqrt(a_err**2 + b_err**2 + c_err**2)
        print(f"    c - (a + b) = {diff:.3f} ± {diff_err:.3f}")

    if len(rho_inc) > 0:
        print(f"\n  Gate independence:")
        print(f"    Mean ρ = {mean_rho:.3f} ± {stderr_rho:.3f}")

    print(f"\n  Data quality:")
    print(f"    Total points: {len(t_values)}")
    print(f"    Included in fit: {n_included}")

    print(f"\n  CONCLUSIONS:")
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
        diff = c - (a + b)
        diff_err = np.sqrt(a_err**2 + b_err**2 + c_err**2)
        if abs(diff) < 2 * diff_err:
            print("    ✓ Multiplicativity SUPPORTED (c ≈ a + b within 2σ)")
        else:
            print("    ✗ Multiplicativity unclear or violated")
    else:
        print("    ✗ Multiplicativity: insufficient data for exponent fits")

    if len(rho_inc) > 0 and abs(mean_rho - 1.0) < 0.1:
        print("    ✓ Gate independence SUPPORTED (ρ ≈ 1)")
    else:
        print("    ✗ Gate independence unclear or violated")

    print(f"\n  Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 70)

    if args.smoke_test:
        print("\nSmoke test summary:")
        for res in results:
            rho_str = f"{res['rho']:.3f}" if np.isfinite(res['rho']) else "nan"
            print(f"  t={res['t']:.2f}: p_geo={res['p_geo']:.4f}, "
                  f"p_neu={res['p_neu']:.4f}, p_both={res['p_both']:.4f}, rho={rho_str}")


if __name__ == "__main__":
    main()
