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
    """Bisection search to find δ such that p_geo(δ) ≈ target_p."""
    print(f"  Calibrating δ for p_geo ≈ {target_p}...")

    # Start with initial bracket
    delta_low, delta_high = 0.05, 1.5

    # Check bounds
    p_low, _ = run_calibration_point(params, delta_low, m_tol,
                                     n_burnin=n_burnin, n_sample=n_sample, seed=42)
    p_high, _ = run_calibration_point(params, delta_high, m_tol,
                                      n_burnin=n_burnin, n_sample=n_sample, seed=42)
    print(f"    Initial: δ={delta_low:.3f} -> p_geo={p_low:.3f}, δ={delta_high:.3f} -> p_geo={p_high:.3f}")

    for it in range(max_iter):
        delta_mid = (delta_low + delta_high) / 2
        p_mid, _ = run_calibration_point(params, delta_mid, m_tol,
                                         n_burnin=n_burnin, n_sample=n_sample, seed=42 + it)
        print(f"    Iter {it+1}: δ={delta_mid:.4f} -> p_geo={p_mid:.3f}")

        if abs(p_mid - target_p) < tol:
            print(f"  Converged: δ = {delta_mid:.4f} (p_geo = {p_mid:.3f})")
            return delta_mid

        if p_mid < target_p:
            delta_low = delta_mid
        else:
            delta_high = delta_mid

    delta_final = (delta_low + delta_high) / 2
    p_final, _ = run_calibration_point(params, delta_final, m_tol,
                                       n_burnin=n_burnin, n_sample=n_sample, seed=99)
    print(f"  Final: δ = {delta_final:.4f} (p_geo = {p_final:.3f})")
    return delta_final


def calibrate_m_tol(params: SimulationParams, delta: float, target_p: float,
                    tol: float = 0.03, max_iter: int = 15,
                    n_burnin: int = 200, n_sample: int = 600) -> float:
    """Bisection search to find m_tol such that p_neu(m_tol) ≈ target_p."""
    print(f"  Calibrating m_tol for p_neu ≈ {target_p:.3f}...")

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
            print(f"  Converged: m_tol = {m_mid:.4f} (p_neu = {p_mid:.3f})")
            return m_mid

        if p_mid < target_p:
            m_low = m_mid
        else:
            m_high = m_mid

    m_final = (m_low + m_high) / 2
    _, p_final = run_calibration_point(params, delta, m_final,
                                       n_burnin=n_burnin, n_sample=n_sample, seed=99)
    print(f"  Final: m_tol = {m_final:.4f} (p_neu = {p_final:.3f})")
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


def create_figures(results: List[Dict], kappa_values: List[float], delta: float, output_dir: str = "."):
    """Create the 4 required figures."""

    kappa_arr = np.array(kappa_values)
    p_geo_arr = np.array([r['p_geo'] for r in results])
    p_neu_arr = np.array([r['p_neu'] for r in results])
    p_both_arr = np.array([r['p_both'] for r in results])
    rho_arr = np.array([r['rho'] for r in results])
    stderr_geo_arr = np.array([r['stderr_geo'] for r in results])
    stderr_neu_arr = np.array([r['stderr_neu'] for r in results])
    stderr_both_arr = np.array([r['stderr_both'] for r in results])
    stderr_rho_arr = np.array([r['stderr_rho'] for r in results])

    saturated = np.array([r['saturated'] for r in results], dtype=bool)

    # 1.png: p_geo, p_neu, p_both vs κ
    plt.figure(figsize=(10, 7))
    plt.errorbar(kappa_arr, p_geo_arr, yerr=stderr_geo_arr, fmt='bo-', label='p_geo', capsize=4)
    plt.errorbar(kappa_arr, p_neu_arr, yerr=stderr_neu_arr, fmt='gs-', label='p_neu', capsize=4)
    plt.errorbar(kappa_arr, p_both_arr, yerr=stderr_both_arr, fmt='r^-', label='p_both', capsize=4)
    plt.xlabel('κ', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title(f'Gate Probabilities vs κ (δ={delta:.3f})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/1.png")

    # 2.png: rho vs κ
    plt.figure(figsize=(10, 6))
    valid_rho = np.isfinite(rho_arr)
    valid_err = np.isfinite(stderr_rho_arr)
    kappa_plot = kappa_arr[valid_rho]
    rho_plot = rho_arr[valid_rho]
    err_plot = np.where(valid_err[valid_rho], stderr_rho_arr[valid_rho], 0)

    plt.errorbar(kappa_plot, rho_plot, yerr=err_plot, fmt='ko-', markersize=8,
                 linewidth=2, capsize=4, capthick=1.5, label='ρ(κ)')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='ρ=1 (independence)')
    plt.xlabel('κ', fontsize=14)
    plt.ylabel('ρ = p_both / (p_geo × p_neu)', fontsize=14)
    plt.title('Gate Correlation Ratio vs κ', fontsize=14)
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
        plt.scatter(products[valid], p_both_arr[valid], c=kappa_arr[valid], cmap='viridis',
                    s=140, edgecolors='black', linewidths=1.2)
        plt.colorbar(label='κ')
        max_val = max(np.max(products[valid]), np.max(p_both_arr[valid])) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (independence)')
    else:
        max_val = 1.0
        plt.text(0.5, 0.5, 'No valid points for scatter',
                 transform=plt.gca().transAxes, ha='center', va='center')

    plt.xlabel('p_geo × p_neu', fontsize=14)
    plt.ylabel('p_both', fontsize=14)
    plt.title('Gate Independence Test (color = κ)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/3.png")

    # 4.png: Matched-square test
    epsilon = (p_geo_arr + p_neu_arr) / 2
    eps_sq = epsilon**2
    fit_mask = (~saturated) & np.isfinite(eps_sq) & np.isfinite(p_both_arr)

    plt.figure(figsize=(10, 7))
    plt.scatter(eps_sq, p_both_arr, c=kappa_arr, cmap='plasma',
                s=140, edgecolors='black', linewidths=1.2)
    plt.colorbar(label='κ')

    if np.sum(fit_mask) >= 2:
        coeff = np.polyfit(eps_sq[fit_mask], p_both_arr[fit_mask], 1)
        fit_line = np.poly1d(coeff)
        x_fit = np.linspace(0, np.max(eps_sq[fit_mask]) * 1.05, 100)
        plt.plot(x_fit, fit_line(x_fit), 'k--', linewidth=2,
                 label=f'fit: y={coeff[0]:.2f}x+{coeff[1]:.3f}')

    plt.xlabel('ε² where ε=(p_geo+p_neu)/2', fontsize=14)
    plt.ylabel('p_both', fontsize=14)
    plt.title('Matched-square test', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    inset = plt.axes([0.62, 0.15, 0.3, 0.3])
    ratio = np.where(eps_sq > 0, p_both_arr / eps_sq, np.nan)
    inset.plot(kappa_arr, ratio, 'o-', color='tab:purple', markersize=4)
    inset.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    inset.set_xlabel('κ', fontsize=9)
    inset.set_ylabel('p_both/ε²', fontsize=9)
    inset.grid(True, alpha=0.3)

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
    params = SimulationParams(
        L=24, T=1.0, J=0.6, h=0.0,
        kappa=50.0, g=0.5,
        mu=1.0, gamma=0.5, sigma=1.0,
        dtheta=0.6,
        n_burnin=400, n_sample=1600, n_batches=20
    )

    kappa_list = [10, 15, 20, 30, 40, 50, 65, 80, 100]

    if args.smoke_test:
        params.L = 8
        params.n_burnin = 50
        params.n_sample = 100
        params.n_batches = 10
        kappa_list = [30, 50]

    print(f"\nBase parameters:")
    print(f"  L={params.L}, T={params.T}, J={params.J}, g={params.g}")
    print(f"  mu={params.mu}, gamma={params.gamma}, sigma={params.sigma}")

    # Target probability for calibration
    target_p = 0.2

    # ========== CALIBRATION ==========
    print("\n" + "=" * 70)
    print("PHASE 1: CALIBRATION")
    print("=" * 70)

    # Calibrate δ using κ_ref=50
    kappa_ref = 50
    params_ref = replace(params, kappa=kappa_ref)
    delta = calibrate_delta(params_ref, m_tol=0.1, target_p=target_p,
                            n_burnin=200 if not args.smoke_test else params.n_burnin,
                            n_sample=600 if not args.smoke_test else params.n_sample)

    print(f"\n  FINAL CALIBRATION: δ = {delta:.4f} (κ_ref={kappa_ref})")

    # ========== MAIN SWEEP ==========
    print("\n" + "=" * 70)
    print("PHASE 2: MAIN SWEEP")
    print("=" * 70)

    print(f"\n  Running κ sweep with {len(kappa_list)} values...")
    print(f"  (burn-in: {params.n_burnin}, samples: {params.n_sample}, batches: {params.n_batches})")

    jobs = []
    m_tol_map = {}

    for i, kappa in enumerate(kappa_list):
        params_k = replace(params, kappa=kappa)
        p_geo_est, _ = run_calibration_point(
            params_k, delta, m_tol=0.1,
            n_burnin=200 if not args.smoke_test else params.n_burnin,
            n_sample=600 if not args.smoke_test else params.n_sample,
            seed=100 + i
        )
        target_p_neu = float(np.clip(p_geo_est, 0.05, 0.5))
        print(f"\n  κ={kappa:.1f}: p_geo≈{p_geo_est:.3f} => target p_neu={target_p_neu:.3f}")
        m_tol = calibrate_m_tol(
            params_k, delta, target_p=target_p_neu,
            n_burnin=200 if not args.smoke_test else params.n_burnin,
            n_sample=600 if not args.smoke_test else params.n_sample
        )
        m_tol_map[kappa] = m_tol
        jobs.append((params_k, delta, m_tol, 1000 + i))

    n_workers = min(len(kappa_list), cpu_count())
    print(f"\n  Using {n_workers} parallel workers...")

    with Pool(n_workers) as pool:
        results = pool.map(run_main_sweep_point, jobs)

    for kappa, result in zip(kappa_list, results):
        result['kappa'] = kappa
        result['delta'] = delta
        result['m_tol'] = m_tol_map[kappa]
        result['saturated'] = (result['p_geo'] < 0.02) or (result['p_geo'] > 0.98)

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYSIS")
    print("=" * 70)

    # Print table
    print("\n  Results table:")
    print("  " + "-" * 110)
    print(f"  {'κ':>6} | {'δ':>6} | {'m_tol':>6} | {'p_geo':>7} | {'p_neu':>7} | "
          f"{'p_both':>7} | {'ρ':>7} | {'sat':>5}")
    print("  " + "-" * 110)

    p_geo_arr = np.array([r['p_geo'] for r in results])
    p_neu_arr = np.array([r['p_neu'] for r in results])
    p_both_arr = np.array([r['p_both'] for r in results])
    rho_arr = np.array([r['rho'] for r in results])

    for r in results:
        rho_str = f"{r['rho']:.3f}" if np.isfinite(r['rho']) else "nan"
        sat_str = "yes" if r['saturated'] else "no"
        print(f"  {r['kappa']:>6.1f} | {r['delta']:>6.4f} | {r['m_tol']:>6.4f} | "
              f"{r['p_geo']:>7.4f} | {r['p_neu']:>7.4f} | {r['p_both']:>7.4f} | "
              f"{rho_str:>7} | {sat_str:>5}")

    print("  " + "-" * 110)

    valid_rho = np.isfinite(rho_arr)
    rho_valid = rho_arr[valid_rho]
    if len(rho_valid) > 0:
        mean_rho = np.mean(rho_valid)
        stderr_rho = np.std(rho_valid) / np.sqrt(len(rho_valid))
        print(f"\n  Mean ρ across κ: {mean_rho:.3f} ± {stderr_rho:.3f}")

    # ========== OUTPUT FILES ==========
    print("\n" + "=" * 70)
    print("PHASE 4: SAVING OUTPUT")
    print("=" * 70)

    # Save CSV
    csv_filename = "clean_exponent_results.csv"
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = ['kappa', 'delta', 'm_tol', 'p_geo', 'p_neu', 'p_both', 'rho',
                      'stderr_p_geo', 'stderr_p_neu', 'stderr_p_both', 'stderr_rho',
                      'accept_spin', 'accept_theta', 'mean_abs_phi', 'mean_m', 'saturated']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
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
                'accept_spin': r['accept_spin'],
                'accept_theta': r['accept_theta'],
                'mean_abs_phi': r['mean_phi'],
                'mean_m': r['mean_m'],
                'saturated': r['saturated'],
            })
    print(f"  Saved: {csv_filename}")

    # Create figures
    create_figures(results, kappa_list, delta, output_dir=".")

    # ========== FINAL REPORT ==========
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(f"\n  Calibrated threshold:")
    print(f"    δ = {delta:.4f} (κ_ref={kappa_ref})")

    if len(rho_valid) > 0:
        print(f"\n  Gate independence:")
        print(f"    Mean ρ = {mean_rho:.3f} ± {stderr_rho:.3f}")

    print(f"\n  Data quality:")
    print(f"    Total κ points: {len(kappa_list)}")

    print(f"\n  Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 70)

    if args.smoke_test:
        print("\nSmoke test summary:")
        for res in results:
            rho_str = f"{res['rho']:.3f}" if np.isfinite(res['rho']) else "nan"
            print(f"  κ={res['kappa']:.1f}: p_geo={res['p_geo']:.4f}, "
                  f"p_neu={res['p_neu']:.4f}, p_both={res['p_both']:.4f}, rho={rho_str}")


if __name__ == "__main__":
    main()
