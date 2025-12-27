#!/usr/bin/env python3
"""
Two-Gate Scaling and Universality Test

Tests whether the quadratic gate-stacking law is a robust universality class feature:
- p_both ≈ p_geo * p_neu (gate independence; rho ~ 1)
- Effective exponent relating p_both to control parameter is ~2

Test A: Threshold shrinking - vary tolerances as small parameter
Test B: Matched-permeability - tune to p_geo ≈ p_neu ≈ ε, test p_both ≈ ε²

Author: Scaling analysis for CSF/MVerse two-gate model
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Set
import json
import csv
from scipy import stats
from scipy.optimize import brentq
import time
import os
import warnings
warnings.filterwarnings('ignore')


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


@dataclass
class ScalingParams:
    """Simulation parameters for scaling tests."""
    L: int = 16
    T: float = 2.0
    J: float = 0.3
    h: float = 0.0
    kappa: float = 50.0
    g: float = 0.5
    mu: float = 1.0
    gamma: float = 0.5
    sigma: float = 1.0
    dtheta: float = 0.3
    delta: float = 1.0  # Base geometric gate tolerance
    m_tol: float = 0.12  # Base neutrality gate tolerance
    n_burnin: int = 200  # Reduced for speed
    n_sample: int = 500  # Reduced for speed
    n_batches: int = 10  # Reduced for speed
    use_rectangles: bool = False  # Include 2x1 rectangle cycles

    @property
    def N(self) -> int:
        return self.L * self.L

    def copy(self, **kwargs):
        """Return a copy with updated parameters."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(kwargs)
        return ScalingParams(**d)


class ScalingLattice:
    """
    2D periodic square lattice with optional extended cycle support.
    """

    def __init__(self, params: ScalingParams, rng: np.random.Generator = None):
        self.params = params
        self.L = params.L
        self.N = params.N
        self.rng = rng if rng is not None else np.random.default_rng()

        # State variables
        self.spins = np.zeros(self.N, dtype=np.int8)
        self.credits = np.zeros(self.N, dtype=np.float64)

        self._setup_edges()
        self._setup_cycles()
        self.initialize_random()

        # Cached quantities
        self._phi_cache = np.zeros(self.n_cycles, dtype=np.float64)
        self._q_cache = np.zeros(self.N, dtype=np.float64)
        self._update_all_caches()

    def _setup_edges(self):
        """Setup edge indexing."""
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

    def _setup_cycles(self):
        """Setup cycle data structures (plaquettes and optionally rectangles)."""
        L = self.L
        self.cycle_edges = []
        self.cycle_vertices = []

        # Plaquettes (4-cycles)
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

                edges = [(h_bottom, +1), (v_right, +1), (h_top, -1), (v_left, -1)]
                self.cycle_edges.append(edges)
                self.cycle_vertices.append([i, i_right, i_diag, i_up])

        # Optionally add 2x1 rectangles (6-cycles)
        if self.params.use_rectangles:
            # Horizontal 2x1 rectangles
            for y in range(L):
                for x in range(L):
                    i = x + y * L
                    x1 = (x + 1) % L
                    x2 = (x + 2) % L
                    y_up = (y + 1) % L

                    i1 = x1 + y * L
                    i2 = x2 + y * L
                    i_up = x + y_up * L
                    i1_up = x1 + y_up * L
                    i2_up = x2 + y_up * L

                    edges = [
                        (i, +1),  # h: i -> i1
                        (i1, +1),  # h: i1 -> i2
                        (self.N + i2, +1),  # v: i2 -> i2_up
                        (i1_up, -1),  # h: i2_up -> i1_up
                        (i_up, -1),  # h: i1_up -> i_up
                        (self.N + i, -1),  # v: i_up -> i
                    ]
                    self.cycle_edges.append(edges)
                    self.cycle_vertices.append([i, i1, i2, i2_up, i1_up, i_up])

            # Vertical 2x1 rectangles
            for y in range(L):
                for x in range(L):
                    i = x + y * L
                    x_right = (x + 1) % L
                    y1 = (y + 1) % L
                    y2 = (y + 2) % L

                    i_right = x_right + y * L
                    i_up1 = x + y1 * L
                    i_up2 = x + y2 * L
                    i_right_up1 = x_right + y1 * L
                    i_right_up2 = x_right + y2 * L

                    edges = [
                        (i, +1),  # h: i -> i_right
                        (self.N + i_right, +1),  # v: i_right -> i_right_up1
                        (self.N + i_right_up1, +1),  # v: i_right_up1 -> i_right_up2
                        (i_up2, -1),  # h: i_right_up2 -> i_up2
                        (self.N + i_up1, -1),  # v: i_up2 -> i_up1
                        (self.N + i, -1),  # v: i_up1 -> i
                    ]
                    self.cycle_edges.append(edges)
                    self.cycle_vertices.append([i, i_right, i_right_up1, i_right_up2, i_up2, i_up1])

        self.n_cycles = len(self.cycle_edges)

        # For each edge, list of cycles containing it
        self.edge_cycles = [[] for _ in range(self.n_edges)]
        for c_idx, edges in enumerate(self.cycle_edges):
            for e_idx, _ in edges:
                self.edge_cycles[e_idx].append(c_idx)

        # For each site, list of cycles containing it
        self.site_cycles = [[] for _ in range(self.N)]
        for c_idx, verts in enumerate(self.cycle_vertices):
            for v in verts:
                if c_idx not in self.site_cycles[v]:
                    self.site_cycles[v].append(c_idx)

    def get_theta(self, e_idx: int, orientation: int) -> float:
        return orientation * self.theta[e_idx]

    def compute_phi(self, c_idx: int) -> float:
        total = 0.0
        for e_idx, orient in self.cycle_edges[c_idx]:
            total += self.get_theta(e_idx, orient)
        return wrap_angle(total)

    def _update_phi_cache(self, c_idx: int):
        self._phi_cache[c_idx] = self.compute_phi(c_idx)

    def _update_q_cache(self, site: int):
        q = 0.0
        for c_idx in self.site_cycles[site]:
            q += 1.0 - np.cos(self._phi_cache[c_idx])
        self._q_cache[site] = q

    def _update_all_caches(self):
        for c_idx in range(self.n_cycles):
            self._update_phi_cache(c_idx)
        for i in range(self.N):
            self._update_q_cache(i)

    def initialize_random(self):
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(np.int8)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.n_edges)
        self.credits = self.rng.normal(0, self.params.sigma, size=self.N)

    # Energy computations
    def H_s(self) -> float:
        p = self.params
        energy = 0.0
        for e_idx in range(self.n_edges):
            i, j = self.edge_endpoints[e_idx]
            energy -= p.J * self.spins[i] * self.spins[j]
        energy -= p.h * np.sum(self.spins)
        return energy

    def H_Theta(self) -> float:
        return self.params.kappa * np.sum(1.0 - np.cos(self._phi_cache))

    def H_sTheta(self) -> float:
        p = self.params
        energy = 0.0
        for e_idx in range(self.n_edges):
            i, j = self.edge_endpoints[e_idx]
            P_ij = ((1 + self.spins[i]) / 2) * ((1 + self.spins[j]) / 2)
            energy -= p.g * P_ij * np.cos(self.theta[e_idx])
        return energy

    def H_c(self) -> float:
        p = self.params
        c = self.credits
        s = self.spins
        q = self._q_cache
        return np.sum(c**2 / (2 * p.sigma**2) - p.mu * c * s + p.gamma * c * q)

    # Local energy changes
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

        affected_cycles = self.edge_cycles[e_idx]
        old_phis = [self._phi_cache[c_idx] for c_idx in affected_cycles]

        self.theta[e_idx] = theta_new
        new_phis = [self.compute_phi(c_idx) for c_idx in affected_cycles]
        self.theta[e_idx] = theta_old

        delta_HT = 0.0
        for old_phi, new_phi in zip(old_phis, new_phis):
            delta_HT += p.kappa * ((1 - np.cos(new_phi)) - (1 - np.cos(old_phi)))

        affected_sites = set()
        for c_idx in affected_cycles:
            affected_sites.update(self.cycle_vertices[c_idx])

        delta_Hc = 0.0
        for site in affected_sites:
            c_i = self.credits[site]
            delta_q = 0.0
            for c_idx in self.site_cycles[site]:
                if c_idx in affected_cycles:
                    idx = affected_cycles.index(c_idx)
                    delta_q += (1 - np.cos(new_phis[idx])) - (1 - np.cos(old_phis[idx]))
            delta_Hc += p.gamma * c_i * delta_q

        return delta_HT + delta_HsT + delta_Hc

    # Monte Carlo updates
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
            for c_idx in self.edge_cycles[e_idx]:
                self._update_phi_cache(c_idx)
            affected_sites = set()
            for c_idx in self.edge_cycles[e_idx]:
                affected_sites.update(self.cycle_vertices[c_idx])
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

    def sweep(self) -> Tuple[float, float]:
        """One full sweep. Returns (spin_accept_rate, edge_accept_rate)."""
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

        return spin_accepts / self.N, edge_accepts / self.n_edges

    # Observables
    def magnetization(self) -> float:
        return np.mean(self.spins)

    def max_holonomy(self) -> float:
        return np.max(np.abs(self._phi_cache))

    def mean_holonomy_magnitude(self) -> float:
        return np.mean(np.abs(self._phi_cache))

    def geometric_gate(self, delta: float = None) -> bool:
        if delta is None:
            delta = self.params.delta
        return self.max_holonomy() <= delta

    def neutrality_gate(self, m_tol: float = None) -> bool:
        if m_tol is None:
            m_tol = self.params.m_tol
        return np.abs(self.magnetization()) <= m_tol

    def both_gates(self, delta: float = None, m_tol: float = None) -> bool:
        return self.geometric_gate(delta) and self.neutrality_gate(m_tol)


def run_simulation(params: ScalingParams, delta: float = None, m_tol: float = None,
                   seed: int = None, verbose: bool = False) -> Dict:
    """Run simulation and return gate statistics."""
    if delta is None:
        delta = params.delta
    if m_tol is None:
        m_tol = params.m_tol

    rng = np.random.default_rng(seed)
    lattice = ScalingLattice(params, rng)

    # Burn-in
    spin_rates = []
    edge_rates = []
    for i in range(params.n_burnin):
        sr, er = lattice.sweep()
        spin_rates.append(sr)
        edge_rates.append(er)

    # Sampling
    G_geo = []
    G_neu = []
    G_both = []
    m_vals = []
    phi_vals = []

    for i in range(params.n_sample):
        sr, er = lattice.sweep()
        spin_rates.append(sr)
        edge_rates.append(er)

        G_geo.append(1 if lattice.geometric_gate(delta) else 0)
        G_neu.append(1 if lattice.neutrality_gate(m_tol) else 0)
        G_both.append(1 if lattice.both_gates(delta, m_tol) else 0)
        m_vals.append(lattice.magnetization())
        phi_vals.append(lattice.mean_holonomy_magnitude())

    G_geo = np.array(G_geo)
    G_neu = np.array(G_neu)
    G_both = np.array(G_both)

    p_geo = np.mean(G_geo)
    p_neu = np.mean(G_neu)
    p_both = np.mean(G_both)

    product = p_geo * p_neu
    rho = p_both / product if product > 1e-10 else np.nan

    # Batch error estimation
    batch_size = params.n_sample // params.n_batches

    def batch_stats(arr):
        batches = arr[:batch_size * params.n_batches].reshape(params.n_batches, batch_size)
        batch_vals = np.mean(batches, axis=1)
        return np.mean(batch_vals), np.std(batch_vals) / np.sqrt(params.n_batches)

    p_geo_mean, p_geo_err = batch_stats(G_geo)
    p_neu_mean, p_neu_err = batch_stats(G_neu)
    p_both_mean, p_both_err = batch_stats(G_both)

    # Rho error via batch means
    rho_batches = []
    for b in range(params.n_batches):
        start = b * batch_size
        end = (b + 1) * batch_size
        pg_b = np.mean(G_geo[start:end])
        pn_b = np.mean(G_neu[start:end])
        pb_b = np.mean(G_both[start:end])
        prod_b = pg_b * pn_b
        if prod_b > 1e-10:
            rho_batches.append(pb_b / prod_b)
    rho_err = np.std(rho_batches) / np.sqrt(len(rho_batches)) if len(rho_batches) > 1 else np.nan

    return {
        'p_geo': p_geo,
        'p_neu': p_neu,
        'p_both': p_both,
        'rho': rho,
        'stderr_p_geo': p_geo_err,
        'stderr_p_neu': p_neu_err,
        'stderr_p_both': p_both_err,
        'stderr_rho': rho_err,
        'accept_spin': np.mean(spin_rates),
        'accept_theta': np.mean(edge_rates),
        'mean_abs_phi': np.mean(phi_vals),
        'mean_m': np.mean(m_vals),
        'params': params,
        'delta_used': delta,
        'm_tol_used': m_tol,
    }


def test_A_threshold_shrinking(base_params: ScalingParams, delta0: float, m0: float,
                                t_values: List[float], seed: int = 42,
                                verbose: bool = True) -> List[Dict]:
    """
    Test A: Threshold shrinking test.
    Vary tolerances as δ(t) = δ0*t, m_tol(t) = m0*t for t in t_values.
    """
    results = []
    if verbose:
        print(f"\n=== Test A: Threshold Shrinking (L={base_params.L}, T={base_params.T}) ===")
        print(f"Base: δ0={delta0}, m0={m0}")

    for t in t_values:
        delta = delta0 * t
        m_tol = m0 * t

        if verbose:
            print(f"  t={t:.2f}: δ={delta:.3f}, m_tol={m_tol:.3f} ... ", end='', flush=True)

        result = run_simulation(base_params, delta=delta, m_tol=m_tol, seed=seed)
        result['test_type'] = 'A'
        result['t'] = t
        result['target_eps'] = None
        result['delta0'] = delta0
        result['m0'] = m0
        results.append(result)

        if verbose:
            print(f"p_geo={result['p_geo']:.3f}, p_neu={result['p_neu']:.3f}, "
                  f"p_both={result['p_both']:.3f}, ρ={result['rho']:.3f}")

    return results


def fit_exponents(results: List[Dict]) -> Dict:
    """
    Fit log-log slopes for Test A results.
    Returns exponents a (p_geo), b (p_neu), c (p_both) with errors.
    """
    # Filter out saturated points
    valid = [(r['t'], r['p_geo'], r['p_neu'], r['p_both'])
             for r in results
             if 0.01 < r['p_geo'] < 0.99 and 0.01 < r['p_neu'] < 0.99]

    if len(valid) < 3:
        return {'a': np.nan, 'b': np.nan, 'c': np.nan,
                'a_err': np.nan, 'b_err': np.nan, 'c_err': np.nan,
                'n_valid': len(valid)}

    t_vals = np.array([v[0] for v in valid])
    p_geo_vals = np.array([v[1] for v in valid])
    p_neu_vals = np.array([v[2] for v in valid])
    p_both_vals = np.array([v[3] for v in valid])

    log_t = np.log(t_vals)

    # Fit p_geo vs t
    mask_geo = p_geo_vals > 0
    if np.sum(mask_geo) >= 2:
        slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(log_t[mask_geo], np.log(p_geo_vals[mask_geo]))
    else:
        slope_a, se_a = np.nan, np.nan

    # Fit p_neu vs t
    mask_neu = p_neu_vals > 0
    if np.sum(mask_neu) >= 2:
        slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(log_t[mask_neu], np.log(p_neu_vals[mask_neu]))
    else:
        slope_b, se_b = np.nan, np.nan

    # Fit p_both vs t
    mask_both = p_both_vals > 0
    if np.sum(mask_both) >= 2:
        slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(log_t[mask_both], np.log(p_both_vals[mask_both]))
    else:
        slope_c, se_c = np.nan, np.nan

    return {
        'a': slope_a, 'b': slope_b, 'c': slope_c,
        'a_err': se_a, 'b_err': se_b, 'c_err': se_c,
        'n_valid': len(valid),
        't_vals': t_vals,
        'p_geo_vals': p_geo_vals,
        'p_neu_vals': p_neu_vals,
        'p_both_vals': p_both_vals,
    }


def tune_kappa_for_p_geo(base_params: ScalingParams, target_p_geo: float,
                          delta: float, kappa_range: Tuple[float, float] = (10, 200),
                          tol: float = 0.05, max_iter: int = 8, seed: int = 42) -> float:
    """
    Tune kappa to achieve target p_geo using bisection.
    Higher kappa -> smaller holonomies -> higher p_geo.
    """
    kappa_lo, kappa_hi = kappa_range

    # Quick check bounds
    params_lo = base_params.copy(kappa=kappa_lo)
    result_lo = run_simulation(params_lo, delta=delta, seed=seed)
    p_lo = result_lo['p_geo']

    params_hi = base_params.copy(kappa=kappa_hi)
    result_hi = run_simulation(params_hi, delta=delta, seed=seed)
    p_hi = result_hi['p_geo']

    if target_p_geo < p_lo or target_p_geo > p_hi:
        # Target out of range, return closest
        if abs(target_p_geo - p_lo) < abs(target_p_geo - p_hi):
            return kappa_lo
        return kappa_hi

    # Bisection
    for _ in range(max_iter):
        kappa_mid = (kappa_lo + kappa_hi) / 2
        params_mid = base_params.copy(kappa=kappa_mid)
        result_mid = run_simulation(params_mid, delta=delta, seed=seed)
        p_mid = result_mid['p_geo']

        if abs(p_mid - target_p_geo) < tol:
            return kappa_mid

        if p_mid < target_p_geo:
            kappa_lo = kappa_mid
        else:
            kappa_hi = kappa_mid

    return (kappa_lo + kappa_hi) / 2


def tune_h_for_p_neu(base_params: ScalingParams, target_p_neu: float,
                     m_tol: float, h_range: Tuple[float, float] = (-0.5, 0.5),
                     tol: float = 0.05, max_iter: int = 8, seed: int = 42) -> float:
    """
    Tune h to achieve target p_neu.
    h=0 gives symmetric magnetization distribution.
    """
    # At h=0, p_neu is typically highest (centered at m=0)
    params_0 = base_params.copy(h=0)
    result_0 = run_simulation(params_0, m_tol=m_tol, seed=seed)
    p_0 = result_0['p_neu']

    if target_p_neu >= p_0:
        return 0.0  # Can't get higher than h=0

    # Search in positive h direction (breaks symmetry, reduces p_neu)
    h_lo, h_hi = 0.0, h_range[1]

    for _ in range(max_iter):
        h_mid = (h_lo + h_hi) / 2
        params_mid = base_params.copy(h=h_mid)
        result_mid = run_simulation(params_mid, m_tol=m_tol, seed=seed)
        p_mid = result_mid['p_neu']

        if abs(p_mid - target_p_neu) < tol:
            return h_mid

        if p_mid > target_p_neu:
            h_lo = h_mid
        else:
            h_hi = h_mid

    return (h_lo + h_hi) / 2


def test_B_matched_permeability(base_params: ScalingParams, eps_values: List[float],
                                 delta: float, m_tol: float,
                                 seed: int = 42, verbose: bool = True) -> List[Dict]:
    """
    Test B: Matched permeability test.
    For each ε, tune kappa and h to achieve p_geo ≈ p_neu ≈ ε.
    """
    results = []
    if verbose:
        print(f"\n=== Test B: Matched Permeability (L={base_params.L}, T={base_params.T}) ===")

    for eps in eps_values:
        if verbose:
            print(f"  Target ε={eps:.2f}: ", end='', flush=True)

        # First tune kappa for p_geo
        kappa_tuned = tune_kappa_for_p_geo(base_params, eps, delta, seed=seed)

        # Then tune h for p_neu (with kappa fixed)
        params_with_kappa = base_params.copy(kappa=kappa_tuned)
        h_tuned = tune_h_for_p_neu(params_with_kappa, eps, m_tol, seed=seed)

        # Final measurement with tuned parameters
        params_final = params_with_kappa.copy(h=h_tuned)
        result = run_simulation(params_final, delta=delta, m_tol=m_tol, seed=seed)

        result['test_type'] = 'B'
        result['t'] = None
        result['target_eps'] = eps
        result['kappa_tuned'] = kappa_tuned
        result['h_tuned'] = h_tuned
        results.append(result)

        if verbose:
            print(f"κ={kappa_tuned:.1f}, h={h_tuned:.3f} -> "
                  f"p_geo={result['p_geo']:.3f}, p_neu={result['p_neu']:.3f}, "
                  f"p_both={result['p_both']:.3f}, ρ={result['rho']:.3f}")

    return results


def run_scaling_analysis(output_dir: str = 'results/two_gate_scaling'):
    """Main scaling analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Two-Gate Scaling and Universality Analysis")
    print("=" * 70)

    start_time = time.time()
    all_results = []
    exponent_data = []

    # Base parameters
    base_delta = 1.0
    base_m_tol = 0.12
    t_values = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0]  # Focus on nontrivial range
    eps_values = [0.3, 0.5, 0.7]  # Fewer points for speed

    # ==================== SYSTEM SIZE SCALING ====================
    print("\n" + "=" * 70)
    print("PART 1: System Size Scaling")
    print("=" * 70)

    L_values = [12, 16]  # Reduced for speed
    g_values_for_L = [0.5]

    for L in L_values:
        # Adjust parameters for system size
        n_burnin = 200 + L * 5
        n_sample = 400 + L * 10

        for g in g_values_for_L:
            base_params = ScalingParams(
                L=L, T=2.0, J=0.3, kappa=50.0, g=g,
                n_burnin=n_burnin, n_sample=n_sample
            )

            # Test A
            results_A = test_A_threshold_shrinking(base_params, base_delta, base_m_tol, t_values)
            all_results.extend(results_A)

            # Fit exponents
            exp_fit = fit_exponents(results_A)
            exp_fit['L'] = L
            exp_fit['g'] = g
            exp_fit['T'] = base_params.T
            exp_fit['use_rectangles'] = False
            exponent_data.append(exp_fit)

            print(f"  Exponents: a={exp_fit['a']:.3f}±{exp_fit['a_err']:.3f}, "
                  f"b={exp_fit['b']:.3f}±{exp_fit['b_err']:.3f}, "
                  f"c={exp_fit['c']:.3f}±{exp_fit['c_err']:.3f}")
            if not np.isnan(exp_fit['a']) and not np.isnan(exp_fit['b']):
                print(f"  c - (a+b) = {exp_fit['c'] - (exp_fit['a'] + exp_fit['b']):.3f}")

            # Test B (only for one L to save time)
            if L == 16:
                results_B = test_B_matched_permeability(base_params, eps_values, base_delta, base_m_tol)
                all_results.extend(results_B)

    # ==================== COUPLING VARIATION ====================
    print("\n" + "=" * 70)
    print("PART 2: Coupling g Variation")
    print("=" * 70)

    g_values = [0.0, 0.5, 1.0, 2.0]
    L_for_g = 12  # Smaller L for speed

    for g in g_values:
        base_params = ScalingParams(L=L_for_g, T=2.0, J=0.3, kappa=50.0, g=g)

        # Test A
        results_A = test_A_threshold_shrinking(base_params, base_delta, base_m_tol, t_values)
        all_results.extend(results_A)

        exp_fit = fit_exponents(results_A)
        exp_fit['L'] = L_for_g
        exp_fit['g'] = g
        exp_fit['T'] = base_params.T
        exp_fit['use_rectangles'] = False
        exponent_data.append(exp_fit)

        print(f"  g={g}: Exponents a={exp_fit['a']:.3f}, b={exp_fit['b']:.3f}, c={exp_fit['c']:.3f}")

    # ==================== TEMPERATURE VARIATION ====================
    print("\n" + "=" * 70)
    print("PART 3: Temperature Variation")
    print("=" * 70)

    T_values = [1.5, 2.0, 3.0]
    L_for_T = 12  # Smaller L for speed

    for T in T_values:
        base_params = ScalingParams(L=L_for_T, T=T, J=0.3, kappa=50.0, g=0.5)

        # Test A
        results_A = test_A_threshold_shrinking(base_params, base_delta, base_m_tol, t_values)
        all_results.extend(results_A)

        exp_fit = fit_exponents(results_A)
        exp_fit['L'] = L_for_T
        exp_fit['g'] = 0.5
        exp_fit['T'] = T
        exp_fit['use_rectangles'] = False
        exponent_data.append(exp_fit)

        print(f"  T={T}: Exponents a={exp_fit['a']:.3f}, b={exp_fit['b']:.3f}, c={exp_fit['c']:.3f}")

    # ==================== CYCLE VARIATION ====================
    print("\n" + "=" * 70)
    print("PART 4: Cycle Set Variation (Plaquettes + Rectangles)")
    print("=" * 70)

    # With rectangles
    base_params_rect = ScalingParams(
        L=12, T=2.0, J=0.3, kappa=50.0, g=0.5,
        use_rectangles=True, n_burnin=400, n_sample=800
    )

    results_rect = test_A_threshold_shrinking(base_params_rect, base_delta, base_m_tol, t_values)
    all_results.extend(results_rect)

    exp_fit_rect = fit_exponents(results_rect)
    exp_fit_rect['L'] = 12
    exp_fit_rect['g'] = 0.5
    exp_fit_rect['T'] = 2.0
    exp_fit_rect['use_rectangles'] = True
    exponent_data.append(exp_fit_rect)

    print(f"  With rectangles: a={exp_fit_rect['a']:.3f}, b={exp_fit_rect['b']:.3f}, c={exp_fit_rect['c']:.3f}")

    # ==================== SAVE RESULTS ====================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    # Save CSV
    csv_path = f'{output_dir}/results_scaling.csv'
    fieldnames = [
        'L', 'T', 'J', 'h', 'kappa', 'g', 'mu', 'gamma', 'sigma', 'delta', 'm_tol',
        'p_geo', 'p_neu', 'p_both', 'rho',
        'stderr_p_geo', 'stderr_p_neu', 'stderr_p_both', 'stderr_rho',
        'accept_spin', 'accept_theta', 'mean_abs_phi', 'mean_m',
        'test_type', 't', 'target_eps'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            p = r['params']
            row = {
                'L': p.L, 'T': p.T, 'J': p.J, 'h': p.h, 'kappa': p.kappa,
                'g': p.g, 'mu': p.mu, 'gamma': p.gamma, 'sigma': p.sigma,
                'delta': r['delta_used'], 'm_tol': r['m_tol_used'],
                'p_geo': r['p_geo'], 'p_neu': r['p_neu'], 'p_both': r['p_both'],
                'rho': r['rho'],
                'stderr_p_geo': r['stderr_p_geo'], 'stderr_p_neu': r['stderr_p_neu'],
                'stderr_p_both': r['stderr_p_both'], 'stderr_rho': r['stderr_rho'],
                'accept_spin': r['accept_spin'], 'accept_theta': r['accept_theta'],
                'mean_abs_phi': r['mean_abs_phi'], 'mean_m': r['mean_m'],
                'test_type': r['test_type'], 't': r.get('t'),
                'target_eps': r.get('target_eps')
            }
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    # ==================== CREATE FIGURES ====================
    print("\nCreating figures...")
    create_figures(all_results, exponent_data, output_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # ==================== SUMMARY ====================
    print_summary(all_results, exponent_data)


def create_figures(all_results: List[Dict], exponent_data: List[Dict], output_dir: str):
    """Create all figures."""

    # 1.png: Log-log plot for Test A (representative)
    test_A_results = [r for r in all_results
                      if r['test_type'] == 'A' and r['params'].L == 16 and r['params'].g == 0.5
                      and r['params'].T == 2.0 and not r['params'].use_rectangles]

    if test_A_results:
        t_vals = np.array([r['t'] for r in test_A_results])
        p_geo = np.array([r['p_geo'] for r in test_A_results])
        p_neu = np.array([r['p_neu'] for r in test_A_results])
        p_both = np.array([r['p_both'] for r in test_A_results])

        # Filter valid points
        valid = (p_geo > 0.01) & (p_neu > 0.01) & (p_both > 0.01)

        plt.figure(figsize=(10, 8))

        if np.any(valid):
            plt.loglog(t_vals[valid], p_geo[valid], 'bo-', markersize=8, label='p_geo', linewidth=2)
            plt.loglog(t_vals[valid], p_neu[valid], 'g^-', markersize=8, label='p_neu', linewidth=2)
            plt.loglog(t_vals[valid], p_both[valid], 'rs-', markersize=8, label='p_both', linewidth=2)

            # Fit lines
            log_t = np.log(t_vals[valid])
            if np.sum(p_geo[valid] > 0) >= 2:
                a, _, _, _, _ = stats.linregress(log_t, np.log(p_geo[valid]))
                plt.loglog(t_vals[valid], np.exp(np.log(p_geo[valid][0]) + a * (log_t - log_t[0])),
                          'b--', alpha=0.5, label=f'slope a={a:.2f}')
            if np.sum(p_neu[valid] > 0) >= 2:
                b, _, _, _, _ = stats.linregress(log_t, np.log(p_neu[valid]))
                plt.loglog(t_vals[valid], np.exp(np.log(p_neu[valid][0]) + b * (log_t - log_t[0])),
                          'g--', alpha=0.5, label=f'slope b={b:.2f}')
            if np.sum(p_both[valid] > 0) >= 2:
                c, _, _, _, _ = stats.linregress(log_t, np.log(p_both[valid]))
                plt.loglog(t_vals[valid], np.exp(np.log(p_both[valid][0]) + c * (log_t - log_t[0])),
                          'r--', alpha=0.5, label=f'slope c={c:.2f}')

        plt.xlabel('t (tolerance scaling factor)', fontsize=12)
        plt.ylabel('Gate probability', fontsize=12)
        plt.title('Test A: Log-Log Scaling (L=16, g=0.5, T=2.0)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/1.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/1.png")

    # 2.png: c - (a+b) vs L
    L_exps = [e for e in exponent_data if e['g'] == 0.5 and e['T'] == 2.0 and not e.get('use_rectangles', False)]

    if L_exps:
        plt.figure(figsize=(8, 6))
        Ls = [e['L'] for e in L_exps]
        c_minus_ab = [e['c'] - (e['a'] + e['b']) if not np.isnan(e['c']) else np.nan for e in L_exps]

        valid_idx = [i for i, v in enumerate(c_minus_ab) if not np.isnan(v)]
        if valid_idx:
            plt.plot([Ls[i] for i in valid_idx], [c_minus_ab[i] for i in valid_idx],
                    'ko-', markersize=10, linewidth=2)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Multiplicative (c=a+b)')
            plt.xlabel('L (system size)', fontsize=12)
            plt.ylabel('c - (a + b)', fontsize=12)
            plt.title('Test A: Multiplicativity Check vs System Size', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/2.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/2.png")

    # 3.png: Test B - p_both vs p_geo*p_neu
    test_B_results = [r for r in all_results if r['test_type'] == 'B']

    if test_B_results:
        plt.figure(figsize=(8, 8))
        products = [r['p_geo'] * r['p_neu'] for r in test_B_results]
        p_boths = [r['p_both'] for r in test_B_results]

        plt.scatter(products, p_boths, s=100, c='blue', alpha=0.7)
        max_val = max(max(products), max(p_boths)) if products else 1
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (independence)')

        plt.xlabel('p_geo × p_neu', fontsize=12)
        plt.ylabel('p_both', fontsize=12)
        plt.title('Test B: Matched Permeability - Independence Check', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/3.png")

    # 4.png: Test B - p_both vs ε²
    if test_B_results:
        plt.figure(figsize=(8, 8))
        eps_vals = []
        p_boths = []
        for r in test_B_results:
            eps = (r['p_geo'] + r['p_neu']) / 2  # Actual matched epsilon
            if eps > 0.01:
                eps_vals.append(eps)
                p_boths.append(r['p_both'])

        if eps_vals:
            eps_vals = np.array(eps_vals)
            p_boths = np.array(p_boths)
            eps_sq = eps_vals ** 2

            plt.scatter(eps_sq, p_boths, s=100, c='blue', alpha=0.7, label='Data')
            max_val = max(max(eps_sq), max(p_boths))
            plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (p_both=ε²)')

            # Fit slope in log space
            if len(eps_vals) >= 2:
                valid = (eps_vals > 0) & (np.array(p_boths) > 0)
                if np.sum(valid) >= 2:
                    slope, intercept, r, p, se = stats.linregress(
                        np.log(eps_vals[valid]), np.log(np.array(p_boths)[valid]))
                    plt.title(f'Test B: p_both vs ε² (fitted exponent: {slope:.2f})', fontsize=14)
                else:
                    plt.title('Test B: p_both vs ε²', fontsize=14)
            else:
                plt.title('Test B: p_both vs ε²', fontsize=14)

        plt.xlabel('ε² (where ε ≈ p_geo ≈ p_neu)', fontsize=12)
        plt.ylabel('p_both', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/4.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/4.png")

    # 5.png: rho vs g for several L
    plt.figure(figsize=(10, 6))

    for L in [12, 16, 24]:
        g_rho_data = []
        for g in [0.0, 0.5, 1.0, 2.0]:
            matching = [r for r in all_results
                       if r['test_type'] == 'A' and r['params'].L == L and r['params'].g == g
                       and r['t'] == 1.0 and r['params'].T == 2.0]
            if matching:
                rho_vals = [r['rho'] for r in matching if not np.isnan(r['rho'])]
                if rho_vals:
                    g_rho_data.append((g, np.mean(rho_vals)))

        if g_rho_data:
            gs, rhos = zip(*g_rho_data)
            plt.plot(gs, rhos, 'o-', markersize=8, linewidth=2, label=f'L={L}')

    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='ρ=1')
    plt.xlabel('g (spin-phase coupling)', fontsize=12)
    plt.ylabel('ρ = p_both / (p_geo × p_neu)', fontsize=12)
    plt.title('Gate Correlation ρ vs Coupling Strength', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/5.png")

    # 6.png: rho vs L for g=0.5 and g=2.0
    plt.figure(figsize=(10, 6))

    for g in [0.5, 2.0]:
        L_rho_data = []
        for L in [12, 16, 24]:
            matching = [r for r in all_results
                       if r['test_type'] == 'A' and r['params'].L == L and r['params'].g == g
                       and r['t'] == 1.0 and r['params'].T == 2.0]
            if matching:
                rho_vals = [r['rho'] for r in matching if not np.isnan(r['rho'])]
                if rho_vals:
                    L_rho_data.append((L, np.mean(rho_vals)))

        if L_rho_data:
            Ls, rhos = zip(*L_rho_data)
            plt.plot(Ls, rhos, 'o-', markersize=10, linewidth=2, label=f'g={g}')

    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='ρ=1')
    plt.xlabel('L (system size)', fontsize=12)
    plt.ylabel('ρ = p_both / (p_geo × p_neu)', fontsize=12)
    plt.title('Gate Correlation ρ vs System Size', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/6.png")


def print_summary(all_results: List[Dict], exponent_data: List[Dict]):
    """Print interpretation summary."""
    print("\n" + "=" * 70)
    print("SUMMARY AND INTERPRETATION")
    print("=" * 70)

    # Check rho values
    rho_vals = [r['rho'] for r in all_results if not np.isnan(r['rho']) and
                0.05 < r['p_geo'] < 0.95 and 0.05 < r['p_neu'] < 0.95]

    if rho_vals:
        print(f"\n1. Gate Independence (ρ ≈ 1?):")
        print(f"   Non-saturated regime: ρ ranges from {min(rho_vals):.3f} to {max(rho_vals):.3f}")
        print(f"   Mean ρ = {np.mean(rho_vals):.3f} ± {np.std(rho_vals):.3f}")
        if abs(np.mean(rho_vals) - 1.0) < 0.1:
            print(f"   RESULT: ρ remains O(1) and near 1 in non-saturated regime ✓")
        else:
            print(f"   RESULT: ρ shows deviation from 1")

    # Check Test A exponents
    valid_exps = [e for e in exponent_data if not np.isnan(e['c']) and e['n_valid'] >= 3]

    if valid_exps:
        print(f"\n2. Test A - Multiplicativity (c ≈ a + b?):")
        c_minus_ab = [e['c'] - (e['a'] + e['b']) for e in valid_exps]
        print(f"   c - (a+b) ranges from {min(c_minus_ab):.3f} to {max(c_minus_ab):.3f}")
        print(f"   Mean deviation: {np.mean(c_minus_ab):.3f} ± {np.std(c_minus_ab):.3f}")

        if abs(np.mean(c_minus_ab)) < 0.2:
            print(f"   RESULT: c ≈ a + b within errors (multiplicativity holds) ✓")
        else:
            print(f"   RESULT: c deviates from a + b")

        # Check if a ≈ b implies c ≈ 2a
        matched_exps = [e for e in valid_exps if abs(e['a'] - e['b']) < 0.3]
        if matched_exps:
            c_over_a = [e['c'] / e['a'] for e in matched_exps if abs(e['a']) > 0.1]
            if c_over_a:
                print(f"\n3. Test A - Quadratic Stacking (when a≈b, c≈2a?):")
                print(f"   c/a ratios: {[f'{x:.2f}' for x in c_over_a]}")
                print(f"   Mean c/a = {np.mean(c_over_a):.2f}")
                if abs(np.mean(c_over_a) - 2.0) < 0.3:
                    print(f"   RESULT: c ≈ 2a when a ≈ b (quadratic stacking) ✓")
                else:
                    print(f"   RESULT: c/a deviates from 2")

    # System size dependence
    L_exps = [e for e in exponent_data if e['g'] == 0.5 and not e.get('use_rectangles', False)]
    if len(L_exps) >= 2:
        print(f"\n4. Size Dependence:")
        for e in sorted(L_exps, key=lambda x: x['L']):
            if not np.isnan(e['c']):
                print(f"   L={e['L']}: c={e['c']:.3f}, c-(a+b)={e['c']-(e['a']+e['b']):.3f}")

    # Coupling dependence
    g_exps = [e for e in exponent_data if e['L'] == 16 and not e.get('use_rectangles', False)]
    if len([e for e in g_exps if not np.isnan(e['c'])]) >= 2:
        print(f"\n5. Coupling Dependence:")
        for e in sorted(g_exps, key=lambda x: x['g']):
            if not np.isnan(e['c']):
                print(f"   g={e['g']}: c={e['c']:.3f}, c-(a+b)={e['c']-(e['a']+e['b']):.3f}")

    # Cycle set variation
    rect_exps = [e for e in exponent_data if e.get('use_rectangles', False)]
    plaq_exps = [e for e in exponent_data if not e.get('use_rectangles', False) and e['L'] == 12]

    if rect_exps and plaq_exps:
        print(f"\n6. Cycle Set Variation:")
        for e in plaq_exps:
            if not np.isnan(e['c']):
                print(f"   Plaquettes only (L={e['L']}): c={e['c']:.3f}")
        for e in rect_exps:
            if not np.isnan(e['c']):
                print(f"   Plaquettes + rectangles (L={e['L']}): c={e['c']:.3f}")

    print("\n" + "=" * 70)
    print("NOTE: This is validation (not proof) of multiplicativity/quadratic")
    print("stacking in a toy model. Results may vary with parameters.")
    print("=" * 70)


if __name__ == "__main__":
    run_scaling_analysis()
