#!/usr/bin/env python3
"""
Two-Gate Interface Model Monte Carlo Simulation

CSF/MVerse coupled (s, Theta, c) model on a 2D periodic square lattice.
Tests the gate-independence/multiplicative suppression claim by estimating:
    p_geo, p_neu, p_both, and rho = p_both/(p_geo*p_neu)

Model components:
- Ising spins s_i ∈ {+1, -1}
- Edge phases theta_ij ∈ (-π, π] with theta_ji = -theta_ij
- Credit field c_i ∈ R (explicit eta surrogate)

Author: Monte Carlo Implementation for CSF/MVerse Two-Gate Model
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import json
import csv
from multiprocessing import Pool, cpu_count
import time


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


@dataclass
class SimulationParams:
    """All simulation parameters."""
    L: int = 16              # Lattice size (reduced for speed)
    T: float = 4.0           # Temperature (high enough for spin fluctuations)
    J: float = 0.3           # Ising coupling (weak for fluctuating magnetization)
    h: float = 0.0           # External field
    kappa: float = 50.0      # Holonomy closure strength (high for small holonomies)
    g: float = 0.5           # Spin-phase coupling
    mu: float = 1.0          # Credit-spin coupling
    gamma: float = 0.5       # Credit-holonomy coupling
    sigma: float = 1.0       # Credit field variance scale
    dtheta: float = 0.3      # Phase update step
    delta: float = 1.0       # Geometric gate tolerance (based on max|Φ| distribution)
    m_tol: float = 0.10      # Neutrality gate tolerance
    n_burnin: int = 300      # Burn-in sweeps
    n_sample: int = 600      # Sampling sweeps
    n_batches: int = 20      # Number of batches for error estimation

    @property
    def N(self) -> int:
        return self.L * self.L


class TwoGateLattice:
    """
    2D periodic square lattice with spins, edge phases, and credit fields.

    Coordinates: site (x, y) with x, y ∈ {0, ..., L-1}
    Site index: i = x + y * L

    Edges: stored as directed pairs, but theta_ji = -theta_ij enforced.
    We store theta for edges (i, j) where i < j (canonical ordering).

    Plaquettes: elementary 4-cycles (squares).
    Plaquette at (x, y) uses vertices:
        (x, y), (x+1, y), (x+1, y+1), (x, y+1)
    """

    def __init__(self, params: SimulationParams, rng: np.random.Generator = None):
        self.params = params
        self.L = params.L
        self.N = params.N
        self.rng = rng if rng is not None else np.random.default_rng()

        # State variables
        self.spins = np.zeros(self.N, dtype=np.int8)
        self.credits = np.zeros(self.N, dtype=np.float64)

        # Edge phases: indexed by canonical edge (smaller index first)
        # For edge (i, j) with i < j, theta[edge_idx] = theta_ij
        self._setup_edges()
        self._setup_plaquettes()

        # Initialize state
        self.initialize_random()

        # Cached quantities
        self._phi_cache = np.zeros(self.n_plaquettes, dtype=np.float64)
        self._q_cache = np.zeros(self.N, dtype=np.float64)
        self._update_all_caches()

    def _setup_edges(self):
        """Setup edge indexing for horizontal and vertical edges."""
        L = self.L

        # Horizontal edges: (x, y) -- (x+1, y)
        # Vertical edges: (x, y) -- (x, y+1)
        # Total: 2 * L^2 edges

        self.n_edges = 2 * self.N
        self.theta = np.zeros(self.n_edges, dtype=np.float64)

        # Edge index: horizontal edges first, then vertical
        # Horizontal edge at (x, y): index = x + y*L
        # Vertical edge at (x, y): index = N + x + y*L

        # For each edge, store the two endpoint indices
        self.edge_endpoints = np.zeros((self.n_edges, 2), dtype=np.int32)

        for y in range(L):
            for x in range(L):
                i = x + y * L

                # Horizontal edge (i, i_right)
                x_right = (x + 1) % L
                i_right = x_right + y * L
                h_edge = i  # horizontal edge index
                self.edge_endpoints[h_edge] = [i, i_right]

                # Vertical edge (i, i_up)
                y_up = (y + 1) % L
                i_up = x + y_up * L
                v_edge = self.N + i  # vertical edge index
                self.edge_endpoints[v_edge] = [i, i_up]

        # For each site, list of incident edges and their orientation
        # orientation = +1 if site is first endpoint, -1 if second
        self.site_edges = [[] for _ in range(self.N)]
        for e_idx in range(self.n_edges):
            i, j = self.edge_endpoints[e_idx]
            self.site_edges[i].append((e_idx, +1))
            self.site_edges[j].append((e_idx, -1))

    def _setup_plaquettes(self):
        """Setup plaquette data structures."""
        L = self.L
        self.n_plaquettes = self.N  # One plaquette per site

        # Each plaquette stores: 4 edges with orientations
        # Plaquette at (x, y): goes around i -> i_right -> i_diag -> i_up -> i
        # Edges: right (h at (x,y)), up (v at (x+1,y)), left (h at (x,y+1)), down (v at (x,y))

        self.plaquette_edges = []  # List of (edge_idx, orientation) for each plaquette
        self.plaquette_vertices = []  # List of 4 vertex indices

        for y in range(L):
            for x in range(L):
                i = x + y * L
                x_right = (x + 1) % L
                y_up = (y + 1) % L

                i_right = x_right + y * L
                i_up = x + y_up * L
                i_diag = x_right + y_up * L

                # Edge indices and orientations for CCW traversal
                h_bottom = i                    # (i -> i_right), orientation +1
                v_right = self.N + i_right      # (i_right -> i_diag), orientation +1
                h_top = i_up                    # (i_diag -> i_up), orientation -1 (reverse)
                v_left = self.N + i             # (i_up -> i), orientation -1 (reverse)

                edges = [
                    (h_bottom, +1),
                    (v_right, +1),
                    (h_top, -1),
                    (v_left, -1)
                ]

                self.plaquette_edges.append(edges)
                self.plaquette_vertices.append([i, i_right, i_diag, i_up])

        # For each edge, list of plaquettes it belongs to
        self.edge_plaquettes = [[] for _ in range(self.n_edges)]
        for p_idx, edges in enumerate(self.plaquette_edges):
            for e_idx, _ in edges:
                self.edge_plaquettes[e_idx].append(p_idx)

        # For each site, list of plaquettes containing it
        self.site_plaquettes = [[] for _ in range(self.N)]
        for p_idx, verts in enumerate(self.plaquette_vertices):
            for v in verts:
                self.site_plaquettes[v].append(p_idx)

    def get_theta(self, e_idx: int, orientation: int) -> float:
        """Get oriented phase for edge."""
        return orientation * self.theta[e_idx]

    def compute_phi(self, p_idx: int) -> float:
        """Compute holonomy for plaquette p_idx."""
        total = 0.0
        for e_idx, orient in self.plaquette_edges[p_idx]:
            total += self.get_theta(e_idx, orient)
        return wrap_angle(total)

    def _update_phi_cache(self, p_idx: int):
        """Update cached holonomy for one plaquette."""
        self._phi_cache[p_idx] = self.compute_phi(p_idx)

    def _update_q_cache(self, site: int):
        """Update cached q_i for one site."""
        q = 0.0
        for p_idx in self.site_plaquettes[site]:
            q += 1.0 - np.cos(self._phi_cache[p_idx])
        self._q_cache[site] = q

    def _update_all_caches(self):
        """Recompute all cached quantities."""
        for p_idx in range(self.n_plaquettes):
            self._update_phi_cache(p_idx)
        for i in range(self.N):
            self._update_q_cache(i)

    def initialize_random(self):
        """Random initialization of all state variables."""
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(np.int8)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.n_edges)
        self.credits = self.rng.normal(0, self.params.sigma, size=self.N)

    # === Energy computations ===

    def H_s(self) -> float:
        """Ising sector energy: -J sum_{<i,j>} s_i s_j - h sum_i s_i."""
        L = self.L
        p = self.params

        # Nearest neighbor sum
        energy = 0.0
        for e_idx in range(self.n_edges):
            i, j = self.edge_endpoints[e_idx]
            energy -= p.J * self.spins[i] * self.spins[j]

        # External field
        energy -= p.h * np.sum(self.spins)
        return energy

    def H_Theta(self) -> float:
        """Holonomy closure energy: kappa * sum_C (1 - cos Phi_C)."""
        return self.params.kappa * np.sum(1.0 - np.cos(self._phi_cache))

    def H_sTheta(self) -> float:
        """Spin-phase coupling: -g * sum_{<i,j>} P_ij(s) * cos(theta_ij)."""
        p = self.params
        energy = 0.0
        for e_idx in range(self.n_edges):
            i, j = self.edge_endpoints[e_idx]
            P_ij = ((1 + self.spins[i]) / 2) * ((1 + self.spins[j]) / 2)
            energy -= p.g * P_ij * np.cos(self.theta[e_idx])
        return energy

    def H_c(self) -> float:
        """Credit field energy: sum_i [c_i^2/(2*sigma^2) - mu*c_i*s_i + gamma*c_i*q_i]."""
        p = self.params
        c = self.credits
        s = self.spins
        q = self._q_cache

        return np.sum(
            c**2 / (2 * p.sigma**2)
            - p.mu * c * s
            + p.gamma * c * q
        )

    def H_c_effective(self) -> float:
        """
        Integrated-out credit field energy (for validation).
        c_i* = sigma^2 * (mu*s_i - gamma*q_i)
        H_c_eff = -(sigma^2/2) * sum_i (mu*s_i - gamma*q_i)^2
        """
        p = self.params
        s = self.spins.astype(np.float64)
        q = self._q_cache
        return -0.5 * p.sigma**2 * np.sum((p.mu * s - p.gamma * q)**2)

    def total_energy(self) -> float:
        """Total Hamiltonian H = H_s + H_Theta + H_sTheta + H_c."""
        return self.H_s() + self.H_Theta() + self.H_sTheta() + self.H_c()

    # === Local energy changes for Metropolis ===

    def delta_H_spin_flip(self, site: int) -> float:
        """
        Compute energy change for flipping spin at site.
        Affected terms:
        - H_s: neighbor interactions and external field
        - H_sTheta: edge coupling terms for incident edges
        - H_c: the c_i term at this site
        """
        p = self.params
        s_old = self.spins[site]
        s_new = -s_old

        # H_s change: Ising term
        delta_Hs = 0.0
        for e_idx, orient in self.site_edges[site]:
            i, j = self.edge_endpoints[e_idx]
            neighbor = j if i == site else i
            # Old contribution: -J * s_old * s_neighbor
            # New contribution: -J * s_new * s_neighbor
            delta_Hs += -p.J * (s_new - s_old) * self.spins[neighbor]

        # External field
        delta_Hs += -p.h * (s_new - s_old)

        # H_sTheta change: edges incident to site
        delta_HsT = 0.0
        for e_idx, orient in self.site_edges[site]:
            i, j = self.edge_endpoints[e_idx]
            other = j if i == site else i
            s_other = self.spins[other]

            P_old = ((1 + s_old) / 2) * ((1 + s_other) / 2)
            P_new = ((1 + s_new) / 2) * ((1 + s_other) / 2)

            cos_theta = np.cos(self.theta[e_idx])
            delta_HsT += -p.g * (P_new - P_old) * cos_theta

        # H_c change at this site
        c_i = self.credits[site]
        q_i = self._q_cache[site]
        # H_c for site: c_i^2/(2sigma^2) - mu*c_i*s_i + gamma*c_i*q_i
        # Only -mu*c_i*s_i changes
        delta_Hc = -p.mu * c_i * (s_new - s_old)

        return delta_Hs + delta_HsT + delta_Hc

    def delta_H_edge_update(self, e_idx: int, theta_new: float) -> float:
        """
        Compute energy change for updating edge phase.
        Affected terms:
        - H_Theta: plaquettes containing this edge
        - H_sTheta: this edge's coupling term
        - H_c: through q_i for vertices of affected plaquettes
        """
        p = self.params
        theta_old = self.theta[e_idx]

        # H_sTheta change for this edge
        i, j = self.edge_endpoints[e_idx]
        P_ij = ((1 + self.spins[i]) / 2) * ((1 + self.spins[j]) / 2)
        delta_HsT = -p.g * P_ij * (np.cos(theta_new) - np.cos(theta_old))

        # Find affected plaquettes
        affected_plaquettes = self.edge_plaquettes[e_idx]

        # Compute new phi values for affected plaquettes
        old_phis = [self._phi_cache[p_idx] for p_idx in affected_plaquettes]

        # Temporarily update theta to compute new phis
        self.theta[e_idx] = theta_new
        new_phis = [self.compute_phi(p_idx) for p_idx in affected_plaquettes]
        self.theta[e_idx] = theta_old  # Restore

        # H_Theta change
        delta_HT = 0.0
        for old_phi, new_phi in zip(old_phis, new_phis):
            delta_HT += p.kappa * ((1 - np.cos(new_phi)) - (1 - np.cos(old_phi)))

        # H_c change through q_i
        # q_i = sum over plaquettes containing i of (1 - cos Phi)
        # Affected sites: all vertices of affected plaquettes
        affected_sites = set()
        for p_idx in affected_plaquettes:
            affected_sites.update(self.plaquette_vertices[p_idx])

        delta_Hc = 0.0
        for site in affected_sites:
            c_i = self.credits[site]

            # Compute old and new q_i
            q_old = self._q_cache[site]

            # Compute delta_q from affected plaquettes containing this site
            delta_q = 0.0
            for p_idx in self.site_plaquettes[site]:
                if p_idx in affected_plaquettes:
                    idx = affected_plaquettes.index(p_idx)
                    delta_q += (1 - np.cos(new_phis[idx])) - (1 - np.cos(old_phis[idx]))

            # H_c term affected: gamma * c_i * q_i
            delta_Hc += p.gamma * c_i * delta_q

        return delta_HT + delta_HsT + delta_Hc

    # === Monte Carlo updates ===

    def update_spin(self, site: int) -> bool:
        """Metropolis update for single spin."""
        delta_H = self.delta_H_spin_flip(site)

        if delta_H <= 0 or self.rng.random() < np.exp(-delta_H / self.params.T):
            self.spins[site] *= -1
            return True
        return False

    def update_edge(self, e_idx: int) -> bool:
        """Metropolis update for single edge phase."""
        theta_old = self.theta[e_idx]
        delta = self.rng.uniform(-self.params.dtheta, self.params.dtheta)
        theta_new = wrap_angle(theta_old + delta)

        delta_H = self.delta_H_edge_update(e_idx, theta_new)

        if delta_H <= 0 or self.rng.random() < np.exp(-delta_H / self.params.T):
            self.theta[e_idx] = theta_new

            # Update caches for affected plaquettes and sites
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
        """
        Gibbs sampling for credit field at site.
        H_c contribution at site: c^2/(2*sigma^2) - c*(mu*s - gamma*q)
        Conditional: Normal(mean = sigma^2*(mu*s - gamma*q), var = sigma^2*T)

        Derivation:
        P(c|s,Theta) ∝ exp(-[c^2/(2σ²) - c(μs - γq)]/T)
        = exp(-[c² - 2σ²c(μs - γq)]/(2σ²T))
        = exp(-(c - σ²(μs - γq))²/(2σ²T)) * const

        So: mean = σ²(μs - γq), variance = σ²T
        """
        p = self.params
        s_i = self.spins[site]
        q_i = self._q_cache[site]

        mean = p.sigma**2 * (p.mu * s_i - p.gamma * q_i)
        std = p.sigma * np.sqrt(p.T)

        self.credits[site] = self.rng.normal(mean, std)

    def sweep(self):
        """One full sweep: update all spins, edges, and credits."""
        # Random order for spins
        spin_order = self.rng.permutation(self.N)
        for site in spin_order:
            self.update_spin(site)

        # Random order for edges
        edge_order = self.rng.permutation(self.n_edges)
        for e_idx in edge_order:
            self.update_edge(e_idx)

        # Gibbs update for credits (order doesn't matter for Gibbs)
        for site in range(self.N):
            self.update_credit_gibbs(site)

    # === Observables ===

    def magnetization(self) -> float:
        """Compute m = (1/N) sum_i s_i."""
        return np.mean(self.spins)

    def max_holonomy(self) -> float:
        """Compute max_C |Phi_C|."""
        return np.max(np.abs(self._phi_cache))

    def mean_holonomy_magnitude(self) -> float:
        """Compute mean |Phi_C|."""
        return np.mean(np.abs(self._phi_cache))

    def geometric_gate(self) -> bool:
        """G_geo = 1 iff max_C |Phi_C| <= delta."""
        return self.max_holonomy() <= self.params.delta

    def neutrality_gate(self) -> bool:
        """G_neu = 1 iff |m| <= m_tol."""
        return np.abs(self.magnetization()) <= self.params.m_tol

    def both_gates(self) -> bool:
        """G_both = 1 iff G_geo=1 and G_neu=1."""
        return self.geometric_gate() and self.neutrality_gate()

    def measure(self) -> Dict:
        """Measure all observables."""
        return {
            'm': self.magnetization(),
            'max_phi': self.max_holonomy(),
            'mean_phi': self.mean_holonomy_magnitude(),
            'G_geo': 1 if self.geometric_gate() else 0,
            'G_neu': 1 if self.neutrality_gate() else 0,
            'G_both': 1 if self.both_gates() else 0,
            'H_total': self.total_energy(),
        }


def run_simulation(params: SimulationParams, seed: int = None, verbose: bool = False) -> Dict:
    """
    Run full simulation with burn-in and sampling.
    Returns statistics of gate probabilities and observables.
    """
    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    # Burn-in
    if verbose:
        print(f"  Burn-in: {params.n_burnin} sweeps...")
    for i in range(params.n_burnin):
        lattice.sweep()
        if verbose and (i + 1) % 50 == 0:
            print(f"    Burn-in sweep {i+1}/{params.n_burnin}")

    # Sampling
    if verbose:
        print(f"  Sampling: {params.n_sample} sweeps...")

    measurements = []
    for i in range(params.n_sample):
        lattice.sweep()
        measurements.append(lattice.measure())
        if verbose and (i + 1) % 100 == 0:
            print(f"    Sample sweep {i+1}/{params.n_sample}")

    # Compute statistics
    G_geo = np.array([m['G_geo'] for m in measurements])
    G_neu = np.array([m['G_neu'] for m in measurements])
    G_both = np.array([m['G_both'] for m in measurements])
    m_vals = np.array([m['m'] for m in measurements])
    mean_phi_vals = np.array([m['mean_phi'] for m in measurements])

    p_geo = np.mean(G_geo)
    p_neu = np.mean(G_neu)
    p_both = np.mean(G_both)

    # Rho with safe division
    product = p_geo * p_neu
    if product > 1e-10:
        rho = p_both / product
    else:
        rho = np.nan

    # Batch error estimation
    batch_size = params.n_sample // params.n_batches

    def batch_means(arr):
        batches = arr[:batch_size * params.n_batches].reshape(params.n_batches, batch_size)
        batch_vals = np.mean(batches, axis=1)
        return np.mean(batch_vals), np.std(batch_vals) / np.sqrt(params.n_batches)

    p_geo_mean, p_geo_err = batch_means(G_geo)
    p_neu_mean, p_neu_err = batch_means(G_neu)
    p_both_mean, p_both_err = batch_means(G_both)

    # Compute rho error via batch means
    batch_size = params.n_sample // params.n_batches
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
    if len(rho_batches) > 1:
        rho_err = np.std(rho_batches) / np.sqrt(len(rho_batches))
    else:
        rho_err = np.nan

    return {
        'p_geo': p_geo,
        'p_geo_err': p_geo_err,
        'p_neu': p_neu,
        'p_neu_err': p_neu_err,
        'p_both': p_both,
        'p_both_err': p_both_err,
        'rho': rho,
        'rho_err': rho_err,
        'mean_m': np.mean(m_vals),
        'mean_phi': np.mean(mean_phi_vals),
        'params': params,
        'phi_samples': lattice._phi_cache.copy(),
        'm_samples': m_vals,
    }


def run_single_point(args):
    """Wrapper for parallel execution."""
    params, seed = args
    return run_simulation(params, seed=seed, verbose=False)


def run_parameter_sweep(base_params: SimulationParams,
                        sweep_name: str,
                        sweep_values: List,
                        n_seeds: int = 1) -> List[Dict]:
    """Run parameter sweep with optional multiple seeds for statistics."""
    results = []

    print(f"\n=== Parameter sweep: {sweep_name} ===")

    for val in sweep_values:
        params = SimulationParams(**{k: v for k, v in base_params.__dict__.items()})
        setattr(params, sweep_name, val)

        print(f"\n{sweep_name} = {val}")

        if n_seeds == 1:
            result = run_simulation(params, seed=42, verbose=True)
        else:
            # Multiple seeds for better statistics
            seeds = list(range(n_seeds))
            args_list = [(params, s) for s in seeds]

            with Pool(min(n_seeds, cpu_count())) as pool:
                seed_results = pool.map(run_single_point, args_list)

            # Average over seeds
            result = {
                'p_geo': np.mean([r['p_geo'] for r in seed_results]),
                'p_neu': np.mean([r['p_neu'] for r in seed_results]),
                'p_both': np.mean([r['p_both'] for r in seed_results]),
                'rho': np.nanmean([r['rho'] for r in seed_results]),
                'mean_m': np.mean([r['mean_m'] for r in seed_results]),
                'mean_phi': np.mean([r['mean_phi'] for r in seed_results]),
                'params': params,
            }

        result['sweep_param'] = sweep_name
        result['sweep_value'] = val
        results.append(result)

        print(f"  p_geo={result['p_geo']:.4f}, p_neu={result['p_neu']:.4f}, "
              f"p_both={result['p_both']:.4f}, rho={result['rho']:.4f}")

    return results


def validate_sampler(params: SimulationParams, seed: int = 123):
    """
    Validation checks:
    1. Compare explicit-c sampler vs integrated-out energy
    2. Check acceptance rates
    3. Diagnostic output for delta_H values
    """
    print("\n=== Validation Checks ===")
    print(f"Parameters: T={params.T}, J={params.J}, kappa={params.kappa}, g={params.g}")

    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    # Burn-in
    print("Running 100 burn-in sweeps...")
    for _ in range(100):
        lattice.sweep()

    # Diagnostic: credit field statistics
    print(f"\nCredit field statistics:")
    print(f"  mean(|c|) = {np.mean(np.abs(lattice.credits)):.4f}")
    print(f"  max(|c|) = {np.max(np.abs(lattice.credits)):.4f}")
    print(f"  std(c) = {np.std(lattice.credits):.4f}")

    # Diagnostic: q_i statistics
    print(f"\nHolonomy q_i statistics:")
    print(f"  mean(q) = {np.mean(lattice._q_cache):.4f}")
    print(f"  max(q) = {np.max(lattice._q_cache):.4f}")

    # Diagnostic: spin and magnetization
    m = np.mean(lattice.spins)
    print(f"\nMagnetization: m = {m:.4f}")

    # Sample delta_H values for spin flips
    delta_Hs = []
    for site in rng.choice(lattice.N, size=min(200, lattice.N), replace=False):
        delta_Hs.append(lattice.delta_H_spin_flip(site))

    delta_Hs = np.array(delta_Hs)
    print(f"\nDelta_H for spin flips (200 samples):")
    print(f"  min = {np.min(delta_Hs):.4f}")
    print(f"  median = {np.median(delta_Hs):.4f}")
    print(f"  max = {np.max(delta_Hs):.4f}")
    print(f"  mean = {np.mean(delta_Hs):.4f}")

    # Compare H_c vs H_c_eff at optimal c
    H_c_explicit = lattice.H_c()

    # Set c to optimal values
    old_credits = lattice.credits.copy()
    for i in range(lattice.N):
        s_i = lattice.spins[i]
        q_i = lattice._q_cache[i]
        lattice.credits[i] = params.sigma**2 * (params.mu * s_i - params.gamma * q_i)

    H_c_at_optimal = lattice.H_c()
    H_c_eff = lattice.H_c_effective()

    print(f"\nEnergy comparisons:")
    print(f"  H_c (explicit, random c): {H_c_explicit:.4f}")
    print(f"  H_c (at c=c*): {H_c_at_optimal:.4f}")
    print(f"  H_c_eff (integrated out): {H_c_eff:.4f}")
    print(f"  Difference at c*: {abs(H_c_at_optimal - H_c_eff):.6f}")

    # Check Gibbs variance
    expected_var = params.sigma**2 * params.T
    print(f"\nGibbs sampling check:")
    print(f"  Expected c variance: sigma^2 * T = {expected_var:.4f}")

    # Restore credits
    lattice.credits = old_credits

    # Check acceptance rates by counting
    n_test = 100
    spin_accepts = 0
    edge_accepts = 0

    for _ in range(n_test):
        for site in range(lattice.N):
            if lattice.update_spin(site):
                spin_accepts += 1
        for e_idx in range(lattice.n_edges):
            if lattice.update_edge(e_idx):
                edge_accepts += 1
        for site in range(lattice.N):
            lattice.update_credit_gibbs(site)

    spin_rate = spin_accepts / (n_test * lattice.N)
    edge_rate = edge_accepts / (n_test * lattice.n_edges)

    print(f"\nAcceptance rates over {n_test} sweeps:")
    print(f"  Spin flip: {spin_rate:.3f}")
    print(f"  Edge phase: {edge_rate:.3f}")
    print("  Credit (Gibbs): 1.000 (exact sampling)")

    return True


def create_figures(all_results: Dict, output_dir: str = "."):
    """Create and save all figures."""

    # 1.png: Rho vs kappa (with error bars if available)
    if 'kappa_sweep' in all_results:
        kappa_results = all_results['kappa_sweep']
        kappas = [r['sweep_value'] for r in kappa_results]
        rhos = [r['rho'] for r in kappa_results]
        rho_errs = [r.get('rho_err', 0) for r in kappa_results]

        plt.figure(figsize=(8, 6))
        if any(e > 0 for e in rho_errs):
            plt.errorbar(kappas, rhos, yerr=rho_errs, fmt='bo-', markersize=8,
                        linewidth=2, capsize=4, capthick=2)
        else:
            plt.plot(kappas, rhos, 'bo-', markersize=8, linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='ρ=1 (independence)')
        plt.xlabel('κ (holonomy closure strength)', fontsize=12)
        plt.ylabel('ρ = p_both / (p_geo × p_neu)', fontsize=12)
        plt.title('Gate Correlation vs Holonomy Strength', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/1.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/1.png")

    # 2.png: p_both vs p_geo * p_neu scatter with y=x reference
    all_points = []
    for sweep_name in ['kappa_sweep', 'g_sweep', 'threshold_sweep']:
        if sweep_name in all_results:
            for r in all_results[sweep_name]:
                all_points.append({
                    'p_geo': r['p_geo'],
                    'p_neu': r['p_neu'],
                    'p_both': r['p_both'],
                    'sweep': sweep_name
                })

    if all_points:
        plt.figure(figsize=(8, 8))

        colors = {'kappa_sweep': 'blue', 'g_sweep': 'green', 'threshold_sweep': 'orange'}
        labels = {'kappa_sweep': 'κ sweep', 'g_sweep': 'g sweep', 'threshold_sweep': 'threshold sweep'}

        for sweep in ['kappa_sweep', 'g_sweep', 'threshold_sweep']:
            points = [p for p in all_points if p['sweep'] == sweep]
            if points:
                products = [p['p_geo'] * p['p_neu'] for p in points]
                p_boths = [p['p_both'] for p in points]
                plt.scatter(products, p_boths, c=colors[sweep], s=100, alpha=0.7, label=labels[sweep])

        # y=x line
        max_val = max(max([p['p_geo'] * p['p_neu'] for p in all_points]),
                      max([p['p_both'] for p in all_points]))
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (independence)')

        plt.xlabel('p_geo × p_neu', fontsize=12)
        plt.ylabel('p_both', fontsize=12)
        plt.title('Gate Independence Test', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/2.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/2.png")

    # 3.png: Heatmap of rho in (kappa, g) plane
    if 'heatmap_data' in all_results:
        hm = all_results['heatmap_data']

        plt.figure(figsize=(8, 6))
        im = plt.imshow(hm['rho_grid'], origin='lower', aspect='auto',
                       extent=[hm['g_values'][0], hm['g_values'][-1],
                              hm['kappa_values'][0], hm['kappa_values'][-1]],
                       cmap='RdBu_r', vmin=0.5, vmax=1.5)
        plt.colorbar(im, label='ρ')
        plt.xlabel('g (spin-phase coupling)', fontsize=12)
        plt.ylabel('κ (holonomy strength)', fontsize=12)
        plt.title('ρ in (κ, g) Parameter Space', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/3.png")

    # 4.png: Histograms (|Φ| distribution + m distribution) for representative point
    if 'representative' in all_results:
        rep = all_results['representative']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Holonomy histogram (|Φ|)
        phi_samples = rep.get('phi_samples', [])
        if len(phi_samples) > 0:
            abs_phi = np.abs(phi_samples)
            axes[0].hist(abs_phi, bins=50, density=True, alpha=0.7, color='blue')
            axes[0].axvline(x=rep['params'].delta, color='r', linestyle='--',
                           label=f'δ={rep["params"].delta}')
            axes[0].set_xlabel('|Φ_C| (plaquette holonomy magnitude)', fontsize=12)
            axes[0].set_ylabel('Density', fontsize=12)
            axes[0].set_title('Holonomy Magnitude Distribution', fontsize=14)
            axes[0].legend()

        # Magnetization histogram
        m_samples = rep.get('m_samples', [])
        if len(m_samples) > 0:
            axes[1].hist(m_samples, bins=50, density=True, alpha=0.7, color='green')
            axes[1].axvline(x=rep['params'].m_tol, color='r', linestyle='--',
                           label=f'm_tol={rep["params"].m_tol}')
            axes[1].axvline(x=-rep['params'].m_tol, color='r', linestyle='--')
            axes[1].set_xlabel('m (magnetization)', fontsize=12)
            axes[1].set_ylabel('Density', fontsize=12)
            axes[1].set_title('Magnetization Distribution', fontsize=14)
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/4.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/4.png")


def save_results_csv(all_results: Dict, filename: str):
    """Save results to CSV with requested column format."""
    rows = []

    # Collect all results from sweeps
    for sweep_name in ['kappa_sweep', 'g_sweep', 'threshold_sweep']:
        if sweep_name in all_results:
            for r in all_results[sweep_name]:
                p = r['params']
                rows.append({
                    'L': p.L,
                    'T': p.T,
                    'J': p.J,
                    'h': p.h,
                    'kappa': p.kappa,
                    'g': p.g,
                    'mu': p.mu,
                    'gamma': p.gamma,
                    'sigma': p.sigma,
                    'delta': p.delta,
                    'm_tol': p.m_tol,
                    'p_geo': r['p_geo'],
                    'p_neu': r['p_neu'],
                    'p_both': r['p_both'],
                    'rho': r['rho'],
                    'mean_abs_phi': r['mean_phi'],
                    'mean_m': r['mean_m'],
                    'stderr_rho': r.get('rho_err', ''),
                })

    # Add representative point if present
    if 'representative' in all_results:
        r = all_results['representative']
        p = r['params']
        rows.append({
            'L': p.L,
            'T': p.T,
            'J': p.J,
            'h': p.h,
            'kappa': p.kappa,
            'g': p.g,
            'mu': p.mu,
            'gamma': p.gamma,
            'sigma': p.sigma,
            'delta': p.delta,
            'm_tol': p.m_tol,
            'p_geo': r['p_geo'],
            'p_neu': r['p_neu'],
            'p_both': r['p_both'],
            'rho': r['rho'],
            'mean_abs_phi': r['mean_phi'],
            'mean_m': r['mean_m'],
            'stderr_rho': r.get('rho_err', ''),
        })

    if rows:
        fieldnames = ['L', 'T', 'J', 'h', 'kappa', 'g', 'mu', 'gamma', 'sigma',
                      'delta', 'm_tol', 'p_geo', 'p_neu', 'p_both', 'rho',
                      'mean_abs_phi', 'mean_m', 'stderr_rho']
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {filename}")


def main():
    """Main execution."""
    print("=" * 60)
    print("Two-Gate Interface Model Monte Carlo Simulation")
    print("=" * 60)

    start_time = time.time()

    # Base parameters
    base_params = SimulationParams()

    # Validation
    validate_sampler(base_params)

    all_results = {}

    # Sweep 1: kappa
    print("\n" + "=" * 60)
    print("SWEEP 1: Varying κ (holonomy closure strength)")
    print("=" * 60)
    kappa_values = [20.0, 35.0, 50.0, 75.0, 100.0]
    all_results['kappa_sweep'] = run_parameter_sweep(base_params, 'kappa', kappa_values)

    # Sweep 2: g
    print("\n" + "=" * 60)
    print("SWEEP 2: Varying g (spin-phase coupling)")
    print("=" * 60)
    g_values = [0.0, 0.5, 1.0, 2.0]
    all_results['g_sweep'] = run_parameter_sweep(base_params, 'g', g_values)

    # Sweep 3: thresholds (delta and m_tol combinations)
    print("\n" + "=" * 60)
    print("SWEEP 3: Varying gate thresholds")
    print("=" * 60)

    threshold_results = []
    delta_values = [0.8, 1.0, 1.2]
    m_tol_values = [0.08, 0.12, 0.18]

    for delta in delta_values:
        for m_tol in m_tol_values:
            params = SimulationParams(delta=delta, m_tol=m_tol)
            print(f"\nδ={delta}, m_tol={m_tol}")
            result = run_simulation(params, seed=42, verbose=True)
            result['sweep_param'] = 'delta_mtol'
            result['sweep_value'] = f'{delta}_{m_tol}'
            threshold_results.append(result)
            print(f"  p_geo={result['p_geo']:.4f}, p_neu={result['p_neu']:.4f}, "
                  f"p_both={result['p_both']:.4f}, rho={result['rho']:.4f}")

    all_results['threshold_sweep'] = threshold_results

    # Representative point with stored samples
    print("\n" + "=" * 60)
    print("Running representative simulation for histograms...")
    print("=" * 60)
    all_results['representative'] = run_simulation(base_params, seed=42, verbose=True)

    # Optional: Heatmap data (smaller grid for speed)
    print("\n" + "=" * 60)
    print("Generating (κ, g) heatmap data...")
    print("=" * 60)

    kappa_hm = [30.0, 50.0, 75.0]
    g_hm = [0.0, 0.5, 1.0, 2.0]
    rho_grid = np.zeros((len(kappa_hm), len(g_hm)))

    for i, kappa in enumerate(kappa_hm):
        for j, g in enumerate(g_hm):
            params = SimulationParams(kappa=kappa, g=g)
            # Shorter runs for heatmap
            params.n_burnin = 100
            params.n_sample = 400
            result = run_simulation(params, seed=42, verbose=False)
            rho_grid[i, j] = result['rho']
            print(f"  κ={kappa}, g={g}: ρ={result['rho']:.4f}")

    all_results['heatmap_data'] = {
        'kappa_values': kappa_hm,
        'g_values': g_hm,
        'rho_grid': rho_grid
    }

    # Create output directory
    import os
    output_dir = 'results/two_gate_sim'
    os.makedirs(output_dir, exist_ok=True)

    # Create figures
    print("\n" + "=" * 60)
    print("Creating figures...")
    print("=" * 60)
    create_figures(all_results, output_dir)

    # Save results
    save_results_csv(all_results, f'{output_dir}/results.csv')

    # Save full results as JSON (excluding large arrays)
    json_results = {}
    for key in ['kappa_sweep', 'g_sweep', 'threshold_sweep']:
        if key in all_results:
            json_results[key] = []
            for r in all_results[key]:
                json_results[key].append({
                    'sweep_param': r.get('sweep_param'),
                    'sweep_value': r.get('sweep_value'),
                    'p_geo': float(r['p_geo']),
                    'p_neu': float(r['p_neu']),
                    'p_both': float(r['p_both']),
                    'rho': float(r['rho']) if not np.isnan(r['rho']) else None,
                    'mean_m': float(r['mean_m']),
                    'mean_phi': float(r['mean_phi']),
                })

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {output_dir}/results.json")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nKey findings:")

    # Analyze kappa sweep
    if 'kappa_sweep' in all_results:
        rhos = [r['rho'] for r in all_results['kappa_sweep']]
        kappas = [r['sweep_value'] for r in all_results['kappa_sweep']]
        print(f"\nκ sweep: ρ ranges from {min(rhos):.3f} to {max(rhos):.3f}")

        close_to_1 = [k for k, r in zip(kappas, rhos) if abs(r - 1.0) < 0.1]
        if close_to_1:
            print(f"  ρ ≈ 1 (independence) at κ = {close_to_1}")

        deviating = [(k, r) for k, r in zip(kappas, rhos) if abs(r - 1.0) >= 0.1]
        if deviating:
            print(f"  ρ deviates from 1 at: {[(k, f'{r:.3f}') for k, r in deviating]}")

    # Identify nontrivial regimes
    print("\nNontrivial gate regimes (0.1 < p < 0.9):")
    for sweep_name in ['kappa_sweep', 'g_sweep']:
        if sweep_name in all_results:
            for r in all_results[sweep_name]:
                p_geo, p_neu = r['p_geo'], r['p_neu']
                if 0.1 < p_geo < 0.9 and 0.1 < p_neu < 0.9:
                    print(f"  {sweep_name}, {r['sweep_param']}={r['sweep_value']}: "
                          f"p_geo={p_geo:.3f}, p_neu={p_neu:.3f}, ρ={r['rho']:.3f}")


if __name__ == "__main__":
    main()
