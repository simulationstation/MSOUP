#!/usr/bin/env python3
"""
Phase 4 Option 1: Helicity modulus (stiffness) from edge cosine term only.
"""

import argparse
import contextlib
import math
import os
from dataclasses import dataclass, replace
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


@dataclass
class StiffnessParams:
    """Simulation parameters."""
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
    n_burnin: int = 600
    n_sample: int = 3000
    n_batches: int = 30

    @property
    def N(self) -> int:
        return self.L * self.L


class TwoGateLattice:
    """2D periodic square lattice with spins, edge phases, and credit fields."""

    def __init__(self, params: StiffnessParams, rng: np.random.Generator):
        self.params = params
        self.L = params.L
        self.N = params.N
        self.rng = rng

        self.spins = np.zeros(self.N, dtype=np.int8)
        self.credits = np.zeros(self.N, dtype=np.float64)

        self._setup_edges()
        self._setup_plaquettes()
        self.initialize_random()

        self._phi_cache = np.zeros(self.n_plaquettes, dtype=np.float64)
        self._q_cache = np.zeros(self.N, dtype=np.float64)
        self._update_all_caches()

        self.x_edge_indices = np.arange(self.N, dtype=np.int32)
        self.x_edge_endpoints = self.edge_endpoints[self.x_edge_indices]

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
                    (v_left, -1),
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

    def initialize_random(self):
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(np.int8)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.n_edges)
        self.credits = self.rng.normal(0, self.params.sigma, size=self.N)

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

    def delta_H_spin_flip(self, site: int) -> float:
        p = self.params
        s_old = self.spins[site]
        s_new = -s_old

        delta_Hs = 0.0
        for e_idx, _ in self.site_edges[site]:
            i, j = self.edge_endpoints[e_idx]
            neighbor = j if i == site else i
            delta_Hs += -p.J * (s_new - s_old) * self.spins[neighbor]

        delta_Hs += -p.h * (s_new - s_old)

        delta_HsT = 0.0
        for e_idx, _ in self.site_edges[site]:
            i, j = self.edge_endpoints[e_idx]
            other = j if i == site else i
            s_other = self.spins[other]

            p_old = ((1 + s_old) / 2) * ((1 + s_other) / 2)
            p_new = ((1 + s_new) / 2) * ((1 + s_other) / 2)

            cos_theta = np.cos(self.theta[e_idx])
            delta_HsT += -p.g * (p_new - p_old) * cos_theta

        c_i = self.credits[site]
        delta_Hc = -p.mu * c_i * (s_new - s_old)

        return delta_Hs + delta_HsT + delta_Hc

    def delta_H_edge_update(self, e_idx: int, theta_new: float) -> float:
        p = self.params
        theta_old = self.theta[e_idx]

        i, j = self.edge_endpoints[e_idx]
        p_ij = ((1 + self.spins[i]) / 2) * ((1 + self.spins[j]) / 2)
        delta_HsT = -p.g * p_ij * (np.cos(theta_new) - np.cos(theta_old))

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
        if delta_H <= 0 or self.rng.random() < math.exp(-delta_H / self.params.T):
            self.spins[site] *= -1
            return True
        return False

    def update_edge(self, e_idx: int) -> bool:
        theta_old = self.theta[e_idx]
        delta = self.rng.uniform(-self.params.dtheta, self.params.dtheta)
        theta_new = wrap_angle(theta_old + delta)

        delta_H = self.delta_H_edge_update(e_idx, theta_new)

        if delta_H <= 0 or self.rng.random() < math.exp(-delta_H / self.params.T):
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
        std = p.sigma * math.sqrt(p.T)

        self.credits[site] = self.rng.normal(mean, std)

    def sweep(self) -> Tuple[int, int]:
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
        return float(np.mean(self.spins))

    def mean_holonomy_magnitude(self) -> float:
        return float(np.mean(np.abs(self._phi_cache)))

    def measure_stiffness_x(self) -> Tuple[float, float, float]:
        spins = self.spins
        i_idx = self.x_edge_endpoints[:, 0]
        j_idx = self.x_edge_endpoints[:, 1]
        active = (spins[i_idx] == 1) & (spins[j_idx] == 1)
        p_ij = active.astype(np.float64)
        theta_x = self.theta[self.x_edge_indices]
        cos_theta = np.cos(theta_x)
        sin_theta = np.sin(theta_x)

        a_val = self.params.g * np.sum(p_ij * cos_theta)
        b_val = self.params.g * np.sum(p_ij * sin_theta)
        f_active_x = float(np.mean(p_ij))
        return float(a_val), float(b_val), f_active_x


def probe_magnetization(params: StiffnessParams, seed: int, n_burnin: int, n_sample: int) -> float:
    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    for _ in range(n_burnin):
        lattice.sweep()

    abs_m = []
    for _ in range(n_sample):
        lattice.sweep()
        abs_m.append(abs(lattice.magnetization()))

    return float(np.mean(abs_m))


def probe_kappa_stats(params: StiffnessParams, seed: int, n_burnin: int, n_sample: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    lattice = TwoGateLattice(params, rng)

    for _ in range(n_burnin):
        lattice.sweep()

    abs_phi_samples = []
    for _ in range(n_sample):
        lattice.sweep()
        abs_phi_samples.append(np.abs(lattice._phi_cache).copy())

    abs_phi = np.concatenate(abs_phi_samples)
    return {
        "q95": float(np.quantile(abs_phi, 0.95)),
        "median": float(np.median(abs_phi)),
        "max": float(np.max(abs_phi)),
    }


def screen_spin_order(base_params: StiffnessParams) -> Tuple[float, float]:
    params = replace(base_params, L=24, J=0.6, T=1.0)
    print("Spin-order screen: baseline J=0.6, T=1.0")
    mean_abs_m = probe_magnetization(params, seed=123, n_burnin=200, n_sample=400)
    print(f"  mean|m|={mean_abs_m:.3f}")
    if mean_abs_m > 0.5:
        params = replace(params, T=1.6)
        print("  Ordered: increasing T to 1.6")
        mean_abs_m = probe_magnetization(params, seed=124, n_burnin=200, n_sample=400)
        print(f"  mean|m|={mean_abs_m:.3f}")
        if mean_abs_m > 0.5:
            params = replace(params, J=0.4)
            print("  Still ordered: setting J=0.4")
            mean_abs_m = probe_magnetization(params, seed=125, n_burnin=200, n_sample=400)
            print(f"  mean|m|={mean_abs_m:.3f}")
    if mean_abs_m > 0.3:
        print("  Warning: mean|m| still > 0.3 after adjustments.")

    print(f"Spin-order screen final: J={params.J:.2f}, T={params.T:.2f}")
    return params.J, params.T


def screen_kappa(params: StiffnessParams) -> Tuple[List[float], Dict[float, Dict[str, float]]]:
    kappa_candidates = [20.0, 30.0, 40.0, 50.0, 65.0, 80.0, 100.0]
    usable = []
    stats_map: Dict[float, Dict[str, float]] = {}

    print("Kappa saturation screen:")
    for idx, kappa in enumerate(kappa_candidates):
        params_k = replace(params, kappa=min(kappa, 100.0))
        stats = probe_kappa_stats(params_k, seed=200 + idx, n_burnin=200, n_sample=400)
        stats_map[params_k.kappa] = stats
        q95 = stats["q95"]
        print(f"  κ={params_k.kappa:.1f}: q95={q95:.3f}, median={stats['median']:.3f}, max={stats['max']:.3f}")

        if 0.15 <= q95 <= 0.75:
            usable.append(params_k.kappa)
            if len(usable) >= 4:
                print("  Found >=4 usable κ values; stopping scan.")
                break

        if q95 < 0.15:
            print("  Saturation detected (q95 < 0.15); stopping scan.")
            break

    return usable, stats_map


def pick_kappa_values(usable: List[float]) -> Tuple[float, Optional[float], Optional[float]]:
    if not usable:
        raise RuntimeError("No usable kappa values found during screening.")

    usable_sorted = sorted(usable)
    if 50.0 in usable_sorted:
        kappa_working = 50.0
    else:
        kappa_working = min(usable_sorted, key=lambda x: abs(x - 50.0))

    idx = usable_sorted.index(kappa_working)
    kappa_low = usable_sorted[idx - 1] if idx - 1 >= 0 else None
    kappa_high = usable_sorted[idx + 1] if idx + 1 < len(usable_sorted) else None
    return kappa_working, kappa_low, kappa_high


def run_stiffness_task(task: Dict) -> Dict[str, float]:
    params: StiffnessParams = task["params"]
    seed: int = task["seed"]
    log_path: Path = task["log_path"]
    q95_abs_phi: Optional[float] = task.get("q95_abs_phi")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file, \
            contextlib.redirect_stdout(log_file), \
            contextlib.redirect_stderr(log_file):
        print(f"Task L={params.L}, kappa={params.kappa}, g={params.g}")
        print(f"Seed={seed}")

        rng = np.random.default_rng(seed)
        lattice = TwoGateLattice(params, rng)

        for _ in range(params.n_burnin):
            lattice.sweep()

        batch_size = params.n_sample // params.n_batches
        n_use = batch_size * params.n_batches

        if n_use == 0:
            raise ValueError("n_sample too small for batching.")

        guard_window = 300
        guard_active = n_use >= guard_window
        guard_spin_acc = 0
        guard_edge_acc = 0
        guard_f_active = 0.0

        batch_a = []
        batch_b2 = []
        batch_f = []

        total_spin_acc = 0
        total_edge_acc = 0
        sum_abs_phi = 0.0
        sum_m = 0.0

        sweep_in_batch = 0
        a_sum = 0.0
        b2_sum = 0.0
        f_sum = 0.0

        for sweep_idx in range(n_use):
            spin_acc, edge_acc = lattice.sweep()
            total_spin_acc += spin_acc
            total_edge_acc += edge_acc

            a_val, b_val, f_active_x = lattice.measure_stiffness_x()
            b2_val = b_val * b_val
            a_sum += a_val
            b2_sum += b2_val
            f_sum += f_active_x

            sum_abs_phi += lattice.mean_holonomy_magnitude()
            sum_m += lattice.magnetization()

            if guard_active and sweep_idx < guard_window:
                guard_spin_acc += spin_acc
                guard_edge_acc += edge_acc
                guard_f_active += f_active_x

            sweep_in_batch += 1
            if sweep_in_batch == batch_size:
                batch_a.append(a_sum / batch_size)
                batch_b2.append(b2_sum / batch_size)
                batch_f.append(f_sum / batch_size)
                sweep_in_batch = 0
                a_sum = 0.0
                b2_sum = 0.0
                f_sum = 0.0

            if guard_active and sweep_idx + 1 == guard_window:
                guard_accept_theta = guard_edge_acc / (guard_window * lattice.n_edges)
                guard_f_active_avg = guard_f_active / guard_window
                if guard_accept_theta < 0.05:
                    print(f"FAIL: accept_theta={guard_accept_theta:.4f} < 0.05")
                    return _fail_result(
                        params,
                        seed,
                        q95_abs_phi,
                        "low_accept_theta",
                        accept_spin=guard_spin_acc / (guard_window * lattice.N),
                        accept_theta=guard_accept_theta,
                        f_active_x=guard_f_active_avg,
                    )
                if guard_f_active_avg < 0.02:
                    print(f"FAIL: f_active_x={guard_f_active_avg:.4f} < 0.02")
                    return _fail_result(
                        params,
                        seed,
                        q95_abs_phi,
                        "low_f_active_x",
                        accept_spin=guard_spin_acc / (guard_window * lattice.N),
                        accept_theta=guard_accept_theta,
                        f_active_x=guard_f_active_avg,
                    )

        upsilon_batches = []
        f_batches = []
        for a_mean, b2_mean, f_mean in zip(batch_a, batch_b2, batch_f):
            upsilon = (a_mean / lattice.N) - (b2_mean / (lattice.N * params.T))
            upsilon_batches.append(upsilon)
            f_batches.append(f_mean)

        upsilon_mean = float(np.mean(upsilon_batches))
        upsilon_stderr = float(np.std(upsilon_batches, ddof=1) / math.sqrt(len(upsilon_batches)))
        f_active_mean = float(np.mean(f_batches))
        f_active_stderr = float(np.std(f_batches, ddof=1) / math.sqrt(len(f_batches)))

        accept_spin = total_spin_acc / (n_use * lattice.N)
        accept_theta = total_edge_acc / (n_use * lattice.n_edges)
        mean_abs_phi = sum_abs_phi / n_use
        mean_m = sum_m / n_use

        return {
            "status": "SUCCESS",
            "fail_reason": "",
            "L": params.L,
            "T": params.T,
            "J": params.J,
            "h": params.h,
            "kappa": params.kappa,
            "g": params.g,
            "mu": params.mu,
            "gamma": params.gamma,
            "sigma": params.sigma,
            "Upsilon_x": upsilon_mean,
            "stderr_Upsilon_x": upsilon_stderr,
            "f_active_x": f_active_mean,
            "stderr_f_active_x": f_active_stderr,
            "accept_spin": accept_spin,
            "accept_theta": accept_theta,
            "mean_abs_phi": mean_abs_phi,
            "mean_m": mean_m,
            "q95_abs_phi": q95_abs_phi if q95_abs_phi is not None else float("nan"),
            "seed": seed,
        }


def _fail_result(
    params: StiffnessParams,
    seed: int,
    q95_abs_phi: Optional[float],
    reason: str,
    accept_spin: float,
    accept_theta: float,
    f_active_x: float,
) -> Dict[str, float]:
    return {
        "status": "FAIL",
        "fail_reason": reason,
        "L": params.L,
        "T": params.T,
        "J": params.J,
        "h": params.h,
        "kappa": params.kappa,
        "g": params.g,
        "mu": params.mu,
        "gamma": params.gamma,
        "sigma": params.sigma,
        "Upsilon_x": float("nan"),
        "stderr_Upsilon_x": float("nan"),
        "f_active_x": f_active_x,
        "stderr_f_active_x": float("nan"),
        "accept_spin": accept_spin,
        "accept_theta": accept_theta,
        "mean_abs_phi": float("nan"),
        "mean_m": float("nan"),
        "q95_abs_phi": q95_abs_phi if q95_abs_phi is not None else float("nan"),
        "seed": seed,
    }


def build_tasks(
    params: StiffnessParams,
    l_list: List[int],
    kappa_working: float,
    kappa_low: Optional[float],
    kappa_high: Optional[float],
    g_sweep: List[float],
    kappa_sweep: List[float],
    base_seed: int,
    q95_lookup: Dict[float, float],
    log_dir: Path,
) -> List[Dict]:
    tasks = []
    for L in l_list:
        combos = set()
        combos.add((kappa_working, 0.5))
        for g_val in g_sweep:
            combos.add((kappa_working, g_val))
        for k_val in kappa_sweep:
            combos.add((k_val, 0.5))

        for kappa, g_val in sorted(combos):
            params_task = replace(params, L=L, kappa=kappa, g=g_val)
            seed = base_seed + 100000 * L + 1000 * int(kappa) + 10 * int(100 * g_val)
            log_path = log_dir / f"task_L{L}_k{int(kappa)}_g{g_val:.2f}.log"
            tasks.append({
                "params": params_task,
                "seed": seed,
                "log_path": log_path,
                "q95_abs_phi": q95_lookup.get(kappa),
            })

    return tasks


def save_results_csv(results: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "status",
        "fail_reason",
        "L",
        "T",
        "J",
        "h",
        "kappa",
        "g",
        "mu",
        "gamma",
        "sigma",
        "Upsilon_x",
        "stderr_Upsilon_x",
        "f_active_x",
        "stderr_f_active_x",
        "accept_spin",
        "accept_theta",
        "mean_abs_phi",
        "mean_m",
        "q95_abs_phi",
        "seed",
    ]

    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def create_figures(
    results: List[Dict[str, float]],
    output_dir: Path,
    kappa_working: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    success = [r for r in results if r["status"] == "SUCCESS"]
    if not success:
        print("No SUCCESS rows available for plotting.")
        return

    def _by_key(rows, key):
        return sorted(rows, key=lambda r: r[key])

    baseline = [
        r for r in success
        if math.isclose(r["g"], 0.5) and r["kappa"] == kappa_working
    ]

    if baseline:
        baseline_sorted = _by_key(baseline, "L")
        L_vals = [r["L"] for r in baseline_sorted]
        upsilon = [r["Upsilon_x"] for r in baseline_sorted]
        upsilon_err = [r["stderr_Upsilon_x"] for r in baseline_sorted]
        f_active = [r["f_active_x"] for r in baseline_sorted]
        f_active_err = [r["stderr_f_active_x"] for r in baseline_sorted]

        plt.figure(figsize=(7, 5))
        plt.errorbar(L_vals, upsilon, yerr=upsilon_err, fmt="o-", capsize=4)
        plt.xlabel("L")
        plt.ylabel(r"$\Upsilon_x$")
        plt.title(r"Stiffness $\Upsilon_x$ vs L (baseline)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "1.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.errorbar(L_vals, f_active, yerr=f_active_err, fmt="o-", capsize=4)
        plt.xlabel("L")
        plt.ylabel(r"$f_{\mathrm{active},x}$")
        plt.title(r"Active x-edge fraction vs L (baseline)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "4.png", dpi=150)
        plt.close()

    g_sweep_rows = [r for r in success if r["kappa"] == kappa_working]
    if g_sweep_rows:
        plt.figure(figsize=(7, 5))
        for L in sorted({r["L"] for r in g_sweep_rows}):
            rows = _by_key([r for r in g_sweep_rows if r["L"] == L], "g")
            g_vals = [r["g"] for r in rows]
            upsilon = [r["Upsilon_x"] for r in rows]
            upsilon_err = [r["stderr_Upsilon_x"] for r in rows]
            plt.errorbar(g_vals, upsilon, yerr=upsilon_err, fmt="o-", capsize=3, label=f"L={L}")
        plt.xlabel("g")
        plt.ylabel(r"$\Upsilon_x$")
        plt.title(r"$\Upsilon_x$ vs g (by L)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "2.png", dpi=150)
        plt.close()

    kappa_rows = [r for r in success if math.isclose(r["g"], 0.5)]
    if kappa_rows:
        plt.figure(figsize=(7, 5))
        for L in sorted({r["L"] for r in kappa_rows}):
            rows = _by_key([r for r in kappa_rows if r["L"] == L], "kappa")
            kappas = [r["kappa"] for r in rows]
            upsilon = [r["Upsilon_x"] for r in rows]
            upsilon_err = [r["stderr_Upsilon_x"] for r in rows]
            plt.errorbar(kappas, upsilon, yerr=upsilon_err, fmt="o-", capsize=3, label=f"L={L}")
        plt.xlabel(r"$\kappa$")
        plt.ylabel(r"$\Upsilon_x$")
        plt.title(r"$\Upsilon_x$ vs $\kappa$ (by L)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "3.png", dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 Option 1 stiffness runner")
    parser.add_argument("--output-dir", default="results/stiffness_phase4_option1", help="Output directory")
    parser.add_argument("--base-seed", type=int, default=1000, help="Base seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test configuration")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    base_params = StiffnessParams()

    J_final, T_final = screen_spin_order(base_params)
    base_params = replace(base_params, J=J_final, T=T_final)

    usable_kappas, kappa_stats = screen_kappa(base_params)
    kappa_working, kappa_low, kappa_high = pick_kappa_values(usable_kappas)

    print(f"Selected κ_working={kappa_working}, κ_low={kappa_low}, κ_high={kappa_high}")

    q95_lookup = {kappa: stats["q95"] for kappa, stats in kappa_stats.items()}

    if args.smoke_test:
        l_list = [12]
        g_sweep = [0.5]
        kappa_sweep = [kappa_working]
        base_params = replace(base_params, n_burnin=50, n_sample=100, n_batches=5)
        processes = 2
    else:
        l_list = [12, 16, 24, 32]
        g_sweep = [0.3, 0.5, 1.0]
        kappa_sweep = [v for v in [kappa_low, kappa_working, kappa_high] if v is not None]
        processes = max(1, min(10, (cpu_count() or 2) - 2))

    tasks = build_tasks(
        base_params,
        l_list,
        kappa_working,
        kappa_low,
        kappa_high,
        g_sweep,
        kappa_sweep,
        args.base_seed,
        q95_lookup,
        log_dir,
    )

    print(f"Launching {len(tasks)} tasks with {processes} processes...")
    with Pool(processes=processes) as pool:
        results = pool.map(run_stiffness_task, tasks)

    output_csv = output_dir / "stiffness_results.csv"
    save_results_csv(results, output_csv)
    print(f"Saved results: {output_csv}")

    create_figures(results, output_dir, kappa_working)
    print(f"Figures saved under {output_dir}")
