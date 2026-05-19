"""
CASAS – Collision Probability Engine (Strategy Pipeline Variant)
================================================================
A functional execution architecture tracking exact logical parity 
with independent Alfano, Foster, Chan, and MC solvers.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable, Final

import numpy as np
from scipy import special, integrate

from CDM_parser import CDMRecord, CovarianceMatrix

log = logging.getLogger(__name__)

try:
    from casas_propagator import propagate, propagate_covariance, PropConfig, backend
    _DA_AVAILABLE = True
    log.info(f"DA propagator loaded: backend={backend()}")
except ImportError:
    _DA_AVAILABLE = False
    log.warning("casas_propagator not found – using legacy Python covariance propagation")


# ── Configuration & Static Mappings ──────────────────────────────────────────

HBR_TABLE: Final[Dict[str, float]] = {
    "PAYLOAD": 0.005, "DEBRIS": 0.001, "ROCKET BODY": 0.003,
    "UNKNOWN": 0.003, "DEFAULT": 0.003
}

def get_hbr(obj_type: str) -> float:
    return HBR_TABLE.get(obj_type.upper(), HBR_TABLE["DEFAULT"])

def estimate_hbr_from_rcs(rcs_m2: float) -> float:
    if rcs_m2 <= 0.0 or not math.isfinite(rcs_m2):
        return HBR_TABLE["DEFAULT"]
    return float(np.clip(math.sqrt(rcs_m2 / math.pi) / 1000.0, 1e-6, 0.050))


# ── Geometry Structure ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConjunctionGeometry:
    miss_distance_km: float
    rel_vel_kms: float
    sigma_x: float
    sigma_y: float
    rho_xy: float
    combined_hbr_km: float

    @property
    def pc_miss_distance(self) -> float:
        return self.miss_distance_km / max(self.combined_hbr_km, 1e-9)

    @property
    def sigma_combined(self) -> float:
        return math.sqrt(self.sigma_x * self.sigma_y)


def build_encounter_geometry(rec: CDMRecord) -> ConjunctionGeometry:
    comb = rec.combined_covariance.matrix
    vr = np.array([rec.rel_vel_r, rec.rel_vel_t, rec.rel_vel_n], dtype=float)
    v_mag = np.linalg.norm(vr) or 1e-9
    v_hat = vr / v_mag

    pos_rel = np.array([rec.rel_pos_r, rec.rel_pos_t, rec.rel_pos_n], dtype=float)
    if np.linalg.norm(pos_rel) < 1e-9:
        pos_rel = np.array([0.0, 1.0, 0.0])

    x_hat = pos_rel - np.dot(pos_rel, v_hat) * v_hat
    x_norm = np.linalg.norm(x_hat)
    if x_norm < 1e-9:
        x_hat = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), v_hat) * v_hat
        x_norm = np.linalg.norm(x_hat)
    x_hat /= x_norm

    y_hat = np.cross(v_hat, x_hat)
    y_hat /= (np.linalg.norm(y_hat) or 1.0)

    P3 = comb[:3, :3]
    sig_x = math.sqrt(max(x_hat @ P3 @ x_hat, 1e-12))
    sig_y = math.sqrt(max(y_hat @ P3 @ y_hat, 1e-12))
    rho = max(-0.999, min(0.999, (x_hat @ P3 @ y_hat) / (sig_x * sig_y + 1e-15)))

    return ConjunctionGeometry(
        miss_distance_km=rec.miss_distance_km,
        rel_vel_kms=rec.relative_speed_ms / 1000.0,
        sigma_x=sig_x, sigma_y=sig_y, rho_xy=rho,
        combined_hbr_km=estimate_hbr_from_rcs(rec.t_metrics.rcs_estimate) + get_hbr(rec.c_object_type)
    )


def _diagonalize(geom: ConjunctionGeometry) -> Tuple[float, float, float, float, np.ndarray]:
    cov2 = np.array([
        [geom.sigma_x**2, geom.rho_xy * geom.sigma_x * geom.sigma_y],
        [geom.rho_xy * geom.sigma_x * geom.sigma_y, geom.sigma_y**2]
    ])
    vals, vecs = np.linalg.eigh(cov2)
    sig1, sig2 = np.sqrt(np.maximum(vals, 1e-24))
    m_prin = vecs.T @ np.array([geom.miss_distance_km, 0.0])
    return float(sig1), float(sig2), float(m_prin[0]), float(m_prin[1]), vecs


def _isotropic_bessel_core(d: float, R: float, sigma: float) -> float:
    if sigma < 1e-12 or R < 1e-12:
        return 0.0
    if R / sigma < 0.05:
        return (R**2 / (2.0 * sigma**2)) * math.exp(-d**2 / (2.0 * sigma**2))

    try:
        res, _ = integrate.quad(
            lambda r: (r / sigma**2) * math.exp(-0.5 * (r - d)**2 / sigma**2) * special.i0e(r * d / sigma**2),
            0.0, R, epsabs=1e-12, limit=150
        )
        return float(np.clip(res, 0.0, 1.0))
    except Exception:
        return (R**2 / (2.0 * sigma**2)) * math.exp(-d**2 / (2.0 * sigma**2))


# ── The Mathematical Solvers (Strategies) ─────────────────────────────────────

def pc_alfano(geom: ConjunctionGeometry) -> float:
    R = geom.combined_hbr_km
    if R < 1e-12: return 0.0

    sig1, sig2, mx, my, _ = _diagonalize(geom)
    if sig1 < 1e-12 or sig2 < 1e-12: return 0.0

    sigma_eff = math.sqrt(sig1 * sig2)
    if R / sigma_eff < 0.05:
        return float((R**2 / (2.0 * sigma_eff**2)) * math.exp(-0.5 * ((mx / sig1)**2 + (my / sig2)**2)))

    def strip_integrand(y: float) -> float:
        pdf_y = math.exp(-((y - my) / (sig2 * math.sqrt(2.0)))**2) / (sig2 * math.sqrt(2.0 * math.pi))
        disc = R**2 - y**2
        if disc <= 0.0: return 0.0
        x_lim = math.sqrt(disc)
        strip = 0.5 * (math.erf((x_lim - mx) / (sig1 * math.sqrt(2.0))) - math.erf((-x_lim - mx) / (sig1 * math.sqrt(2.0))))
        return pdf_y * strip

    try:
        res, _ = integrate.quad(strip_integrand, -R, R, epsabs=1e-12, limit=150)
        return float(np.clip(res, 0.0, 1.0))
    except Exception:
        return _isotropic_bessel_core(math.sqrt(mx**2 + my**2), R, sigma_eff)


def pc_foster(geom: ConjunctionGeometry) -> float:
    R, d, sx, sy, rho = geom.combined_hbr_km, geom.miss_distance_km, geom.sigma_x, geom.sigma_y, geom.rho_xy
    if sx < 1e-12 or sy < 1e-12: return 0.0

    try:
        def density_function(x: float, y: float) -> float:
            dx, dy = x - d, y
            z = (dx**2 / sx**2 - 2 * rho * dx * dy / (sx * sy) + dy**2 / sy**2) / (1 - rho**2)
            return np.exp(-0.5 * z) / (2 * np.pi * sx * sy * np.sqrt(1 - rho**2))

        pc, _ = integrate.dblquad(
            lambda y, x: density_function(x, y) if x**2 + y**2 <= R**2 else 0.0,
            -R, R,
            lambda x: -np.sqrt(max(R**2 - x**2, 0.0)),
            lambda x: np.sqrt(max(R**2 - x**2, 0.0))
        )
        return float(np.clip(pc, 0.0, 1.0))
    except Exception:
        return pc_alfano(geom)


def pc_chan(geom: ConjunctionGeometry) -> float:
    R = geom.combined_hbr_km
    if R < 1e-12: return 0.0

    sig1, sig2, mx, my, _ = _diagonalize(geom)
    sigma_eff = math.sqrt(sig1 * sig2)
    if sigma_eff < 1e-12: return 0.0

    d_eff = math.sqrt(mx**2 + my**2)
    u, v = R**2 / (2.0 * sigma_eff**2), d_eff**2 / (2.0 * sigma_eff**2)
    if u > 10.0 or v > 10.0:
        return _isotropic_bessel_core(d_eff, R, sigma_eff)

    total, v_pow = 0.0, 1.0
    for m in range(30):
        inner, u_pow = 0.0, 1.0
        for k in range(m + 1):
            inner += u_pow / math.factorial(k)
            u_pow *= u
        term = (v_pow / math.factorial(m)) * inner
        total += term
        v_pow *= v
        if m > 5 and abs(term) < 1e-15 * max(abs(total), 1.0): break

    return float(np.clip(1.0 - math.exp(-(u + v)) * total, 0.0, 1.0))


def pc_monte_carlo(geom: ConjunctionGeometry) -> float:
    R, d, sx, sy, rho = geom.combined_hbr_km, geom.miss_distance_km, geom.sigma_x, geom.sigma_y, geom.rho_xy
    if sx < 1e-12 or sy < 1e-12: return 0.0

    rng = np.random.default_rng(None)
    samples = rng.multivariate_normal(np.zeros(2), np.array([[sx**2, rho*sx*sy], [rho*sx*sy, sy**2]]), size=100_000)
    pc = np.count_nonzero(np.linalg.norm(samples - np.array([d, 0.0]), axis=1) < R) / 100_000
    
    log.info(f"MC: Pc={pc:.2e} ± {math.sqrt(pc * (1.0 - pc) / 100_000):.2e} (100,000 samples)")
    return float(pc)


# ── Execution Registry & Pipeline ───────────────────────────────────────────

# Strategy Engine Map
SOLVER_REGISTRY: Final[Dict[str, Callable[[ConjunctionGeometry], float]]] = {
    "alfano": pc_alfano,
    "foster": pc_foster,
    "chan": pc_chan,
    "monte_carlo": pc_monte_carlo
}

@dataclass
class PcResult:
    alfano: float
    foster: float
    chan: float
    monte_carlo: float
    consensus: float
    risk_level: str

    @property
    def log10_pc(self) -> float:
        return math.log10(max(self.consensus, 1e-99))


def compute_all_pc(rec: CDMRecord, fast: bool = False) -> PcResult:
    geom = build_encounter_geometry(rec)
    
    # Evaluate via dynamic execution lookup loop
    metrics = {
        key: (0.0 if (key == "monte_carlo" and fast) else solver(geom))
        for key, solver in SOLVER_REGISTRY.items()
    }

    metrics["consensus"] = metrics["monte_carlo"] if (not fast and metrics["monte_carlo"] > 0.0) else metrics["foster"]
    
    # Evaluate classification threshold bounds
    c = metrics["consensus"]
    if c >= 1e-4:   risk = "CRITICAL"
    elif c >= 1e-5: risk = "HIGH"
    elif c >= 1e-6: risk = "MEDIUM"
    elif c >= 1e-7: risk = "LOW"
    else:           risk = "NEGLIGIBLE"

    return PcResult(
        alfano=metrics["alfano"], foster=metrics["foster"], chan=metrics["chan"], 
        monte_carlo=metrics["monte_carlo"], consensus=metrics["consensus"], risk_level=risk
    )