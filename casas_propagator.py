"""
CASAS – DA Propagator Python Interface (COMPLETE v2 Integration)
=================================================================
casas_propagator.py

Drop-in integration layer for CASAS.
- Uses the compiled C++ backend (casas_cpp) when available.
- Falls back to a pure-Python two-body+J2 propagator automatically.

v2 Features:
  • RK4 fixed-step propagation (original v1 API)
  • RK7(8) Dormand-Prince high-accuracy propagation
  • Adaptive step-size control with PI controller
  • Order-2 DA propagation (Hessian/STT for non-linear covariance)
  • DACE-style mean correction and full covariance mapping
  • EKF measurement update (Joseph stabilised form)

Usage
-----
from casas_propagator import (
    propagate, propagate_rk78, propagate_order2, propagate_order2_full,
    propagate_covariance, propagate_covariance_order2,
    propagate_mean_order2, ekf_update,
    PropConfig, PropResult, backend
)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ── Try loading C++ backend ────────────────────────────────────
# CRITICAL FIX: The compiled module is named 'casas_cpp', NOT 'casas_da'
try:
    import casas_cpp as _cpp
    _USE_CPP = True
except ImportError:
    _USE_CPP = False
    warnings.warn(
        "casas_cpp C++ extension not found – using Python fallback propagator.\n"
        "Build instructions:\n"
        "  1. Ensure casas_da.hpp and bindings.cpp are in the same directory\n"
        "  2. Run:  bash build.sh\n"
        "  3. The resulting .so/.pyd file should be importable as 'casas_cpp'",
        ImportWarning,
        stacklevel=2,
    )


# ================================================================
# Public configuration dataclass (mirrors C++ PropConfig v2)
# ================================================================
@dataclass
class PropConfig:
    """Force-model and integrator configuration matching C++ PropConfig."""
    # v1 fields
    dt_step_s: float = 60.0
    use_j2:    bool  = True
    use_drag:  bool  = True
    bstar:     float = 0.0
    f107:      float = 150.0
    ap:        float = 15.0
    # v2 fields
    use_rk78:  bool  = False
    adaptive:  bool  = False
    abs_tol:   float = 1e-9
    rel_tol:   float = 1e-9
    use_srp:   bool  = False
    cr_am:     float = 0.01
    order2:    bool  = False

    def _to_cpp(self):
        """Convert to casas_cpp.PropConfig C++ object."""
        if not _USE_CPP:
            raise RuntimeError("C++ backend not available")
        c = _cpp.PropConfig()
        c.dt_step_s = self.dt_step_s
        c.use_j2    = self.use_j2
        c.use_drag  = self.use_drag
        c.bstar     = self.bstar
        c.f107      = self.f107
        c.ap        = self.ap
        c.use_rk78  = self.use_rk78
        c.adaptive  = self.adaptive
        c.abs_tol   = self.abs_tol
        c.rel_tol   = self.rel_tol
        c.use_srp   = self.use_srp
        c.cr_am     = self.cr_am
        c.order2    = self.order2
        return c


# ================================================================
# Public result dataclasses
# ================================================================
@dataclass
class PropResult:
    """Result from 1st-order propagation (RK4 or RK78)."""
    state: List[float]
    stm:   List[List[float]]


@dataclass
class PropResult2:
    """Result from order-2 propagation: state + STM + STT."""
    state: List[float]
    stm:   List[List[float]]
    stt:   List[List[List[float]]]


@dataclass
class PropResult2Full:
    """DACE-style full order-2 result with mean correction."""
    state: List[float]
    mean:  List[float]
    stm:   List[List[float]]
    stt:   List[List[List[float]]]


@dataclass
class EKFResult:
    """EKF measurement update result."""
    xp:          List[float]
    Pp:          List[List[float]]
    innovation:  List[float]
    S:           List[List[float]]
    nis:         float


# ================================================================
# Pure-Python fallback (two-body + J2, no drag)
# ================================================================
_MU  = 398600.4418
_J2  = 1.08263e-3
_RE  = 6378.137


def _accel_j2(r: np.ndarray) -> np.ndarray:
    """Two-body + J2 acceleration in km/s²."""
    x, y, z = r
    r2  = float(np.dot(r, r))
    r1  = np.sqrt(r2)
    r3  = r2 * r1
    r5  = r3 * r2
    z2r2 = (z*z) / r2
    a_2b = -_MU / r3 * r
    fac = 1.5 * _J2 * _MU * _RE**2 / r5
    a_j2 = fac * np.array([x * (5.0*z2r2 - 1.0), y * (5.0*z2r2 - 1.0), z * (5.0*z2r2 - 3.0)])
    return a_2b + a_j2


def _rk4_step_py(state: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 step on concatenated [r, v] state (6,)."""
    def deriv(s):
        r, v = s[:3], s[3:]
        return np.concatenate([v, _accel_j2(r)])
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _stm_py(x0: np.ndarray, dt_total: float, dt_step: float) -> np.ndarray:
    """Compute 6×6 STM via finite differences (fallback only)."""
    eps_vals = [1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6]
    def prop_scalar(s):
        n = max(1, int(np.ceil(abs(dt_total) / dt_step)))
        dt = dt_total / n
        for _ in range(n):
            s = _rk4_step_py(s, dt)
        return s
    xf0 = prop_scalar(x0.copy())
    stm = np.zeros((6, 6))
    for j in range(6):
        xp = x0.copy(); xp[j] += eps_vals[j]
        xm = x0.copy(); xm[j] -= eps_vals[j]
        stm[:, j] = (prop_scalar(xp) - prop_scalar(xm)) / (2 * eps_vals[j])
    return stm


def _propagate_python(x0_km: List[float], dt_total_s: float, cfg: PropConfig) -> PropResult:
    """Pure-Python two-body+J2 propagator with finite-difference STM."""
    x0 = np.asarray(x0_km, dtype=float)
    n  = max(1, int(np.ceil(abs(dt_total_s) / cfg.dt_step_s)))
    dt = dt_total_s / n
    state = x0.copy()
    for _ in range(n):
        state = _rk4_step_py(state, dt)
    stm = _stm_py(x0, dt_total_s, cfg.dt_step_s)
    return PropResult(state=state.tolist(), stm=stm.tolist())


# ================================================================
# Public API – v1 (original, unchanged signatures)
# ================================================================

def propagate(x0_km: List[float], dt_s: float, cfg: Optional[PropConfig] = None) -> PropResult:
    """Propagate an ECI state vector using RK4 (fixed step)."""
    if cfg is None:
        cfg = PropConfig()
    if _USE_CPP:
        raw = _cpp.da_propagate(list(x0_km), float(dt_s), cfg._to_cpp())
        return PropResult(state=list(raw.state), stm=[list(row) for row in raw.stm])
    return _propagate_python(x0_km, dt_s, cfg)


def propagate_covariance(P0: List[List[float]], stm: List[List[float]]) -> List[List[float]]:
    """Map covariance forward:  P(t) = Φ · P0 · Φᵀ"""
    if _USE_CPP:
        raw = _cpp.propagate_covariance([list(row) for row in P0], [list(row) for row in stm])
        return [list(row) for row in raw]
    phi = np.asarray(stm)
    p0  = np.asarray(P0)
    return (phi @ p0 @ phi.T).tolist()


# ================================================================
# Public API – v2 (NEW features)
# ================================================================

def propagate_rk78(x0_km: List[float], dt_s: float, cfg: Optional[PropConfig] = None) -> PropResult:
    """High-accuracy RK7(8) Dormand-Prince propagation."""
    if cfg is None:
        cfg = PropConfig()
    cfg.use_rk78 = True
    if _USE_CPP:
        raw = _cpp.da_propagate_rk78(list(x0_km), float(dt_s), cfg._to_cpp())
        return PropResult(state=list(raw.state), stm=[list(row) for row in raw.stm])
    warnings.warn("RK78 not available in Python fallback – using RK4", RuntimeWarning)
    return _propagate_python(x0_km, dt_s, cfg)


def propagate_order2(x0_km: List[float], dt_s: float, cfg: Optional[PropConfig] = None) -> PropResult2:
    """Order-2 DA propagation: computes STM + 6×6×6 STT."""
    if cfg is None:
        cfg = PropConfig()
    cfg.order2 = True
    if _USE_CPP:
        raw = _cpp.da_propagate_order2(list(x0_km), float(dt_s), cfg._to_cpp())
        return PropResult2(state=list(raw.state), stm=[list(row) for row in raw.stm], stt=raw.stt)
    warnings.warn("Order-2 not available in fallback – returning 1st-order with zero STT", RuntimeWarning)
    res1 = _propagate_python(x0_km, dt_s, cfg)
    zero_stt = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(6)]
    return PropResult2(state=res1.state, stm=res1.stm, stt=zero_stt)


def propagate_order2_full(x0_km: List[float], dt_s: float, P0: List[List[float]], cfg: Optional[PropConfig] = None) -> PropResult2Full:
    """DACE-style full order-2 propagation with mean correction."""
    if cfg is None:
        cfg = PropConfig()
    cfg.order2 = True
    if _USE_CPP:
        raw = _cpp.da_propagate_order2_full(list(x0_km), float(dt_s), [list(row) for row in P0], cfg._to_cpp())
        return PropResult2Full(state=list(raw.state), mean=list(raw.mean), stm=[list(row) for row in raw.stm], stt=raw.stt)
    warnings.warn("Order-2 full not available in fallback", RuntimeWarning)
    res1 = _propagate_python(x0_km, dt_s, cfg)
    zero_stt = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(6)]
    return PropResult2Full(state=res1.state, mean=res1.state, stm=res1.stm, stt=zero_stt)


def ekf_update(z: List[float], H: List[List[float]], R: List[List[float]], xm: List[float], Pm: List[List[float]]) -> EKFResult:
    """EKF Measurement Update — Joseph stabilised form."""
    if _USE_CPP:
        raw = _cpp.ekf_update(z, H, R, xm, Pm)
        return EKFResult(
            xp=list(raw.xp),
            Pp=raw.Pp,
            innovation=list(raw.innovation),
            S=None,            # or remove entirely
            nis=raw.nis
        )
    # Python fallback
    z_arr = np.asarray(z); H_arr = np.asarray(H); R_arr = np.asarray(R)
    xm_arr = np.asarray(xm); Pm_arr = np.asarray(Pm)
    innovation = z_arr - H_arr @ xm_arr
    S = H_arr @ Pm_arr @ H_arr.T + R_arr
    K = Pm_arr @ H_arr.T @ np.linalg.inv(S)
    xp = xm_arr + K @ innovation
    Pp = (np.eye(6) - K @ H_arr) @ Pm_arr
    nis = float(innovation @ np.linalg.inv(S) @ innovation)
    return EKFResult(xp=xp.tolist(), Pp=Pp.tolist(), innovation=innovation.tolist(), S=S.tolist(), nis=nis)


# ================================================================
# Utility
# ================================================================

def backend() -> str:
    """Return which backend is active: 'cpp' or 'python'."""
    return "cpp" if _USE_CPP else "python"


def version() -> str:
    """Return version string of the C++ backend."""
    if _USE_CPP:
        return getattr(_cpp, "__version__", "unknown")
    return "python-fallback"


# ================================================================
# Quick smoke-test
# ================================================================
if __name__ == "__main__":
    print(f"Backend: {backend()}")
    if _USE_CPP:
        print(f"C++ version: {version()}")
    x0 = [6778.0, 0.0, 0.0, 0.0, 7.668, 0.0]
    cfg = PropConfig(use_j2=True, use_drag=False)
    res = propagate(x0, dt_s=3600.0, cfg=cfg)
    print(f"State: {[f'{v:.3f}' for v in res.state]}")
    print(f"STM[0,0]: {res.stm[0][0]:.6f}")
    P0 = [[1e-4 if i == j else 0.0 for j in range(6)] for i in range(6)]
    Pf = propagate_covariance(P0, res.stm)
    print(f"Pf[0,0]: {Pf[0][0]:.6e}")