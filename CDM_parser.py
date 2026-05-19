"""
CASAS – CDM Parser
==================
Parses Conjunction Data Messages (CDMs) from CSV exports.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

log = logging.getLogger(__name__)

# ================== PROJECT DIRECTORY SETUP ==================
# This ensures all files are read/saved relative to this script's folder
SCRIPT_DIR = Path(__file__).parent.absolute()


# ============================================================
    


    
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418
R_EARTH  = 6378.137


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CovarianceMatrix:
    """
    Full 6×6 covariance matrix in RTN frame.
    Row/column order: [R, T, N, Rdot, Tdot, Ndot]
    """
    matrix: np.ndarray   # (6,6) symmetric positive-semi-definite

    # -- Convenience accessors -------------------------------------------
    @property
    def position_block(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def velocity_block(self) -> np.ndarray:
        return self.matrix[3:, 3:]

    @property
    def sigma_r(self) -> float:
        return math.sqrt(max(self.matrix[0, 0], 0.0))

    @property
    def sigma_t(self) -> float:
        return math.sqrt(max(self.matrix[1, 1], 0.0))

    @property
    def sigma_n(self) -> float:
        return math.sqrt(max(self.matrix[2, 2], 0.0))

    @property
    def pos_det(self) -> float:
        return float(np.linalg.det(self.position_block))

    @property
    def combined_pos_sigma(self) -> float:
        """RMS of diagonal position uncertainties."""
        return math.sqrt((self.matrix[0,0] + self.matrix[1,1] + self.matrix[2,2]) / 3.0)

    def is_positive_definite(self) -> bool:
        try:
            np.linalg.cholesky(self.matrix + np.eye(6)*1e-12)
            return True
        except np.linalg.LinAlgError:
            return False

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for i in range(6):
            for j in range(i + 1):
                d[f"C{i}{j}"] = float(self.matrix[i, j])
        return d


@dataclass
class OrbitalState:
    """Keplerian orbital elements extracted from CDM fields."""
    sma_km: float         # Semi-major axis (km)
    eccentricity: float
    inclination_deg: float
    apogee_alt_km: float
    perigee_alt_km: float
    span_days: float       # OD span

    @property
    def period_s(self) -> float:
        return 2 * math.pi * math.sqrt(self.sma_km**3 / MU_EARTH)

    @property
    def period_h(self) -> float:
        return self.period_s / 3600.0

    @property
    def mean_motion(self) -> float:
        """rad/s"""
        return math.sqrt(MU_EARTH / self.sma_km**3)


@dataclass
class ObjectMetrics:
    """
    Per-object tracking quality and physical characteristics.
    """
    time_lastob_start: float   # days before TCA
    time_lastob_end: float
    recommended_od_span: float
    actual_od_span: float
    obs_available: int
    obs_used: int
    residuals_accepted: float  # %
    weighted_rms: float
    rcs_estimate: float        # m²
    cd_area_over_mass: float   # m²/kg
    cr_area_over_mass: float
    sedr: float                # Specific Energy Dissipation Rate


@dataclass
class SpaceWeather:
    F10: float    # Solar flux index (10.7 cm)
    F3M: float    # 3-month average
    SSN: float    # Sunspot number
    AP: float     # Geomagnetic index


@dataclass
class CDMRecord:
    """
    Complete Conjunction Data Message record.
    All fields from the ESA/NASA CDM standard are preserved.
    """
    # -- Identity --
    row_id: int
    event_id: int
    mission_id: int
    time_to_tca_days: float   # days until TCA

    # -- Risk --
    risk: float               # log10(Pc) typically
    max_risk_estimate: float
    max_risk_scaling: float

    # -- Geometry at TCA --
    miss_distance_m: float    # metres
    relative_speed_ms: float  # m/s

    # -- RTN relative position / velocity --
    rel_pos_r: float          # m
    rel_pos_t: float
    rel_pos_n: float
    rel_vel_r: float          # m/s
    rel_vel_t: float
    rel_vel_n: float

    # -- Ancillary geometry --
    geocentric_latitude: float  # deg
    azimuth: float              # deg
    elevation: float            # deg
    mahalanobis_distance: float

    # -- Target object --
    t_orbital: OrbitalState
    t_metrics: ObjectMetrics
    t_covariance: CovarianceMatrix
    t_sigma_r: float     # km
    t_sigma_t: float
    t_sigma_n: float
    t_sigma_rdot: float
    t_sigma_tdot: float
    t_sigma_ndot: float
    t_pos_cov_det: float

    # -- Chaser (secondary) object --
    c_object_type: str
    c_orbital: OrbitalState
    c_metrics: ObjectMetrics
    c_covariance: CovarianceMatrix
    c_sigma_r: float
    c_sigma_t: float
    c_sigma_n: float
    c_sigma_rdot: float
    c_sigma_tdot: float
    c_sigma_ndot: float
    c_pos_cov_det: float

    # -- Space weather --
    space_weather: SpaceWeather

    # -- Combined covariance (sum in RTN) --
    combined_covariance: Optional[CovarianceMatrix] = field(default=None)

    def __post_init__(self):
        self.combined_covariance = CovarianceMatrix(
            matrix=self.t_covariance.matrix + self.c_covariance.matrix
        )

    @property
    def miss_distance_km(self) -> float:
        return self.miss_distance_m / 1000.0

    @property
    def relative_speed_kms(self) -> float:
        return self.relative_speed_ms / 1000.0

    @property
    def time_to_tca_hours(self) -> float:
        return self.time_to_tca_days * 24.0

    @property
    def risk_level(self) -> str:
        """Human-readable risk level from the risk (log10 Pc) field."""
        if self.risk >= -4:
            return "CRITICAL"
        elif self.risk >= -5:
            return "HIGH"
        elif self.risk >= -6:
            return "MEDIUM"
        elif self.risk >= -7:
            return "LOW"
        else:
            return "NEGLIGIBLE"

    @property
    def pc_value(self) -> float:
        """Probability of collision (linear)."""
        try:
            return 10.0 ** self.risk
        except Exception:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id":             self.row_id,
            "event_id":           self.event_id,
            "mission_id":         self.mission_id,
            "time_to_tca_days":   self.time_to_tca_days,
            "time_to_tca_hours":  self.time_to_tca_hours,
            "risk":               self.risk,
            "pc_value":           self.pc_value,
            "risk_level":         self.risk_level,
            "max_risk_estimate":  self.max_risk_estimate,
            "max_risk_scaling":   self.max_risk_scaling,
            "miss_distance_m":    self.miss_distance_m,
            "miss_distance_km":   self.miss_distance_km,
            "relative_speed_ms":  self.relative_speed_ms,
            "rel_pos_r":          self.rel_pos_r,
            "rel_pos_t":          self.rel_pos_t,
            "rel_pos_n":          self.rel_pos_n,
            "rel_vel_r":          self.rel_vel_r,
            "rel_vel_t":          self.rel_vel_t,
            "rel_vel_n":          self.rel_vel_n,
            "mahalanobis_distance": self.mahalanobis_distance,
            "geocentric_latitude":  self.geocentric_latitude,
            "azimuth":            self.azimuth,
            "elevation":          self.elevation,
            # Target
            "t_sma_km":           self.t_orbital.sma_km,
            "t_ecc":              self.t_orbital.eccentricity,
            "t_inc_deg":          self.t_orbital.inclination_deg,
            "t_apogee_km":        self.t_orbital.apogee_alt_km,
            "t_perigee_km":       self.t_orbital.perigee_alt_km,
            "t_sigma_r_km":       self.t_sigma_r,
            "t_sigma_t_km":       self.t_sigma_t,
            "t_sigma_n_km":       self.t_sigma_n,
            "t_pos_cov_det":      self.t_pos_cov_det,
            # Chaser
            "c_object_type":      self.c_object_type,
            "c_sma_km":           self.c_orbital.sma_km,
            "c_ecc":              self.c_orbital.eccentricity,
            "c_inc_deg":          self.c_orbital.inclination_deg,
            "c_apogee_km":        self.c_orbital.apogee_alt_km,
            "c_perigee_km":       self.c_orbital.perigee_alt_km,
            "c_sigma_r_km":       self.c_sigma_r,
            "c_sigma_t_km":       self.c_sigma_t,
            "c_sigma_n_km":       self.c_sigma_n,
            "c_pos_cov_det":      self.c_pos_cov_det,
            # Space weather
            "F10":                self.space_weather.F10,
            "F3M":                self.space_weather.F3M,
            "SSN":                self.space_weather.SSN,
            "AP":                 self.space_weather.AP,
        }


# ---------------------------------------------------------------------------
# Covariance reconstruction helpers
# ---------------------------------------------------------------------------

def _build_covariance_6x6(
    sigma_r: float, sigma_t: float, sigma_n: float,
    sigma_rdot: float, sigma_tdot: float, sigma_ndot: float,
    ct_r: float, cn_r: float, cn_t: float,
    crdot_r: float, crdot_t: float, crdot_n: float,
    ctdot_r: float, ctdot_t: float, ctdot_n: float, ctdot_rdot: float,
    cndot_r: float, cndot_t: float, cndot_n: float, cndot_rdot: float, cndot_tdot: float,
) -> np.ndarray:
    """
    Reconstruct the 6×6 covariance matrix (RTN frame) from CDM fields.

    State vector order: [R, T, N, Rdot, Tdot, Ndot]
    The CDM stores correlation coefficients (not raw covariances) for
    off-diagonal elements relative to sigma products.
    """
    # Diagonal variances
    var = np.array([
        sigma_r**2, sigma_t**2, sigma_n**2,
        sigma_rdot**2, sigma_tdot**2, sigma_ndot**2
    ])

    # Off-diagonal correlation × sigma_i × sigma_j
    # Index map: R=0, T=1, N=2, Rdot=3, Tdot=4, Ndot=5
    cov = np.diag(var)

    def _fill(i, j, corr):
        val = corr * math.sqrt(var[i]) * math.sqrt(var[j])
        cov[i, j] = val
        cov[j, i] = val

    _fill(1, 0, ct_r)        # T-R
    _fill(2, 0, cn_r)        # N-R
    _fill(2, 1, cn_t)        # N-T
    _fill(3, 0, crdot_r)     # Rdot-R
    _fill(3, 1, crdot_t)     # Rdot-T
    _fill(3, 2, crdot_n)     # Rdot-N
    _fill(4, 0, ctdot_r)     # Tdot-R
    _fill(4, 1, ctdot_t)     # Tdot-T
    _fill(4, 2, ctdot_n)     # Tdot-N
    _fill(4, 3, ctdot_rdot)  # Tdot-Rdot
    _fill(5, 0, cndot_r)     # Ndot-R
    _fill(5, 1, cndot_t)     # Ndot-T
    _fill(5, 2, cndot_n)     # Ndot-N
    _fill(5, 3, cndot_rdot)  # Ndot-Rdot
    _fill(5, 4, cndot_tdot)  # Ndot-Tdot

    return cov


# ---------------------------------------------------------------------------
# CDM CSV Parser
# ---------------------------------------------------------------------------

class CDMParser:
    """
    Parses CDM records from a tab-separated CSV file.
    Handles missing values, type coercion, and covariance reconstruction.
    """

    _FLOAT_FIELDS = {
        "time_to_tca", "risk", "max_risk_estimate", "max_risk_scaling",
        "miss_distance", "relative_speed",
        "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
        "t_time_lastob_start", "t_time_lastob_end",
        "t_recommended_od_span", "t_actual_od_span",
        "t_residuals_accepted", "t_weighted_rms",
        "t_rcs_estimate", "t_cd_area_over_mass", "t_cr_area_over_mass", "t_sedr",
        "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc",
        "t_ct_r", "t_cn_r", "t_cn_t",
        "t_crdot_r", "t_crdot_t", "t_crdot_n",
        "t_ctdot_r", "t_ctdot_t", "t_ctdot_n", "t_ctdot_rdot",
        "t_cndot_r", "t_cndot_t", "t_cndot_n", "t_cndot_rdot", "t_cndot_tdot",
        "c_time_lastob_start", "c_time_lastob_end",
        "c_recommended_od_span", "c_actual_od_span",
        "c_residuals_accepted", "c_weighted_rms",
        "c_rcs_estimate", "c_cd_area_over_mass", "c_cr_area_over_mass", "c_sedr",
        "c_j2k_sma", "c_j2k_ecc", "c_j2k_inc",
        "c_ct_r", "c_cn_r", "c_cn_t",
        "c_crdot_r", "c_crdot_t", "c_crdot_n",
        "c_ctdot_r", "c_ctdot_t", "c_ctdot_n", "c_ctdot_rdot",
        "c_cndot_r", "c_cndot_t", "c_cndot_n", "c_cndot_rdot", "c_cndot_tdot",
        "t_span", "c_span", "t_h_apo", "t_h_per", "c_h_apo", "c_h_per",
        "geocentric_latitude", "azimuth", "elevation", "mahalanobis_distance",
        "t_position_covariance_det", "c_position_covariance_det",
        "t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t",
        "t_sigma_n", "c_sigma_n",
        "t_sigma_rdot", "c_sigma_rdot", "t_sigma_tdot", "c_sigma_tdot",
        "t_sigma_ndot", "c_sigma_ndot",
        "F10", "F3M", "SSN", "AP",
    }
    _INT_FIELDS = {
        "event_id", "mission_id",
        "t_obs_available", "t_obs_used",
        "c_obs_available", "c_obs_used",
    }

    @staticmethod
    def _safe_float(val: str, default: float = 0.0) -> float:
        try:
            return float(val.strip()) if val.strip() else default
        except (ValueError, AttributeError):
            return default

    @staticmethod
    def _safe_int(val: str, default: int = 0) -> int:
        try:
            return int(float(val.strip())) if val.strip() else default
        except (ValueError, AttributeError):
            return default

    def parse_file(self, filepath: str | Path, delimiter: str = "\t") -> List[CDMRecord]:
        """Parse all CDM records from a delimited file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CDM file not found: {path}")

        records: List[CDMRecord] = []
        sf = self._safe_float
        si = self._safe_int

        with open(path, "r", encoding="utf-8") as fh:
            # Auto-detect delimiter
            sample = fh.read(4096)
            fh.seek(0)
            if "\t" in sample and "," not in sample:
                delimiter = "\t"
            elif "," in sample:
                delimiter = ","

            reader = csv.DictReader(fh, delimiter=delimiter)
            for row_idx, row in enumerate(reader):
                try:
                    rec = self._parse_row(row_idx, row, sf, si)
                    records.append(rec)
                except Exception as exc:
                    log.warning("Row %d skipped – %s", row_idx, exc)

        log.info("Parsed %d CDM records from %s", len(records), path.name)
        return records

    def parse_dataframe(self, df) -> List[CDMRecord]:
        """Parse from a pandas DataFrame (alternative entry point)."""
        records: List[CDMRecord] = []
        sf = self._safe_float
        si = self._safe_int
        for row_idx, row in enumerate(df.to_dict(orient="records")):
            row = {k: str(v) for k, v in row.items()}
            try:
                rec = self._parse_row(row_idx, row, sf, si)
                records.append(rec)
            except Exception as exc:
                log.warning("Row %d skipped – %s", row_idx, exc)
        return records

    # -- Internal --

    def _parse_row(self, row_idx: int, row: dict, sf, si) -> CDMRecord:
        g = row.get


        # --- Key normalization (Dashboard/DB dicts vs CDM CSV exports) ---
        # Dashboard database rows use compact keys like 't_sma_km', whereas CDM CSV
        # exports use J2K keys like 't_j2k_sma'. Normalize so both inputs work.
        if 't_sma_km' in row and 't_j2k_sma' not in row:
            row['t_j2k_sma'] = row.get('t_sma_km', '0')
        if 't_ecc' in row and 't_j2k_ecc' not in row:
            row['t_j2k_ecc'] = row.get('t_ecc', '0')
        if 't_inc_deg' in row and 't_j2k_inc' not in row:
            row['t_j2k_inc'] = row.get('t_inc_deg', '0')
        if 't_apogee_km' in row and 't_h_apo' not in row:
            row['t_h_apo'] = row.get('t_apogee_km', '0')
        if 't_perigee_km' in row and 't_h_per' not in row:
            row['t_h_per'] = row.get('t_perigee_km', '0')
        if 't_sigma_r_km' in row and 't_sigma_r' not in row:
            row['t_sigma_r'] = row.get('t_sigma_r_km', '0')
        if 't_sigma_t_km' in row and 't_sigma_t' not in row:
            row['t_sigma_t'] = row.get('t_sigma_t_km', '0')
        if 't_sigma_n_km' in row and 't_sigma_n' not in row:
            row['t_sigma_n'] = row.get('t_sigma_n_km', '0')

        if 'c_sma_km' in row and 'c_j2k_sma' not in row:
            row['c_j2k_sma'] = row.get('c_sma_km', '0')
        if 'c_ecc' in row and 'c_j2k_ecc' not in row:
            row['c_j2k_ecc'] = row.get('c_ecc', '0')
        if 'c_inc_deg' in row and 'c_j2k_inc' not in row:
            row['c_j2k_inc'] = row.get('c_inc_deg', '0')
        if 'c_apogee_km' in row and 'c_h_apo' not in row:
            row['c_h_apo'] = row.get('c_apogee_km', '0')
        if 'c_perigee_km' in row and 'c_h_per' not in row:
            row['c_h_per'] = row.get('c_perigee_km', '0')
        if 'c_sigma_r_km' in row and 'c_sigma_r' not in row:
            row['c_sigma_r'] = row.get('c_sigma_r_km', '0')
        if 'c_sigma_t_km' in row and 'c_sigma_t' not in row:
            row['c_sigma_t'] = row.get('c_sigma_t_km', '0')
        if 'c_sigma_n_km' in row and 'c_sigma_n' not in row:
            row['c_sigma_n'] = row.get('c_sigma_n_km', '0')

        # --- Geometry / relative state normalisation (DB → CSV keys) ---
        if 'time_to_tca_days' in row and 'time_to_tca' not in row:
            row['time_to_tca'] = row.get('time_to_tca_days', '0')
        if 'miss_distance_m' in row and 'miss_distance' not in row:
            row['miss_distance'] = row.get('miss_distance_m', '0')
        if 'relative_speed_ms' in row and 'relative_speed' not in row:
            row['relative_speed'] = row.get('relative_speed_ms', '0')
        if 'rel_pos_r' in row and 'relative_position_r' not in row:
            row['relative_position_r'] = row.get('rel_pos_r', '0')
        if 'rel_pos_t' in row and 'relative_position_t' not in row:
            row['relative_position_t'] = row.get('rel_pos_t', '0')
        if 'rel_pos_n' in row and 'relative_position_n' not in row:
            row['relative_position_n'] = row.get('rel_pos_n', '0')
        if 'rel_vel_r' in row and 'relative_velocity_r' not in row:
            row['relative_velocity_r'] = row.get('rel_vel_r', '0')
        if 'rel_vel_t' in row and 'relative_velocity_t' not in row:
            row['relative_velocity_t'] = row.get('rel_vel_t', '0')
        if 'rel_vel_n' in row and 'relative_velocity_n' not in row:
            row['relative_velocity_n'] = row.get('rel_vel_n', '0')

        # --- Derive missing semi-major axis from apogee + perigee ---
        _R_EARTH = 6378.137
        t_sma = sf(g("t_j2k_sma", "0"))
        t_apo = sf(g("t_h_apo", "0"))
        t_per = sf(g("t_h_per", "0"))
        if t_sma <= 0 and t_apo > 0 and t_per > 0:
            row["t_j2k_sma"] = str((t_apo + t_per) / 2.0 + _R_EARTH)

        c_sma = sf(g("c_j2k_sma", "0"))
        c_apo = sf(g("c_h_apo", "0"))
        c_per = sf(g("c_h_per", "0"))
        if c_sma <= 0 and c_apo > 0 and c_per > 0:
            row["c_j2k_sma"] = str((c_apo + c_per) / 2.0 + _R_EARTH)

        # --- Target covariance ---
        t_cov_mat = _build_covariance_6x6(
            sigma_r=sf(g("t_sigma_r", "0")),
            sigma_t=sf(g("t_sigma_t", "0")),
            sigma_n=sf(g("t_sigma_n", "0")),
            sigma_rdot=sf(g("t_sigma_rdot", "0")),
            sigma_tdot=sf(g("t_sigma_tdot", "0")),
            sigma_ndot=sf(g("t_sigma_ndot", "0")),
            ct_r=sf(g("t_ct_r", "0")),
            cn_r=sf(g("t_cn_r", "0")),
            cn_t=sf(g("t_cn_t", "0")),
            crdot_r=sf(g("t_crdot_r", "0")),
            crdot_t=sf(g("t_crdot_t", "0")),
            crdot_n=sf(g("t_crdot_n", "0")),
            ctdot_r=sf(g("t_ctdot_r", "0")),
            ctdot_t=sf(g("t_ctdot_t", "0")),
            ctdot_n=sf(g("t_ctdot_n", "0")),
            ctdot_rdot=sf(g("t_ctdot_rdot", "0")),
            cndot_r=sf(g("t_cndot_r", "0")),
            cndot_t=sf(g("t_cndot_t", "0")),
            cndot_n=sf(g("t_cndot_n", "0")),
            cndot_rdot=sf(g("t_cndot_rdot", "0")),
            cndot_tdot=sf(g("t_cndot_tdot", "0")),
        )

        # --- Chaser covariance ---
        c_cov_mat = _build_covariance_6x6(
            sigma_r=sf(g("c_sigma_r", "0")),
            sigma_t=sf(g("c_sigma_t", "0")),
            sigma_n=sf(g("c_sigma_n", "0")),
            sigma_rdot=sf(g("c_sigma_rdot", "0")),
            sigma_tdot=sf(g("c_sigma_tdot", "0")),
            sigma_ndot=sf(g("c_sigma_ndot", "0")),
            ct_r=sf(g("c_ct_r", "0")),
            cn_r=sf(g("c_cn_r", "0")),
            cn_t=sf(g("c_cn_t", "0")),
            crdot_r=sf(g("c_crdot_r", "0")),
            crdot_t=sf(g("c_crdot_t", "0")),
            crdot_n=sf(g("c_crdot_n", "0")),
            ctdot_r=sf(g("c_ctdot_r", "0")),
            ctdot_t=sf(g("c_ctdot_t", "0")),
            ctdot_n=sf(g("c_ctdot_n", "0")),
            ctdot_rdot=sf(g("c_ctdot_rdot", "0")),
            cndot_r=sf(g("c_cndot_r", "0")),
            cndot_t=sf(g("c_cndot_t", "0")),
            cndot_n=sf(g("c_cndot_n", "0")),
            cndot_rdot=sf(g("c_cndot_rdot", "0")),
            cndot_tdot=sf(g("c_cndot_tdot", "0")),
        )

        t_orbital = OrbitalState(
            sma_km=sf(g("t_j2k_sma", "0")),
            eccentricity=sf(g("t_j2k_ecc", "0")),
            inclination_deg=sf(g("t_j2k_inc", "0")),
            apogee_alt_km=sf(g("t_h_apo", "0")),
            perigee_alt_km=sf(g("t_h_per", "0")),
            span_days=sf(g("t_span", "0")),
        )
        c_orbital = OrbitalState(
            sma_km=sf(g("c_j2k_sma", "0")),
            eccentricity=sf(g("c_j2k_ecc", "0")),
            inclination_deg=sf(g("c_j2k_inc", "0")),
            apogee_alt_km=sf(g("c_h_apo", "0")),
            perigee_alt_km=sf(g("c_h_per", "0")),
            span_days=sf(g("c_span", "0")),
        )

        t_metrics = ObjectMetrics(
            time_lastob_start=sf(g("t_time_lastob_start", "0")),
            time_lastob_end=sf(g("t_time_lastob_end", "0")),
            recommended_od_span=sf(g("t_recommended_od_span", "0")),
            actual_od_span=sf(g("t_actual_od_span", "0")),
            obs_available=si(g("t_obs_available", "0")),
            obs_used=si(g("t_obs_used", "0")),
            residuals_accepted=sf(g("t_residuals_accepted", "0")),
            weighted_rms=sf(g("t_weighted_rms", "0")),
            rcs_estimate=sf(g("t_rcs_estimate", "0")),
            cd_area_over_mass=sf(g("t_cd_area_over_mass", "0")),
            cr_area_over_mass=sf(g("t_cr_area_over_mass", "0")),
            sedr=sf(g("t_sedr", "0")),
        )
        c_metrics = ObjectMetrics(
            time_lastob_start=sf(g("c_time_lastob_start", "0")),
            time_lastob_end=sf(g("c_time_lastob_end", "0")),
            recommended_od_span=sf(g("c_recommended_od_span", "0")),
            actual_od_span=sf(g("c_actual_od_span", "0")),
            obs_available=si(g("c_obs_available", "0")),
            obs_used=si(g("c_obs_used", "0")),
            residuals_accepted=sf(g("c_residuals_accepted", "0")),
            weighted_rms=sf(g("c_weighted_rms", "0")),
            rcs_estimate=sf(g("c_rcs_estimate", "0")),
            cd_area_over_mass=sf(g("c_cd_area_over_mass", "0")),
            cr_area_over_mass=sf(g("c_cr_area_over_mass", "0")),
            sedr=sf(g("c_sedr", "0")),
        )

        return CDMRecord(
            row_id=row_idx,
            event_id=si(g("event_id", "0")),
            mission_id=si(g("mission_id", "0")),
            time_to_tca_days=sf(g("time_to_tca", "0")),
            risk=sf(g("risk", "-99")),
            max_risk_estimate=sf(g("max_risk_estimate", "-99")),
            max_risk_scaling=sf(g("max_risk_scaling", "0")),
            miss_distance_m=sf(g("miss_distance", "0")),
            relative_speed_ms=sf(g("relative_speed", "0")),
            rel_pos_r=sf(g("relative_position_r", "0")),
            rel_pos_t=sf(g("relative_position_t", "0")),
            rel_pos_n=sf(g("relative_position_n", "0")),
            rel_vel_r=sf(g("relative_velocity_r", "0")),
            rel_vel_t=sf(g("relative_velocity_t", "0")),
            rel_vel_n=sf(g("relative_velocity_n", "0")),
            geocentric_latitude=sf(g("geocentric_latitude", "0")),
            azimuth=sf(g("azimuth", "0")),
            elevation=sf(g("elevation", "0")),
            mahalanobis_distance=sf(g("mahalanobis_distance", "0")),
            t_orbital=t_orbital,
            t_metrics=t_metrics,
            t_covariance=CovarianceMatrix(matrix=t_cov_mat),
            t_sigma_r=sf(g("t_sigma_r", "0")),
            t_sigma_t=sf(g("t_sigma_t", "0")),
            t_sigma_n=sf(g("t_sigma_n", "0")),
            t_sigma_rdot=sf(g("t_sigma_rdot", "0")),
            t_sigma_tdot=sf(g("t_sigma_tdot", "0")),
            t_sigma_ndot=sf(g("t_sigma_ndot", "0")),
            t_pos_cov_det=sf(g("t_position_covariance_det", "0")),
            c_object_type=g("c_object_type", "UNKNOWN").strip(),
            c_orbital=c_orbital,
            c_metrics=c_metrics,
            c_covariance=CovarianceMatrix(matrix=c_cov_mat),
            c_sigma_r=sf(g("c_sigma_r", "0")),
            c_sigma_t=sf(g("c_sigma_t", "0")),
            c_sigma_n=sf(g("c_sigma_n", "0")),
            c_sigma_rdot=sf(g("c_sigma_rdot", "0")),
            c_sigma_tdot=sf(g("c_sigma_tdot", "0")),
            c_sigma_ndot=sf(g("c_sigma_ndot", "0")),
            c_pos_cov_det=sf(g("c_position_covariance_det", "0")),
            space_weather=SpaceWeather(
                F10=sf(g("F10", "0")),
                F3M=sf(g("F3M", "0")),
                SSN=sf(g("SSN", "0")),
                AP=sf(g("AP", "0")),
            ),
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = CDMParser()
    file_path = r"C:\Users\Lkd\Desktop\proj\cdm\test_data.csv"
    try:
        records = parser.parse_file(file_path)

        print(f"\nParsed {len(records)} records\n")

        # Show first record as a sanity check
        if records:
            first = records[0]

            print("=== FIRST RECORD SUMMARY ===")
            print(f"Event ID: {first.event_id}")
            print(f"Miss distance (km): {first.miss_distance_km}")
            print(f"Relative speed (km/s): {first.relative_speed_kms}")
            print(f"Risk level: {first.risk_level}")
            print(f"Pc: {first.pc_value:.3e}")

            # Covariance sanity checks
            print("\n--- Covariance checks ---")
            print("Target positive definite:",
                  first.t_covariance.is_positive_definite())
            print("Chaser positive definite:",
                  first.c_covariance.is_positive_definite())
            print("Combined positive definite:",
                  first.combined_covariance.is_positive_definite())

    except Exception as e:
        print(f"Error: {e}")

   