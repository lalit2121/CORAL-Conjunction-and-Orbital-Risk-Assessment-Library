"""
CASAS – GMAT Interface (RTN-Corrected + Robust SMA Handling)

Fixes applied:
  1. Create ImpulsiveBurn moved BEFORE BeginMissionSequence
     (GMAT requires all Create statements in the initialization block)
  2. SMA validation applied to BOTH target and chaser objects
  3. Chaser SMA fallback offset from target to preserve relative geometry
"""

from __future__ import annotations
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np

from CDM_parser import CDMRecord
from Collision_Probability import PcResult, ManeuverRecommendation


BASE_DIR = Path(__file__).resolve().parent
GMAT_OUTPUT_DIR = BASE_DIR / "gmat" / "output"

MU_EARTH = 398600.4418  # km^3/s^2


# =========================================================
# TIME HELPERS
# =========================================================
def _epoch_utc(days: float) -> str:
    t = datetime.now(timezone.utc) + timedelta(days=days)
    return t.strftime("%d %b %Y %H:%M:%S.000")


def _end_epoch_utc(days: float, extra: float = 0.3) -> str:
    t = datetime.now(timezone.utc) + timedelta(days=days + extra)
    return t.strftime("%d %b %Y %H:%M:%S.000")


def clamp_area(v: float) -> float:
    return max(0.001, min(v, 1000.0))


# =========================================================
# ORBITAL ELEMENT VALIDATION
# =========================================================
def _validate_orbital(orb, fallback_sma: float = 7200.0, label: str = "object"):
    """
    Return a (sma_km, ecc, inc_deg) triple, applying fallbacks for invalid values.
    Uses the apogee/perigee alt fields as a secondary source for SMA if j2k_sma is bad.
    """
    sma = float(orb.sma_km)
    ecc = float(orb.eccentricity)
    inc = float(orb.inclination_deg)

    # --- Try to recover SMA from apogee/perigee altitudes if available ---
    apo = float(orb.apogee_alt_km)
    per = float(orb.perigee_alt_km)
    RE  = 6378.137  # km

    if sma <= 100.0 and apo > 100.0 and per > 0.0:
        # h_apo and h_per are altitudes → add Earth radius
        r_apo = apo + RE
        r_per = per + RE
        sma   = (r_apo + r_per) / 2.0
        ecc   = (r_apo - r_per) / (r_apo + r_per)
        print(f"  ℹ️  Recovered {label} SMA from apo/per: {sma:.1f} km  e={ecc:.4f}")

    # --- Hard fallback ---
    if sma <= 100.0:
        print(f"  ⚠️  {label} SMA still invalid ({sma:.1f} km) → using fallback {fallback_sma:.1f} km")
        sma = fallback_sma
        ecc = max(ecc, 0.001) if ecc > 0 else 0.001

    # --- Inclination fallback ---
    if inc <= 0.0 or inc > 180.0:
        inc = 51.6  # ISS-like default

    # --- Eccentricity clamp ---
    ecc = max(0.0, min(ecc, 0.999))

    return sma, ecc, inc


# =========================================================
# ORBIT CONVERSION
# =========================================================
def keplerian_to_cartesian(a, e, inc_deg, raan=0.0, aop=0.0, ta=0.0):
    a       = float(a)
    e       = float(e)
    inc_deg = float(inc_deg)

    if a <= 0:
        raise ValueError(f"Cannot convert orbit: semi-major axis = {a} km")

    inc = np.deg2rad(inc_deg)
    p   = a * (1.0 - e**2)
    r   = p / (1.0 + e * np.cos(ta))

    r_pqw = np.array([r * np.cos(ta), r * np.sin(ta), 0.0])
    v_pqw = np.array([
        -np.sqrt(MU_EARTH / p) * np.sin(ta),
         np.sqrt(MU_EARTH / p) * (e + np.cos(ta)),
         0.0,
    ])

    R3_W = np.array([[ np.cos(aop), -np.sin(aop), 0],
                     [ np.sin(aop),  np.cos(aop), 0],
                     [ 0,           0,            1]])
    R1_i = np.array([[1, 0,           0          ],
                     [0, np.cos(inc), -np.sin(inc)],
                     [0, np.sin(inc),  np.cos(inc)]])
    R3_O = np.array([[ np.cos(raan), -np.sin(raan), 0],
                     [ np.sin(raan),  np.cos(raan), 0],
                     [ 0,            0,             1]])

    Q = R3_O @ R1_i @ R3_W
    return Q @ r_pqw, Q @ v_pqw


def build_rtn_frame(r, v):
    R_hat = r / np.linalg.norm(r)
    h     = np.cross(r, v)
    N_hat = h / np.linalg.norm(h)
    T_hat = np.cross(N_hat, R_hat)
    return np.column_stack((R_hat, T_hat, N_hat))


def build_chaser_state(rec, r1, v1, Q):
    # rel_pos/vel in CDM are stored in metres → convert to km
    dr_rtn = np.array([rec.rel_pos_r, rec.rel_pos_t, rec.rel_pos_n]) / 1000.0
    dv_rtn = np.array([rec.rel_vel_r, rec.rel_vel_t, rec.rel_vel_n]) / 1000.0
    return r1 + Q @ dr_rtn, v1 + Q @ dv_rtn


# =========================================================
# MAIN ENTRY
# =========================================================
def generate_gmat_script(
    rec: CDMRecord,
    pc: Optional[PcResult] = None,
    mnv: Optional[ManeuverRecommendation] = None,
    output_dir: str | Path = GMAT_OUTPUT_DIR,
) -> Path:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path   = output_dir / f"event_{rec.event_id}_mission_{rec.mission_id}.script"
    script = _build_script(rec, pc, mnv)

    path.write_text(script, encoding="ascii", errors="ignore")
    print(f"GMAT script written → {path}")
    return path


# =========================================================
# SCRIPT BUILDER
# =========================================================
def _build_script(rec, pc, mnv):

    print(f"\nGenerating GMAT script for Event {rec.event_id} / Mission {rec.mission_id}")

    # ------------------------------------------------------------------
    # Validate & recover orbital elements for BOTH objects
    # ------------------------------------------------------------------
    t_sma, t_ecc, t_inc = _validate_orbital(rec.t_orbital, fallback_sma=7200.0, label="Target")
    # Chaser fallback uses same SMA as target (close approach → similar altitude)
    c_sma, c_ecc, c_inc = _validate_orbital(rec.c_orbital, fallback_sma=t_sma,   label="Chaser")

    r1, v1 = keplerian_to_cartesian(t_sma, t_ecc, t_inc)
    Q      = build_rtn_frame(r1, v1)
    r2, v2 = build_chaser_state(rec, r1, v1, Q)

    # ------------------------------------------------------------------
    # Area models
    # ------------------------------------------------------------------
    t_drag = clamp_area(rec.t_metrics.cd_area_over_mass * 500)
    t_srp  = clamp_area(rec.t_metrics.cr_area_over_mass * 500)
    c_drag = clamp_area(rec.c_metrics.cd_area_over_mass * 200)
    c_srp  = clamp_area(rec.c_metrics.cr_area_over_mass * 200)

    epoch     = _epoch_utc(rec.time_to_tca_days)
    prop_days = max(rec.time_to_tca_days * 1.3, 2.0)
    cam_dist  = int(max(np.linalg.norm(r1), np.linalg.norm(r2)) * 4)

    # ------------------------------------------------------------------
    # Maneuver parameters (computed up-front for use in Create block)
    # ------------------------------------------------------------------
    include_maneuver = mnv and getattr(mnv, "recommended", False)
    dv_km_s = (mnv.delta_v_ms / 1000.0) if include_maneuver else 0.0

    # ------------------------------------------------------------------
    # Helper: spacecraft block
    # ------------------------------------------------------------------
    def sc_block(name, r, v, drag, srp):
        return [
            f"Create Spacecraft {name};",
            f"{name}.DateFormat = UTCGregorian;",
            f"{name}.Epoch = '{epoch}';",
            f"{name}.CoordinateSystem = EarthMJ2000Eq;",
            f"{name}.DisplayStateType = Cartesian;",
            f"{name}.X  = {r[0]:.6f};",
            f"{name}.Y  = {r[1]:.6f};",
            f"{name}.Z  = {r[2]:.6f};",
            f"{name}.VX = {v[0]:.9f};",
            f"{name}.VY = {v[1]:.9f};",
            f"{name}.VZ = {v[2]:.9f};",
            f"{name}.DragArea = {drag:.4f};",
            f"{name}.SRPArea  = {srp:.4f};",
            "",
        ]

    lines = []

    # ==================== INITIALIZATION BLOCK ====================
    # ALL Create statements must appear before BeginMissionSequence.

    # Header
    lines += [
        "%=============================================================",
        "% CASAS - GMAT Conjunction Simulation Script",
        f"% Event ID    : {rec.event_id}",
        f"% Mission ID  : {rec.mission_id}",
        f"% Target SMA  : {t_sma:.1f} km  e={t_ecc:.4f}  i={t_inc:.2f} deg",
        f"% Chaser SMA  : {c_sma:.1f} km  e={c_ecc:.4f}  i={c_inc:.2f} deg",
        f"% TCA in      : {rec.time_to_tca_days:.3f} days",
        "%=============================================================",
        "",
    ]

    # Force Model
    lines += [
        "Create ForceModel HiFiForces;",
        "HiFiForces.CentralBody = Earth;",
        "HiFiForces.PrimaryBodies = {Earth};",
        "HiFiForces.GravityField.Earth.Degree = 20;",
        "HiFiForces.GravityField.Earth.Order  = 20;",
        "HiFiForces.GravityField.Earth.PotentialFile = 'EGM96.cof';",
        "HiFiForces.Drag.AtmosphereModel = MSISE90;",
        "HiFiForces.PointMasses = {Sun, Luna};",
        "HiFiForces.SRP = On;",
        "",
    ]

    # Propagator
    lines += [
        "Create Propagator HighFidelity;",
        "HighFidelity.FM = HiFiForces;",
        "HighFidelity.Type = RungeKutta89;",
        "HighFidelity.InitialStepSize = 60;",
        "HighFidelity.Accuracy = 1e-12;",
        "HighFidelity.MinStep = 0.001;",
        "HighFidelity.MaxStep = 86400;",
        "",
    ]

    # Spacecraft (both must be created here, in the init block)
    lines += sc_block("SAT1", r1, v1, t_drag, t_srp)
    lines += sc_block("SAT2", r2, v2, c_drag, c_srp)

    # *** FIX: ImpulsiveBurn Create goes HERE, BEFORE BeginMissionSequence ***
    if include_maneuver:
        lines += [
            "% --- Avoidance maneuver (applied to SAT1 at mission start) ---",
            "Create ImpulsiveBurn AvoidBurn;",
            "AvoidBurn.CoordinateSystem = Local;",
            "AvoidBurn.Origin = Earth;",
            "AvoidBurn.Axes   = VNB;",          # Velocity-Normal-Binormal frame
            f"AvoidBurn.Element1 = {dv_km_s:.9f};",   # along-track (V)
            "AvoidBurn.Element2 = 0;",
            "AvoidBurn.Element3 = 0;",
            "",
        ]

    # Visualization (also an init-block Create)
    lines += [
        "Create OrbitView DefaultView;",
        "DefaultView.Add = {SAT1, SAT2, Earth};",
        "DefaultView.CoordinateSystem = EarthMJ2000Eq;",
        "DefaultView.ViewPointReference = Earth;",
        f"DefaultView.ViewPointVector = [0 0 {cam_dist}];",
        "DefaultView.ViewDirection = Earth;",
        "DefaultView.ViewUpAxis = Z;",
        "DefaultView.ShowLabels = true;",
        "DefaultView.OrbitColor = [ 255 0 0 ; 0 0 255 ];",  # SAT1=red, SAT2=blue
        "",
        "Create GroundTrackPlot GroundTrack;",
        "GroundTrack.Add = {SAT1, SAT2};",
        "GroundTrack.CentralBody = Earth;",
        "",
    ]

    # ==================== MISSION SEQUENCE ====================
    lines += ["BeginMissionSequence;", ""]

    if include_maneuver:
        lines += [
            "% Apply avoidance burn to SAT1",
            "Maneuver AvoidBurn(SAT1);",
            "",
        ]

    lines += [
        f"% Propagate both satellites for {prop_days:.2f} days",
        f"Propagate HighFidelity(SAT1, SAT2) {{SAT1.ElapsedDays = {prop_days:.4f}}};",
        "",
    ]

    return "\n".join(lines)


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    from CDM_parser import CDMParser
    parser    = CDMParser()
    test_file = BASE_DIR / "test_data.csv"

    if test_file.exists():
        rec = parser.parse_file(test_file)[0]
        generate_gmat_script(rec)
    else:
        print("Test file not found.")