"""
CASAS – Risk Trend Analyser & Alert Manager
===========================================
Tracks how collision risk evolves across multiple CDM messages for the
same conjunction event (as TCA approaches), and fires structured alerts
when thresholds are crossed.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass,field
from Collision_Probability import compute_all_pc
from CDM_parser import CDMParser
import pandas as pd
from enum import Enum
from typing import List,Dict,Optional, Callable,Tuple
import numpy as np
log = logging.getLogger(__name__)


#---------------------------------------
#---------------Risk Level(NASA CARA / ESA)
#------------------------------------------

class RiskLevel(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW        = "LOW"
    MEDIUM     = "MEDIUM"
    HIGH       = "HIGH"
    CRITICAL   = "CRITICAL"

    @classmethod
    def from_log10_pc(cls, log_pc: float) -> "RiskLevel":
        if log_pc >= -4: return cls.CRITICAL
        if log_pc >= -5: return cls.HIGH
        if log_pc >= -6: return cls.MEDIUM
        if log_pc >= -7: return cls.LOW
        return cls.NEGLIGIBLE

    @classmethod
    def from_pc(cls, pc: float) -> "RiskLevel":
        if pc <= 0: return cls.NEGLIGIBLE
        return cls.from_log10_pc(math.log10(pc))
    
#------------------------------------
#Trand data points ----------------
# --------------------------------------
@dataclass
class TrendPoint:
    time_to_tca_days: float
    risk_log10:       float      # log10(Pc)
    pc_value:         float
    miss_distance_m:  float
    mahalanobis:      float
    sigma_r_t:        float      # target sigma R
    sigma_r_c:        float      # chaser sigma R
    risk_level:       RiskLevel


@dataclass
class RiskTrend:
    event_id:    int
    points:      List[TrendPoint]

    # Computed summaries
    peak_risk:   float = field(init=False)
    peak_pc:     float = field(init=False)
    risk_level:  RiskLevel = field(init=False)
    trend_slope: float = field(init=False)  # Pc rate of change per day
    is_increasing: bool = field(init=False)
    

    def __post_init__(self):
        if not self.points:
            self.peak_risk = -99.0
            self.peak_pc   = 0.0
            self.risk_level = RiskLevel.NEGLIGIBLE
            self.trend_slope = 0.0
            self.is_increasing = False
            return
        

        risks = [p.risk_log10 for p in self.points]
        self.peak_risk  = max(risks)
        self.peak_pc    = max(p.pc_value for p in self.points)
        self.risk_level = RiskLevel.from_log10_pc(self.peak_risk)

        # Compute slope in Pc vs time_to_tca (as TCA approaches, time decreases)
        if len(self.points) >= 2:
            tcas = np.array([p.time_to_tca_days for p in self.points])
            pcs  = np.array([p.pc_value for p in self.points])
            # positive slope = Pc increasing as TCA approaches
            slope = np.polyfit(tcas, pcs, 1)
            self.trend_slope   = float(slope[0])
            self.is_increasing = self.trend_slope < 0   # time decreasing → risk increasing
        else:
            self.trend_slope   = 0.0
            self.is_increasing = False
    def summary(self) -> Dict:
        return {
            "event_id":      self.event_id,
            "cdm_count":     len(self.points),
            "peak_risk":     self.peak_risk,
            "peak_pc":       self.peak_pc,
            "risk_level":    self.risk_level.value,
            "trend_slope":   self.trend_slope,
            "is_increasing": self.is_increasing,
            "min_tca_days":  min(p.time_to_tca_days for p in self.points),
            "max_tca_days":  max(p.time_to_tca_days for p in self.points),
        }
    
class RiskTrendAnalyser:
    """Compute risk trends from a list of CDM DB rows for a single event."""

    @staticmethod
    def from_db_rows(event_id: int, rows: List[Dict]) -> RiskTrend:
        points = []
        for r in rows:
            risk_val = float(r.get("risk", -99) or -99)
            pc_val   = float(r.get("pc_value", 0) or 0)
            pt = TrendPoint(
                time_to_tca_days=float(r.get("time_to_tca_days", 0) or 0),
                risk_log10=risk_val,
                pc_value=pc_val,
                miss_distance_m=float(r.get("miss_distance_m", 0) or 0),
                mahalanobis=float(r.get("mahalanobis_distance", 0) or 0),
                sigma_r_t=float(r.get("t_sigma_r", 0) or 0),
                sigma_r_c=float(r.get("c_sigma_r", 0) or 0),
                risk_level=RiskLevel.from_log10_pc(risk_val),
            )
            points.append(pt)

        return RiskTrend(event_id=event_id, points=sorted(points, key=lambda p: -p.time_to_tca_days))

    @staticmethod
    def predict_at_tca(trend: RiskTrend) -> Optional[float]:
        """
        Linear extrapolation of log10(Pc) to TCA (time_to_tca = 0).
        Returns predicted log10(Pc) at TCA.
        """
        if len(trend.points) < 2:
            return None
        tcas  = np.array([p.time_to_tca_days for p in trend.points])
        risks = np.array([p.risk_log10 for p in trend.points])
        coeffs = np.polyfit(tcas, risks, 1)
        return float(np.polyval(coeffs, 0.0))
    
#=====================================
#  Alert Manager =======================================
#=========================================
@dataclass
class Alert:
    alert_id:       str
    event_id:       int
    level:          RiskLevel
    title:          str
    message:        str
    pc_value:       float
    miss_distance_m: float
    time_to_tca_h:  float
    acknowledged:   bool = False

    def to_dict(self) -> Dict:
        return {
            "alert_id":       self.alert_id,
            "event_id":       self.event_id,
            "level":          self.level.value,
            "title":          self.title,
            "message":        self.message,
            "pc_value":       self.pc_value,
            "miss_distance_m": self.miss_distance_m,
            "time_to_tca_h":  self.time_to_tca_h,
            "acknowledged":   self.acknowledged,
        }
    
class AlertManager:
    DEFAULT_THRESHOLDS = {
        RiskLevel.CRITICAL:   1e-4,
        RiskLevel.HIGH:       1e-5,
        RiskLevel.MEDIUM:     1e-6,
        RiskLevel.LOW:        1e-7,
    }

    ACTIONABLE_LEVELS = {RiskLevel.HIGH, RiskLevel.CRITICAL}  # ← correct place

    def __init__(self, db=None):
        self._db = db
        self._callbacks: List[Callable[[Alert], None]] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_counter = 0

    def register_callback(self, fn: Callable[[Alert], None]):
        self._callbacks.append(fn)

    def evaluate(self, event_id: int, pc: float, miss_m: float,
                 tca_h: float, trend: Optional[RiskTrend] = None,
                 pc_alfano: float = None, pc_foster: float = None,
                 pc_chan: float = None, pc_monte_carlo: float = None) -> Optional[Alert]:

        level = RiskLevel.from_pc(pc)

        if level not in self.ACTIONABLE_LEVELS:  # ← uses class-level constant
            return None

        key = f"{event_id}:{level.value}"
        if key in self._active_alerts and not self._active_alerts[key].acknowledged:
            return self._active_alerts[key]

        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:05d}"

        trend_info = ""
        if trend and trend.is_increasing:
            predicted = RiskTrendAnalyser.predict_at_tca(trend)
            if predicted:
                trend_info = (
                    f" | TREND: Risk increasing – predicted Pc at TCA = "
                    f"10^{predicted:.1f} = {10**predicted:.2e}"
                )

        method_info = ""
        if any(v is not None for v in [pc_alfano, pc_foster, pc_chan, pc_monte_carlo]):
            parts = []
            if pc_alfano      is not None: parts.append(f"Alfano={pc_alfano:.2e}")
            if pc_foster      is not None: parts.append(f"Foster={pc_foster:.2e}")
            if pc_chan        is not None: parts.append(f"Chan={pc_chan:.2e}")
            if pc_monte_carlo is not None: parts.append(f"MC={pc_monte_carlo:.2e}")
            method_info = f" | Methods: {', '.join(parts)}"

        if level == RiskLevel.CRITICAL:
            risk_label = "🔴 CRITICAL – Immediate action required"
        else:
            risk_label = "🟠 HIGH – Maneuver assessment required"

        title = f"[{level.value}] Conjunction Event {event_id}"
        message = (
            f"Event {event_id} | Pc = {pc:.2e}"
            f" | Miss = {miss_m:.0f} m"
            f" | TCA in {tca_h:.1f} h"
            f" | {risk_label}"
            f"{method_info}"
            f"{trend_info}"
        )

        alert = Alert(
            alert_id=alert_id,
            event_id=event_id,
            level=level,
            title=title,
            message=message,
            pc_value=pc,
            miss_distance_m=miss_m,
            time_to_tca_h=tca_h,
        )

        self._active_alerts[key] = alert

        if self._db:
            try:
                self._db.log_alert(
                    event_id=event_id, level=level.value,
                    message=message, pc=pc,
                    miss_m=miss_m, tca_h=tca_h,
                )
            except Exception as exc:
                log.warning("DB alert log failed: %s", exc)

        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as exc:
                log.warning("Alert callback failed: %s", exc)

        log.warning("ALERT %s: %s", alert_id, message)
        return alert

    def evaluate_batch(self, rows: List[Dict]) -> List[Alert]:
        fired = []
        parser = CDMParser()

        for row in rows:
            pc    = float(row.get("pc_value", 0) or 0)
            ev_id = int(row.get("event_id", 0) or 0)
            miss  = float(row.get("miss_distance_m", 0) or 0)
            tca_h = float(row.get("time_to_tca_days", 0) or 0) * 24.0

            level = RiskLevel.from_pc(pc)
            if level not in self.ACTIONABLE_LEVELS:
                continue

            pc_alf = pc_fos = pc_ch = pc_mc = None
            try:
                records = parser.parse_dataframe(pd.DataFrame([row]))
                if records:
                    result = compute_all_pc(records[0], fast=True)
                    pc_alf = result.alfano
                    pc_fos = result.foster
                    pc_ch  = result.chan
                    pc_mc  = result.monte_carlo
            except Exception as e:
                log.warning("Pc computation failed for event %d: %s", ev_id, e)

            alert = self.evaluate(
                ev_id, pc, miss, tca_h,
                pc_alfano=pc_alf,
                pc_foster=pc_fos,
                pc_chan=pc_ch,
                pc_monte_carlo=pc_mc,
            )
            if alert:
                fired.append(alert)

        return fired

    def acknowledge(self, alert_id: str):
        for key, alert in self._active_alerts.items():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return

    @property
    def active_alerts(self) -> List[Alert]:
        return [a for a in self._active_alerts.values() if not a.acknowledged]

    @property
    def all_alerts(self) -> List[Alert]:
        return list(self._active_alerts.values())


if __name__ == "__main__":
    # Fake CDM rows
    rows = [
        {"event_id": 1, "time_to_tca_days": 3, "risk": -6, "pc_value": 1e-6, "miss_distance_m": 500},
        {"event_id": 1, "time_to_tca_days": 2, "risk": -5, "pc_value": 1e-5, "miss_distance_m": 300},
        {"event_id": 1, "time_to_tca_days": 1, "risk": -4, "pc_value": 1e-4, "miss_distance_m": 100},
    ]

    # --- Risk Trend ---
    trend = RiskTrendAnalyser.from_db_rows(1, rows)
    summary = trend.summary()

    print("\n=== RISK TREND SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # --- Alert Manager ---
    manager = AlertManager()

    print("\n=== ALERTS ===")
    alerts = manager.evaluate_batch(rows)

    for alert in alerts:
        print(alert.to_dict())

    print("\n=== ACTIVE ALERTS ===")
    for alert in manager.active_alerts:
        print(alert.to_dict())

