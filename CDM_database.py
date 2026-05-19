"""
CASAS – CDM Database
====================
SQLite persistence for CDM records with full-text indexing,
event grouping, and time-series retrieval for risk trend analysis.
"""
from __future__ import annotations
import json
import logging
import sqlite3
from typing import List, Optional, Dict, Any


import numpy as np

from CDM_parser import CDMRecord, CDMParser
log=logging.getLogger(__name__)

from pathlib import Path
import os

# === FORCE DATABASE TO SAVE IN YOUR PROJECT FOLDER ===
SCRIPT_DIR = Path(__file__).parent.absolute()   # This gets the folder of your script
DB_PATH = SCRIPT_DIR / "CDM_data.db"

print("✅ Database will be saved at:", DB_PATH)



# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cdm_records (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    row_id                  INTEGER,
    event_id                INTEGER,
    mission_id              INTEGER,
    time_to_tca_days        REAL,
    risk                    REAL,
    max_risk_estimate       REAL,
    max_risk_scaling        REAL,
    miss_distance_m         REAL,
    miss_distance_km        REAL,
    relative_speed_ms       REAL,
    rel_pos_r               REAL,
    rel_pos_t               REAL,
    rel_pos_n               REAL,
    rel_vel_r               REAL,
    rel_vel_t               REAL,
    rel_vel_n               REAL,
    pc_value                REAL,
    risk_level              TEXT,
    mahalanobis_distance    REAL,
    geocentric_latitude     REAL,
    azimuth                 REAL,
    elevation               REAL,
    -- Target
    t_sma_km                REAL,
    t_ecc                   REAL,
    t_inc_deg               REAL,
    t_apogee_km             REAL,
    t_perigee_km            REAL,
    t_sigma_r               REAL,
    t_sigma_t               REAL,
    t_sigma_n               REAL,
    t_sigma_rdot            REAL,
    t_sigma_tdot            REAL,
    t_sigma_ndot            REAL,
    t_pos_cov_det           REAL,
    t_obs_available         INTEGER,
    t_obs_used              INTEGER,
    t_weighted_rms          REAL,
    t_rcs_estimate          REAL,
    t_covariance_json       TEXT,
    -- Chaser
    c_object_type           TEXT,
    c_sma_km                REAL,
    c_ecc                   REAL,
    c_inc_deg               REAL,
    c_apogee_km             REAL,
    c_perigee_km            REAL,
    c_sigma_r               REAL,
    c_sigma_t               REAL,
    c_sigma_n               REAL,
    c_sigma_rdot            REAL,
    c_sigma_tdot            REAL,
    c_sigma_ndot            REAL,
    c_pos_cov_det           REAL,
    c_obs_available         INTEGER,
    c_obs_used              INTEGER,
    c_weighted_rms          REAL,
    c_rcs_estimate          REAL,
    c_covariance_json       TEXT,
    -- Combined covariance
    combined_covariance_json TEXT,
    -- Space weather
    F10                     REAL,
    F3M                     REAL,
    SSN                     REAL,
    AP                      REAL,
    -- Derived Pc methods (computed on insert)
    pc_alfano               REAL,
    pc_foster               REAL,
    pc_chan                 REAL,
    pc_monte_carlo          REAL,
    -- Maneuver recommendation
    maneuver_recommended    INTEGER DEFAULT 0,
    maneuver_dv_ms          REAL,
    -- Ingestion timestamp
    ingested_at             TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_event_id      ON cdm_records(event_id);
CREATE INDEX IF NOT EXISTS idx_mission_id    ON cdm_records(mission_id);
CREATE INDEX IF NOT EXISTS idx_risk          ON cdm_records(risk);
CREATE INDEX IF NOT EXISTS idx_tca           ON cdm_records(time_to_tca_days);
CREATE INDEX IF NOT EXISTS idx_risk_level    ON cdm_records(risk_level);
CREATE INDEX IF NOT EXISTS idx_event_tca     ON cdm_records(event_id, time_to_tca_days);

-- Alert log table
CREATE TABLE IF NOT EXISTS alert_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER,
    alert_level     TEXT,
    message         TEXT,
    pc_value        REAL,
    miss_distance_m REAL,
    time_to_tca_h   REAL,
    acknowledged    INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- File ingestion tracking
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT UNIQUE,
    records_loaded  INTEGER,
    loaded_at       TEXT DEFAULT (datetime('now'))
);
"""

class CDMDatabase:
    """SQLite repository for CDM records with high-performance batch operations."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
    # ------------------------------------------------------------------ #
    # Insert
    # ------------------------------------------------------------------ #
    def insert_batch(self, records: List[CDMRecord], source_file: str = ""):
        """Bulk insert CDM records. Returns number inserted."""
        if not records:
            return 0

        chunk_size = 1000
        total_inserted = 0
        
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            rows = [self._to_row(r) for r in chunk]
            keys = list(rows[0].keys())
            placeholders = ",".join(["?"] * len(keys))
            sql = f"INSERT OR REPLACE INTO cdm_records ({','.join(keys)}) VALUES ({placeholders})"

            with self._conn() as conn:
                conn.executemany(sql, [tuple(r[k] for k in keys) for r in rows])
            
            total_inserted += len(chunk)
            log.info("Inserted %d/%d records", total_inserted, len(records))
        
        # Log to ingestion_log after all chunks
        with self._conn() as conn:
            if source_file:
                conn.execute(
                    "INSERT OR REPLACE INTO ingestion_log(filename, records_loaded) VALUES (?,?)",
                    (source_file, len(records)),
                )
        
        return total_inserted

    def _to_row(self, r: CDMRecord) -> Dict[str, Any]:
        cov_t   = json.dumps(r.t_covariance.matrix.tolist())
        cov_c   = json.dumps(r.c_covariance.matrix.tolist())
        cov_cmb = json.dumps(r.combined_covariance.matrix.tolist())
     
        return {
            "row_id":                  r.row_id,
            "event_id":                r.event_id,
            "mission_id":              r.mission_id,
            "time_to_tca_days":        r.time_to_tca_days,
            "risk":                    r.risk,
            "max_risk_estimate":       r.max_risk_estimate,
            "max_risk_scaling":        r.max_risk_scaling,
            "miss_distance_m":         r.miss_distance_m,
            "miss_distance_km":        r.miss_distance_km,
            "relative_speed_ms":       r.relative_speed_ms,
            "rel_pos_r":               r.rel_pos_r,
            "rel_pos_t":               r.rel_pos_t,
            "rel_pos_n":               r.rel_pos_n,
            "rel_vel_r":               r.rel_vel_r,
            "rel_vel_t":               r.rel_vel_t,
            "rel_vel_n":               r.rel_vel_n,
            "pc_value":                r.pc_value,
            "risk_level":              r.risk_level,
            "mahalanobis_distance":    r.mahalanobis_distance,
            "geocentric_latitude":     r.geocentric_latitude,
            "azimuth":                 r.azimuth,
            "elevation":               r.elevation,
            "t_sma_km":                r.t_orbital.sma_km,
            "t_ecc":                   r.t_orbital.eccentricity,
            "t_inc_deg":               r.t_orbital.inclination_deg,
            "t_apogee_km":             r.t_orbital.apogee_alt_km,
            "t_perigee_km":            r.t_orbital.perigee_alt_km,
            "t_sigma_r":               r.t_sigma_r,
            "t_sigma_t":               r.t_sigma_t,
            "t_sigma_n":               r.t_sigma_n,
            "t_sigma_rdot":            r.t_sigma_rdot,
            "t_sigma_tdot":            r.t_sigma_tdot,
            "t_sigma_ndot":            r.t_sigma_ndot,
            "t_pos_cov_det":           r.t_pos_cov_det,
            "t_obs_available":         r.t_metrics.obs_available,
            "t_obs_used":              r.t_metrics.obs_used,
            "t_weighted_rms":          r.t_metrics.weighted_rms,
            "t_rcs_estimate":          r.t_metrics.rcs_estimate,
            "t_covariance_json":       cov_t,
            "c_object_type":           r.c_object_type,
            "c_sma_km":                r.c_orbital.sma_km,
            "c_ecc":                   r.c_orbital.eccentricity,
            "c_inc_deg":               r.c_orbital.inclination_deg,
            "c_apogee_km":             r.c_orbital.apogee_alt_km,
            "c_perigee_km":            r.c_orbital.perigee_alt_km,
            "c_sigma_r":               r.c_sigma_r,
            "c_sigma_t":               r.c_sigma_t,
            "c_sigma_n":               r.c_sigma_n,
            "c_sigma_rdot":            r.c_sigma_rdot,
            "c_sigma_tdot":            r.c_sigma_tdot,
            "c_sigma_ndot":            r.c_sigma_ndot,
            "c_pos_cov_det":           r.c_pos_cov_det,
            "c_obs_available":         r.c_metrics.obs_available,
            "c_obs_used":              r.c_metrics.obs_used,
            "c_weighted_rms":          r.c_metrics.weighted_rms,
            "c_rcs_estimate":          r.c_metrics.rcs_estimate,
            "c_covariance_json":       cov_c,
            "combined_covariance_json": cov_cmb,
            "F10":                     r.space_weather.F10,
            "F3M":                     r.space_weather.F3M,
            "SSN":                     r.space_weather.SSN,
            "AP":                      r.space_weather.AP,
            "pc_alfano":               None,
            "pc_foster":               None,
            "pc_chan":                  None,
            "pc_monte_carlo":           None,
            "maneuver_recommended":    0,
            "maneuver_dv_ms":          None,
        }

     # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def get_all(self, limit: int = 5000) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cdm_records ORDER BY risk DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_event(self, event_id: int) -> List[Dict]:
        """Return all CDMs for a given event, sorted by time to TCA descending."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cdm_records WHERE event_id=? ORDER BY time_to_tca_days DESC",
                (event_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events(self) -> List[Dict]:
        """Aggregate view: one row per event_id with peak risk."""
        sql = """
            SELECT
                event_id,
                mission_id,
                COUNT(*)              AS cdm_count,
                MIN(time_to_tca_days) AS min_tca_days,
                MAX(time_to_tca_days) AS max_tca_days,
                MIN(risk)             AS peak_risk,
                MIN(miss_distance_m)  AS min_miss_m,
                MAX(pc_value)         AS max_pc,
                MAX(risk_level)       AS risk_level
            FROM cdm_records
            GROUP BY event_id
            ORDER BY peak_risk DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_critical_events(self, threshold_risk: float = -5.0) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cdm_records WHERE risk >= ? ORDER BY risk DESC",
                (threshold_risk,),
            ).fetchall()
        return [dict(r) for r in rows]
    
    def get_risk_trend(self, event_id: int) -> List[Dict]:
        """Time series of risk evolution for a single conjunction event."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT time_to_tca_days, risk, pc_value, miss_distance_m,
                          mahalanobis_distance, t_sigma_r, c_sigma_r
                   FROM cdm_records WHERE event_id=?
                   ORDER BY time_to_tca_days DESC""",
                (event_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    
    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM cdm_records").fetchone()[0]

    def update_pc_methods(self, db_id: int, alfano: float, foster: float,
                          chan: float, mc: float):
        with self._conn() as conn:
            conn.execute(
                """UPDATE cdm_records
                   SET pc_alfano=?, pc_foster=?, pc_chan=?, pc_monte_carlo=?
                   WHERE id=?""",
                (alfano, foster, chan, mc, db_id),
            )

    def log_alert(self, event_id: int, level: str, message: str,
                  pc: float, miss_m: float, tca_h: float):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO alert_log(event_id,alert_level,message,pc_value,
                   miss_distance_m,time_to_tca_h) VALUES (?,?,?,?,?,?)""",
                (event_id, level, message, pc, miss_m, tca_h),
            )

    def get_alerts(self, unacknowledged_only: bool = False) -> List[Dict]:
        sql = "SELECT * FROM alert_log"
        if unacknowledged_only:
            sql += " WHERE acknowledged=0"
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int):
        with self._conn() as conn:
            conn.execute("UPDATE alert_log SET acknowledged=1 WHERE id=?", (alert_id,))

    def stats(self) -> Dict:
        with self._conn() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM cdm_records").fetchone()[0]
            events  = conn.execute("SELECT COUNT(DISTINCT event_id) FROM cdm_records").fetchone()[0]
            alerts  = conn.execute("SELECT COUNT(*) FROM alert_log WHERE acknowledged=0").fetchone()[0]
            # NEW: Count only HIGH + CRITICAL actionable alerts
            actionable_alerts = conn.execute(
                "SELECT COUNT(*) FROM alert_log WHERE acknowledged=0 AND alert_level IN ('HIGH','CRITICAL')"
            ).fetchone()[0]
            critical = conn.execute(
                "SELECT COUNT(DISTINCT event_id) FROM cdm_records WHERE risk >= -4"
            ).fetchone()[0]
            high = conn.execute(
                "SELECT COUNT(DISTINCT event_id) FROM cdm_records WHERE risk >= -5 AND risk < -4"
            ).fetchone()[0]
        return {
            "total_records": total,
            "total_events": events,
            "critical_events": critical,
            "high_events": high,
            "unacked_alerts": alerts,
            "actionable_alerts": actionable_alerts,  # ← ADD THIS
        }
    
    def clear_all_data(self):
        """Clear all records and alerts - Use with caution"""
        with self._conn() as conn:
            conn.execute("DELETE FROM cdm_records")
            conn.execute("DELETE FROM alert_log")
            conn.execute("DELETE FROM ingestion_log")
            conn.commit()
        
        # Reset any internal caches if needed
        if hasattr(self, '_event_cache'):
            self._event_cache = None

    # ------------------------------------------------------------------ #
    # Convenience: load from CSV directly
    # ------------------------------------------------------------------ #
    @classmethod
    def load_csv(cls, csv_path: str | Path, db_path: str | Path = DB_PATH) -> "CDMDatabase":
        db = cls(db_path)
        parser = CDMParser()
        records = parser.parse_file(csv_path)
        db.insert_batch(records, source_file=str(csv_path))
        return db
    

if __name__ == "__main__":
    db = CDMDatabase()

    csv_path = r"C:\Users\Lkd\Desktop\proj\cdm\test_data.csv"

    parser = CDMParser()
    records = parser.parse_file(csv_path)

    print("Parsed records:", len(records))   # 🔍 DEBUG

    inserted = db.insert_batch(records, source_file=csv_path)
    print("Inserted:", inserted)

    print("Total records:", db.count())

