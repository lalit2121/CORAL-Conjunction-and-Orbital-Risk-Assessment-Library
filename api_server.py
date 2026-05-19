"""
CASAS – FastAPI REST Server
============================
Production REST API for the Collision Alert & Situational Awareness System.

Endpoints:
  GET  /                      – HTML landing page
  GET  /health                – System health + DB stats
  GET  /api/events            – All conjunction events (aggregated)
  GET  /api/events/{id}       – Single event CDM series
  GET  /api/events/{id}/trend – Risk trend time series
  GET  /api/events/{id}/pc    – Multi-method Pc computation
  GET  /api/events/{id}/gmat  – Download GMAT script
  GET  /api/records           – Raw CDM records (paginated)
  GET  /api/critical          – Critical + high-risk events
  POST /api/ingest            – Ingest new CDM CSV file
  GET  /api/alerts            – Alert log
  POST /api/alerts/{id}/ack   – Acknowledge alert
  GET  /api/stats             – Database statistics
  GET  /api/export/csv        – Export CDM records as CSV
  GET  /api/export/json       – Export CDM records as JSON
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field

from CDM_database import CDMDatabase
from CDM_parser import CDMParser

from Collision_Probability import compute_all_pc, recommend_maneuver
from Risk_Trend import RiskTrendAnalyser, AlertManager, RiskLevel
from GMAT_interface import generate_gmat_script
from Analytics_vis import ExportUtils

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CASAS – Collision Alert & Situational Awareness System",
    description=(
        "Industrial-grade CDM analysis API. "
        "Computes collision probability via Alfano, Foster, Chan, and Monte Carlo methods. "
        "Provides risk trend analysis, GMAT script generation, and structured alerting."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_db: Optional[CDMDatabase] = None
_alert_manager = AlertManager()


def get_db() -> CDMDatabase:
    global _db
    if _db is None:
        _db = CDMDatabase()
    return _db


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    db_records: int
    db_events: int
    unacked_alerts: int
    version: str = "2.0.0"


class EventSummary(BaseModel):
    event_id: int
    mission_id: int
    cdm_count: int
    min_tca_days: float
    max_tca_days: float
    peak_risk: float
    max_pc: float
    min_miss_m: float
    risk_level: str


class TrendPoint(BaseModel):
    time_to_tca_days: float
    risk: float
    pc_value: float
    miss_distance_m: float
    mahalanobis_distance: float
    t_sigma_r: float
    c_sigma_r: float


class PcResponse(BaseModel):
    event_id: int
    pc_alfano: float
    pc_foster: float
    pc_chan: float
    pc_monte_carlo: float
    pc_consensus: float
    risk_level: str
    maneuver_recommended: bool
    maneuver_dv_ms: float
    maneuver_direction: str
    maneuver_rationale: str


class AlertResponse(BaseModel):
    alert_id: str
    event_id: int
    level: str
    title: str
    message: str
    pc_value: float
    miss_distance_m: float
    time_to_tca_h: float
    acknowledged: bool


class StatsResponse(BaseModel):
    total_records: int
    total_events: int
    critical_events: int
    high_events: int
    unacked_alerts: int


class IngestResponse(BaseModel):
    records_ingested: int
    events_detected: int
    critical_alerts: int
    filename: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Info"])
async def landing():
    return HTMLResponse(content=_landing_html(), status_code=200)


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    db = get_db()
    stats = db.stats()
    return HealthResponse(
        status="ok",
        db_records=stats["total_records"],
        db_events=stats["total_events"],
        unacked_alerts=stats["unacked_alerts"],
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_stats():
    return StatsResponse(**get_db().stats())


# ---- Events ---------------------------------------------------------------

@app.get("/api/events", response_model=List[EventSummary], tags=["Events"])
async def list_events():
    rows = get_db().get_events()
    return [
        EventSummary(
            event_id=r["event_id"],
            mission_id=r["mission_id"],
            cdm_count=r["cdm_count"],
            min_tca_days=r["min_tca_days"] or 0,
            max_tca_days=r["max_tca_days"] or 0,
            peak_risk=r["peak_risk"] or -99,
            max_pc=r["max_pc"] or 0,
            min_miss_m=r["min_miss_m"] or 0,
            risk_level=r["risk_level"] or "NEGLIGIBLE",
        )
        for r in rows
    ]


@app.get("/api/events/{event_id}", tags=["Events"])
async def get_event(event_id: int):
    rows = get_db().get_by_event(event_id)
    if not rows:
        raise HTTPException(404, f"Event {event_id} not found")
    return rows


@app.get("/api/events/{event_id}/trend", response_model=List[TrendPoint], tags=["Risk Trend"])
async def get_risk_trend(event_id: int):
    rows = get_db().get_risk_trend(event_id)
    if not rows:
        raise HTTPException(404, f"No trend data for event {event_id}")
    return [
        TrendPoint(
            time_to_tca_days=r["time_to_tca_days"],
            risk=r["risk"] or -99,
            pc_value=r["pc_value"] or 0,
            miss_distance_m=r["miss_distance_m"] or 0,
            mahalanobis_distance=r["mahalanobis_distance"] or 0,
            t_sigma_r=r["t_sigma_r"] or 0,
            c_sigma_r=r["c_sigma_r"] or 0,
        )
        for r in rows
    ]


@app.get("/api/events/{event_id}/pc", response_model=PcResponse, tags=["Collision Probability"])
async def get_collision_probability(
    event_id: int,
    fast: bool = Query(False, description="Skip Foster integration and MC for speed"),
):
    """
    Compute Pc for the most critical CDM of this event using four methods.
    """
    rows = get_db().get_by_event(event_id)
    if not rows:
        raise HTTPException(404, f"Event {event_id} not found")

    # Pick the most critical record (lowest risk = highest Pc)
    worst = min(rows, key=lambda r: r.get("risk", 0) or 0)

    # Rebuild CDMRecord from DB row
    parser = CDMParser()
    records = parser.parse_dataframe(_rows_to_df([worst]))
    if not records:
        raise HTTPException(500, "Failed to reconstruct CDM record")

    rec = records[0]
    pc_res = compute_all_pc(rec, fast=fast)
    mnv    = recommend_maneuver(rec, pc_res)

    return PcResponse(
        event_id=event_id,
        pc_alfano=pc_res.alfano,
        pc_foster=pc_res.foster,
        pc_chan=pc_res.chan,
        pc_monte_carlo=pc_res.monte_carlo,
        pc_consensus=pc_res.consensus,
        risk_level=pc_res.risk_level,
        maneuver_recommended=mnv.recommended,
        maneuver_dv_ms=mnv.delta_v_ms,
        maneuver_direction=mnv.maneuver_direction,
        maneuver_rationale=mnv.rationale,
    )


@app.get("/api/events/{event_id}/gmat", tags=["GMAT"])
async def download_gmat_script(event_id: int):
    """Auto-generate and download a GMAT simulation script for this event."""
    rows = get_db().get_by_event(event_id)
    if not rows:
        raise HTTPException(404, f"Event {event_id} not found")

    worst = min(rows, key=lambda r: r.get("risk", 0) or 0)
    parser = CDMParser()
    records = parser.parse_dataframe(_rows_to_df([worst]))
    if not records:
        raise HTTPException(500, "Failed to reconstruct CDM record")

    rec = records[0]
    pc_res = compute_all_pc(rec, fast=True)
    mnv    = recommend_maneuver(rec, pc_res)

    out_dir = Path("gmat/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = generate_gmat_script(rec, pc_res, mnv, out_dir)

    return FileResponse(
        path=str(script_path),
        media_type="text/plain",
        filename=script_path.name,
    )



# ---- Records ---------------------------------------------------------------

@app.get("/api/records", tags=["Records"])
async def list_records(
    limit: int = Query(100, ge=1, le=5000),
    risk_min: float = Query(-99, description="Minimum risk (log10 Pc) filter"),
):
    rows = get_db().get_all(limit=limit)
    if risk_min > -99:
        rows = [r for r in rows if (r.get("risk") or -99) >= risk_min]
    return rows


@app.get("/api/critical", tags=["Records"])
async def get_critical(threshold: float = Query(-5.0)):
    return get_db().get_critical_events(threshold_risk=threshold)


# ---- Ingest ----------------------------------------------------------------

@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CDM CSV/TSV file"),
):
    """Upload and ingest a new CDM data file."""
    suffix = Path(file.filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    parser = CDMParser()
    records = parser.parse_file(tmp_path)
    db = get_db()
    db.insert_batch(records, source_file=file.filename)

    # Fire alerts
    all_rows = db.get_all(limit=len(records) + 100)
    fired = _alert_manager.evaluate_batch(all_rows[-len(records):])
    # ← ADD THIS BLOCK RIGHT HERE
    for alert in fired:
        if alert.level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            db.log_alert(
                event_id=alert.event_id,
                level=alert.level.value,
                message=alert.message,
                pc=alert.pc_value,
                miss_m=alert.miss_distance_m,
                tca_h=alert.time_to_tca_h,
            )
    
    tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        records_ingested=len(records),
        events_detected=len(set(r.event_id for r in records)),
        critical_alerts=len([a for a in fired if a.level.value in ("CRITICAL","HIGH")]),
        filename=file.filename,
    )


# ---- Alerts ----------------------------------------------------------------

@app.get("/api/alerts", response_model=List[AlertResponse], tags=["Alerts"])
async def list_alerts(unacked_only: bool = Query(False)):
    rows = get_db().get_alerts(unacknowledged_only=unacked_only)
    return [
        AlertResponse(
            alert_id=str(r["id"]),
            event_id=r["event_id"],
            level=r["alert_level"],
            title=f"[{r['alert_level']}] Event {r['event_id']}",
            message=r["message"],
            pc_value=r["pc_value"] or 0,
            miss_distance_m=r["miss_distance_m"] or 0,
            time_to_tca_h=r["time_to_tca_h"] or 0,
            acknowledged=bool(r["acknowledged"]),
        )
        for r in rows
    ]


@app.post("/api/alerts/{alert_id}/ack", tags=["Alerts"])
async def acknowledge_alert(alert_id: int):
    get_db().acknowledge_alert(alert_id)
    return {"status": "acknowledged", "alert_id": alert_id}


# ---- Export ----------------------------------------------------------------

@app.get("/api/export/csv", tags=["Export"])
async def export_csv(limit: int = Query(5000)):
    rows = get_db().get_all(limit=limit)
    if not rows:
        raise HTTPException(404, "No data to export")

    import csv
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.read().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=casas_export.csv"},
    )


@app.get("/api/export/json", tags=["Export"])
async def export_json(limit: int = Query(5000)):
    rows = get_db().get_all(limit=limit)
    return JSONResponse(content=rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_to_df(rows: List[Dict]):
    import pandas as pd
    return pd.DataFrame(rows)


def _landing_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
  <title>CASAS API v2.0</title>
  <style>
    body{font-family:monospace;background:#0d0d1a;color:#e0e0e0;padding:40px;}
    h1{color:#3B82F6;} h2{color:#8B5CF6;}
    a{color:#60A5FA;} code{background:#1a1a2e;padding:2px 6px;border-radius:4px;}
    .card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;margin:8px 0;}
    .tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.8em;margin-right:4px;}
    .get{background:#1d4ed830;color:#60A5FA;border:1px solid #1d4ed8;}
    .post{background:#16533030;color:#34d399;border:1px solid #16a34a;}
  </style>
</head>
<body>
  <h1>🛰 CASAS – Collision Alert & Situational Awareness System</h1>
  <p>Industrial-grade CDM analysis API | v2.0.0</p>
  <h2>Quick Links</h2>
  <p><a href="/docs">📚 Swagger UI</a> &nbsp; <a href="/redoc">📖 ReDoc</a></p>
  <h2>Endpoints</h2>
  <div class="card"><span class="tag get">GET</span> <code>/health</code> – System status</div>
  <div class="card"><span class="tag get">GET</span> <code>/api/events</code> – All events</div>
  <div class="card"><span class="tag get">GET</span> <code>/api/events/{id}/pc</code> – Multi-method Pc</div>
  <div class="card"><span class="tag get">GET</span> <code>/api/events/{id}/trend</code> – Risk trend</div>
  <div class="card"><span class="tag get">GET</span> <code>/api/events/{id}/gmat</code> – Download GMAT script</div>
  <div class="card"><span class="tag post">POST</span> <code>/api/ingest</code> – Upload CDM CSV</div>
  <div class="card"><span class="tag get">GET</span> <code>/api/alerts</code> – Alert log</div>
  <div class="card"><span class="tag get">GET</span> <code>/api/export/csv</code> – Export data</div>
</body>
</html>"""

if __name__ == "__main__":
    import logging
    from pathlib import Path
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "test_data.csv"

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CDM file not found at: {DATA_PATH}")

    db = CDMDatabase.load_csv(str(DATA_PATH))
    _db = db

    # ← ADD THIS BLOCK RIGHT HERE
    all_rows = db.get_critical_events(threshold_risk=-5.0)  # HIGH + CRITICAL only
    fired = _alert_manager.evaluate_batch(all_rows)
    for alert in fired:
        if alert.level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            db.log_alert(
                event_id=alert.event_id,
                level=alert.level.value,
                message=alert.message,
                pc=alert.pc_value,
                miss_m=alert.miss_distance_m,
                tca_h=alert.time_to_tca_h,
            )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")