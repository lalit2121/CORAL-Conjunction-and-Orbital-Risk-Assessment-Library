"""
CASAS – Streamlit Dashboard (Horizontal Pro Layout)
Fixed & Optimized Variant
"""

import sys
import os
from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CDM_database import CDMDatabase, DB_PATH
from CDM_parser import CDMParser

from Analytics_vis import (
    plot_risk_distribution,
    plot_orbital_regime,
    plot_space_weather,
    plot_pc_comparison,
)

from Risk_Trend import RiskTrendAnalyser
from Collision_Probability import compute_all_pc, recommend_maneuver


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CORAL Dashboard",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "ingest_token" not in st.session_state:
    st.session_state.ingest_token = 0


# ----------------------------------------------------------------------
# DB
# ----------------------------------------------------------------------
@st.cache_resource
def get_db():
    return CDMDatabase(DB_PATH)

db = get_db()


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
@st.cache_data(ttl=120)
def load_events(_token):
    res = db.get_events()
    return pd.DataFrame(res) if res else pd.DataFrame()

@st.cache_data(ttl=120)
def load_records(_token):
    res = db.get_all(limit=2000)
    return pd.DataFrame(res) if res else pd.DataFrame()

@st.cache_data(ttl=120)
def load_stats(_token):
    return db.stats() or {}


events_df = load_events(st.session_state.ingest_token)
records_df = load_records(st.session_state.ingest_token)
stats = load_stats(st.session_state.ingest_token)


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    img_path = Path(__file__).parent / "cdmssa.png"

    if img_path.exists():
        st.image(str(img_path), width=120)

    st.title("CORAL: Conjunction Risk Assessment")
    st.caption("High-fidelity collision probability engine for conjunction data messages.")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 CDM Analysis",
            "📈 Risk Trends",
            "🎯 Collision Probability",
            "🌐 Geometry",
            "🛰️ GMAT",
            "📂 Data Management",
        ],
    )

    st.divider()
    st.subheader("📤 Upload")

    replace = st.checkbox("Replace existing data")

    uploaded = st.file_uploader(
        "Upload CDM",
        type=["csv", "txt"],
        key=f"upl_{st.session_state.ingest_token}",
    )

    if st.button("Process", disabled=(uploaded is None)):
        with st.spinner("Processing..."):
            if replace:
                db.clear_all_data()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                parser = CDMParser()
                records = parser.parse_file(tmp_path)
                inserted = db.insert_batch(records)

                st.cache_data.clear()
                st.session_state.ingest_token += 1
                st.success(f"{inserted} records loaded")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ----------------------------------------------------------------------
# OVERVIEW (HORIZONTAL)
# ----------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🛰 Mission Overview")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", stats.get("total_records", 0))
    c2.metric("Events", stats.get("total_events", 0))
    c3.metric("Critical", stats.get("critical_events", 0))
    c4.metric("High", stats.get("high_events", 0))

    st.divider()

    if not events_df.empty:
        # ROW 1
        left, right = st.columns([2, 3])

        with left:
            st.subheader("Risk Distribution")
            st.plotly_chart(
                plot_risk_distribution(events_df.to_dict("records")),
                use_container_width=True
            )

        with right:
            st.subheader("Events Table")
            sort_col = "peak_risk" if "peak_risk" in events_df.columns else events_df.columns[0]
            st.dataframe(
                events_df.sort_values(sort_col, ascending=False),
                use_container_width=True,
                height=450
            )

        st.divider()

        # ROW 2
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Orbital Regime")
            if not records_df.empty:
                st.plotly_chart(
                    plot_orbital_regime(records_df.to_dict("records")),
                    use_container_width=True
                )

        with c2:
            st.subheader("Space Weather")
            if not records_df.empty:
                st.plotly_chart(
                    plot_space_weather(records_df.to_dict("records")),
                    use_container_width=True
                )
    else:
        st.info("No event entries found. Upload raw CDM data files via the sidebar panel.")


# ----------------------------------------------------------------------
# CDM ANALYSIS
# ----------------------------------------------------------------------
elif page == "📊 CDM Analysis":
    st.title("CDM Analysis")

    if not records_df.empty:
        f1, f2, f3 = st.columns(3)

        with f1:
            risk_filter = st.selectbox("Risk", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

        with f2:
            raw_missions = records_df["mission_id"].dropna().unique().tolist()
            mission_options = ["ALL"] + sorted([str(m) for m in raw_missions])
            mission_filter = st.selectbox("Mission", mission_options)

        with f3:
            max_days = float(records_df["time_to_tca_days"].max()) if "time_to_tca_days" in records_df.columns else 10.0
            tca_max = st.slider("TCA Limit (Days)", 0.0, max(max_days, 5.0), 5.0)

        df = records_df.copy()

        if risk_filter != "ALL" and "risk_level" in df.columns:
            df = df[df["risk_level"] == risk_filter]

        if mission_filter != "ALL" and "mission_id" in df.columns:
            # Safe matching using string values to eliminate int-conversion exceptions
            df = df[df["mission_id"].astype(str) == mission_filter]

        if "time_to_tca_days" in df.columns:
            df = df[df["time_to_tca_days"] <= tca_max]

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records loaded to process filter criteria.")


# ----------------------------------------------------------------------
# RISK TRENDS
# ----------------------------------------------------------------------
elif page == "📈 Risk Trends":
    st.title("Risk Trends")

    if not events_df.empty and "event_id" in events_df.columns:
        event_id = st.selectbox("Select Event Tracker", events_df["event_id"].unique())
        rows = db.get_risk_trend(event_id)

        if rows:
            trend = RiskTrendAnalyser.from_db_rows(event_id, rows)

            c1, c2, c3 = st.columns(3)
            c1.metric("Peak Risk", f"{trend.peak_risk:.2f}")
            c2.metric("Peak Pc", f"{trend.peak_pc:.2e}")
            c3.metric("Level", getattr(trend.risk_level, 'value', str(trend.risk_level)))

            from Analytics_vis import plot_risk_trend, plot_mahalanobis

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(plot_risk_trend(rows, event_id), use_container_width=True)

            with col2:
                st.plotly_chart(plot_mahalanobis(rows), use_container_width=True)

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("Could not build trends — missing row values for this target.")
    else:
        st.info("No history logs available.")


# ----------------------------------------------------------------------
# COLLISION PROBABILITY
# ----------------------------------------------------------------------
elif page == "🎯 Collision Probability":
    st.title("Collision Probability Execution")

    if not events_df.empty and "event_id" in events_df.columns:
        event_id = st.selectbox("Target Event Analysis Instance", events_df["event_id"].unique())

        if st.button("Compute Multi-Method Matrix"):
            rows = db.get_by_event(event_id)
            if rows:
                # FIX: Find actual highest probability threat value cleanly
                worst = max(rows, key=lambda r: r.get("risk", r.get("max_pc", 0)))

                parser = CDMParser()
                rec = parser.parse_dataframe(pd.DataFrame([worst]))[0]

                pc = compute_all_pc(rec)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Alfano", f"{pc.alfano:.2e}")
                m2.metric("Foster", f"{pc.foster:.2e}")
                m3.metric("Chan", f"{pc.chan:.2e}")
                m4.metric("MC", f"{pc.monte_carlo:.2e}")
                m5.metric("Consensus", f"{pc.consensus:.2e}")

                # Pass clean parameters explicitly if dictionary mappings fail inside plot routines
                st.plotly_chart(plot_pc_comparison(vars(pc)), use_container_width=True)
            else:
                st.error("No raw database matches found for selected event.")
    else:
        st.info("Load data to perform cross-method verification calculations.")


# ----------------------------------------------------------------------
# GEOMETRY
# ----------------------------------------------------------------------
elif page == "🌐 Geometry":
    st.title("Close Approach Conjunction Geometry")

    if not events_df.empty and "event_id" in events_df.columns:
        event_id = st.selectbox("Geometric Assessment Target", events_df["event_id"].unique())
        rows = db.get_by_event(event_id)

        if rows:
            from Analytics_vis import plot_close_approach_geometry, plot_covariance_ellipse_2d

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_close_approach_geometry(rows), use_container_width=True)
            with c2:
                st.plotly_chart(plot_covariance_ellipse_2d(rows, "RT"), use_container_width=True)
        else:
            st.error("No coordinate points logged for this tracking number.")
    else:
        st.info("Upload active CDMs to inspect state space vectors.")


# ----------------------------------------------------------------------
# GMAT
# ----------------------------------------------------------------------
elif page == "🛰️ GMAT":
    st.title("GMAT Mission Control Interface Script Generator")

    if not events_df.empty and "event_id" in events_df.columns:
        event_id = st.selectbox("Maneuver Targeting Window", events_df["event_id"].unique())

        if st.button("Generate Mission Script"):
            rows = db.get_by_event(event_id)
            if rows:
                worst = max(rows, key=lambda r: r.get("risk", r.get("max_pc", 0)))

                parser = CDMParser()
                rec = parser.parse_dataframe(pd.DataFrame([worst]))[0]
                pc = compute_all_pc(rec)

                from GMAT_interface import generate_gmat_script

                path = generate_gmat_script(rec, pc, None)
                text = Path(path).read_text() if isinstance(path, (str, Path)) else getattr(path, 'read_text', lambda: "")()

                if text:
                    st.code(text, language="gmat")
                    st.download_button("Download Script File (.script)", text, file_name=f"Conjunction_{event_id}.script")
                else:
                    st.error("Script production tool returned an empty sequence descriptor block.")
    else:
        st.info("System configuration requires an loaded operational data catalog.")


# ----------------------------------------------------------------------
# DATA MANAGEMENT
# ----------------------------------------------------------------------
elif page == "📂 Data Management":
    st.title("System Log Data Management")

    c1, c2, c3 = st.columns(3)
    c1.metric("Records Found", stats.get("total_records", 0))
    c2.metric("Events Found", stats.get("total_events", 0))
    c3.metric("Active Alerts", stats.get("unacked_alerts", 0))

    try:
        conn = sqlite3.connect(str(db.db_path))
        logs = pd.read_sql("SELECT * FROM ingestion_log ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        st.subheader("Recent System Ingestion Logs")
        st.dataframe(logs, use_container_width=True)
    except Exception as e:
        st.caption(f"Log history mapping disabled or table uninitialized yet: {e}")

    st.divider()
    if st.button("🚨 Flush Data Storage and Reset Indexes"):
        db.clear_all_data()
        st.cache_data.clear()
        st.session_state.ingest_token += 1
        st.success("Internal relational storage data arrays dropped successfully.")
        st.rerun()