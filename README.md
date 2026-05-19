

---

## 🚀 Overview

**CORAL** (formerly CASAS — *Collision Alert & Situational Awareness System*) is a production-ready Python framework for analyzing Conjunction Data Messages (CDMs), computing multi-method collision probabilities, propagating orbital covariances with high-fidelity C++ differential algebra, and generating actionable maneuver recommendations.

Built for satellite operators, space situational awareness (SSA) engineers, and conjunction assessment analysts, CORAL bridges the gap between raw CDM ingestion and operational decision-making.

### Key Capabilities

| Module | Description |
|--------|-------------|
| **CDM Parser** | Ingests NASA/ESA-standard CDM CSV/TSV exports with automatic key normalization |
| **SQLite Database** | Persistent storage with full-text indexing, event grouping, and trend retrieval |
| **Multi-Method Pc** | Alfano 2D, Foster 2D, Chan series, and Monte Carlo (10⁷ samples) |
| **C++ DA Propagator** | Differential-algebra covariance propagation (RK4/RK78) with J₂ + drag |
| **Risk Trends** | Time-series analysis of Pc evolution as TCA approaches |
| **Alert Manager** | Structured alerting with NASA CARA/ESA risk thresholds |
| **GMAT Interface** | Auto-generated GMAT simulation scripts with maneuver burns |
| **Streamlit Dashboard** | Interactive dark-theme analytics console |
| **FastAPI Server** | Production REST API with OpenAPI/Swagger documentation |

---

## 📁 Project Structure

```
CORAL/
├── 📄 README.md                 # This file
├── 📄 LICENSE                   # MIT License
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore              # Git ignore rules
├── 📄 build.sh                # C++ extension build script
├── 📄 CMakeLists.txt          # CMake configuration (Windows/Linux/macOS)
├── 📄 run.sh                  # Streamlit dashboard launcher
│
├── 🔧 Core Modules
│   ├── CDM_parser.py              # CDM CSV/TSV parser & data structures
│   ├── CDM_database.py            # SQLite persistence layer
│   ├── Collision_Probability.py   # Pc engine (4 methods + DA propagator)
│   ├── Risk_Trend.py              # Trend analysis & alert manager
│   ├── casas_propagator.py        # Python ↔ C++ DA bridge
│   ├── GMAT_interface.py          # GMAT script generator
│   └── Analytics_vis.py           # Plotly visualization toolkit
│
├── 🌐 Interfaces
│   ├── api_server.py              # FastAPI REST server (v2.0.0)
│   └── Dashboard_app.py           # Streamlit analytics dashboard
│
├── ⚡ C++ Backend
│   ├── bindings.cpp               # pybind11 glue code
│   └── casas_da.hpp               # DA propagator header (merged)
│
└── 🧪 Testing
    ├── test.py                    # End-to-end pipeline test
    └── test_integration.py        # C++ v2 integration verification
```

---

## 🛠 Installation

### Prerequisites

- **Python** 3.11 or higher
- **C++ compiler** with C++17 support (GCC 9+, Clang 12+, MSVC 2019+)
- **CMake** 3.18+ (optional but recommended)
- **Eigen3** (auto-fetched by CMake if missing)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CORAL.git
cd CORAL
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Build C++ DA Extension (Optional but Recommended)

The C++ backend provides **10–100× faster** covariance propagation compared to the Python fallback.

**Option A — CMake (Recommended)**
```bash
bash build.sh
```

**Option B — Manual CMake**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
# The .so/.pyd is copied to the project root automatically
```

**Option C — Direct Compile (Fallback)**
```bash
c++ -std=c++17 -O3 -shared -fPIC $(python3 -m pybind11 --includes)     bindings.cpp -o casas_cpp$(python3-config --extension-suffix)
```

> **Verification:** Run `python -c "import casas_cpp; print(casas_cpp.__version__)"` to confirm the build.

---

## 🚦 Quick Start

### A. Parse a CDM File

```python
from CDM_parser import CDMParser

parser = CDMParser()
records = parser.parse_file("test_data.csv")  # Auto-detects comma or tab delimiter

rec = records[0]
print(f"Event {rec.event_id} | Miss: {rec.miss_distance_km:.2f} km | Risk: {rec.risk_level}")
```

### B. Compute Collision Probability (4 Methods)

```python
from Collision_Probability import compute_all_pc

pc = compute_all_pc(rec, fast=False)  # fast=True skips Monte Carlo
print(f"Consensus Pc = {pc.consensus:.2e} → {pc.risk_level}")
# Output: Consensus Pc = 1.23e-05 → HIGH
```

### C. Propagate Covariance with C++ DA

```python
from casas_propagator import propagate, propagate_covariance, PropConfig
import numpy as np

# High-fidelity propagation with J₂ + drag
cfg = PropConfig(use_j2=True, use_drag=True, bstar=2.5e-5, f107=150, ap=15)
result = propagate([6778.0, 0.0, 0.0, 0.0, 7.668, 0.0], dt_s=3600.0, cfg=cfg)

# Map covariance forward
P0 = np.diag([1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8])
Pf = propagate_covariance(P0.tolist(), result.stm)
```

### D. Generate GMAT Simulation Script

```python
from GMAT_interface import generate_gmat_script
from Collision_Probability import compute_all_pc, recommend_maneuver

pc = compute_all_pc(rec, fast=True)
mnv = recommend_maneuver(rec, pc)
script_path = generate_gmat_script(rec, pc, mnv)
print(f"GMAT script written to: {script_path}")
```

### E. Launch the Dashboard

```bash
bash run.sh
# or directly:
streamlit run Dashboard_app.py --theme.base dark
```

Open `http://localhost:8501` in your browser.

### F. Start the REST API

```bash
python api_server.py
```

Open `http://localhost:8000/docs` for interactive Swagger documentation.

---

## 📊 Dashboard Features

The Streamlit dashboard provides six analytical views:

| Page | Visualizations |
|------|----------------|
| **🏠 Overview** | KPI cards, risk distribution donut chart, orbital regime scatter, space weather correlation |
| **📊 CDM Analysis** | Filterable data table by risk level, mission, and TCA horizon |
| **📈 Risk Trends** | Dual-axis Pc/miss-distance evolution + Mahalanobis distance trend |
| **🎯 Collision Probability** | Method comparison bar chart (Alfano/Foster/Chan/MC) |
| **🌐 Geometry** | 3-D RTN relative position evolution + 1/2/3σ covariance ellipses |
| **🛰️ GMAT** | One-click script generation with avoidance maneuver burns |
| **📂 Data Management** | Ingestion log viewer, database clearing, CSV upload |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | System status & DB statistics |
| `GET`  | `/api/events` | All conjunction events (aggregated) |
| `GET`  | `/api/events/{id}` | Single event CDM series |
| `GET`  | `/api/events/{id}/trend` | Risk trend time series |
| `GET`  | `/api/events/{id}/pc` | Multi-method Pc computation |
| `GET`  | `/api/events/{id}/gmat` | Download GMAT script |
| `GET`  | `/api/records` | Raw CDM records (paginated) |
| `GET`  | `/api/critical` | Critical + high-risk events |
| `POST` | `/api/ingest` | Upload new CDM CSV/TSV file |
| `GET`  | `/api/alerts` | Alert log |
| `POST` | `/api/alerts/{id}/ack` | Acknowledge alert |
| `GET`  | `/api/export/csv` | Export records as CSV |
| `GET`  | `/api/export/json` | Export records as JSON |

---

## 🔬 Collision Probability Methods

CORAL implements four independent Pc estimators:

### 1. Alfano 2D (1992)
Industry-standard fast approximation using Bessel-function integration over the encounter plane. Includes a small-object analytical fallback for `R/σ < 0.05`.

### 2. Foster 2D (1992)
NASA CARA preferred method. Double-quadrature integration of the bivariate Gaussian over the hard-body circle. Falls back to Alfano if numerical integration fails.

### 3. Chan Series (1997)
Stable exponential approximation derived from series expansion. Computationally cheapest; suitable for large-scale screening.

### 4. Monte Carlo (Reference)
Ground-truth estimation using 10⁷ multivariate normal samples. Reproducible with optional `seed`. Standard error reported for statistical confidence.

### Consensus & Risk Classification
```
Consensus = median(Alfano, Foster, Chan, [Monte Carlo])
CRITICAL  : Pc ≥ 1×10⁻⁴
HIGH      : Pc ≥ 1×10⁻⁵
MEDIUM    : Pc ≥ 1×10⁻⁶
LOW       : Pc ≥ 1×10⁻⁷
NEGLIGIBLE: Pc < 1×10⁻⁷
```

---

## ⚡ C++ Differential Algebra Propagator

The `casas_cpp` module provides high-fidelity orbital mechanics:

| Feature | Description |
|---------|-------------|
| **Integrators** | RK4 fixed-step, RK7(8) Dormand-Prince adaptive |
| **Force Model** | Two-body + J₂ oblateness + atmospheric drag (MSISE90-style) + solar radiation pressure |
| **DA Order-1** | State Transition Matrix (STM) Φ(t,t₀) — exact, no finite differences |
| **DA Order-2** | State Transition Tensor (STT) ∂²x/∂x₀² for non-linear covariance mapping |
| **Mean Correction** | DACE-style second-order mean correction for non-Gaussian uncertainty |
| **EKF Update** | Joseph-stabilized 3-D measurement update with NIS computation |

### Python API (v2)

```python
from casas_propagator import (
    propagate,              # RK4 (v1 API)
    propagate_rk78,         # Adaptive RK7(8)
    propagate_order2,       # STM + STT
    propagate_order2_full,  # + mean correction
    propagate_covariance,   # P(t) = Φ·P₀·Φᵀ
    ekf_update,             # Kalman filter update
    PropConfig,
    backend                 # 'cpp' or 'python'
)
```

---

## 🧪 Testing

Run the full verification suite:

```bash
# End-to-end pipeline test (parser → DB → Pc → GMAT)
python test.py

# C++ v2 integration test (propagator, covariance, EKF)
python test_integration.py
```

Both tests automatically fall back to the pure-Python propagator if the C++ extension is not compiled.

---

## 🗄 Database Schema

SQLite with WAL mode and indexed columns:

- **`cdm_records`** — 60+ fields per CDM (geometry, covariance, orbital elements, space weather, computed Pc methods)
- **`alert_log`** — Structured alerts with acknowledgment tracking
- **`ingestion_log`** — File upload audit trail

Key indexes: `event_id`, `mission_id`, `risk`, `time_to_tca_days`, `risk_level`.

---

## 🛡 Maneuver Recommendation Engine

When `risk_level` is **HIGH** or **CRITICAL**, CORAL recommends:

- **Δv magnitude** — clipped to [0.5, 50.0] m/s based on required miss-distance increase
- **Direction** — `NORMAL` (if out-of-plane uncertainty is small), `TANGENTIAL` (if along-track dispersion dominates), or `RADIAL`
- **Execution window** — 60% of remaining TCA time, bounded to [6, 24] hours
- **Target miss distance** — 3× current miss distance or ≥ 1 km

---

## 🖥 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11 | 3.12 |
| RAM | 4 GB | 16 GB (for Monte Carlo) |
| Disk | 500 MB | 2 GB (for historical CDM DB) |
| OS | Linux/macOS/Windows 10 | Ubuntu 22.04 LTS |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📜 License

CORAL is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Alfano (1992)** — 2D collision probability formulation
- **Foster & Estes (1992)** — NASA CARA standard method
- **Chan (1997)** — Analytical series expansion
- **Berz (1999)** — Differential Algebra propagation concepts
- **GMAT Team** — General Mission Analysis Tool integration

---

<div align="center">
  <p><strong>🛰 Safeguarding orbital assets through rigorous conjunction analysis</strong></p>
  <p><em>Built with Python, C++, and a deep respect for orbital mechanics.</em></p>
</div>
