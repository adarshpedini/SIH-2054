# AI-Enabled Real-Time Digital Twin — Aero Piston Engine (MALE UAV)

End-to-end prototype: physics-based engine simulator → physics-informed
AI diagnostics (anomaly detection, fault classification, RUL) → FastAPI
sync layer → React/Tailwind Ground Control Station HMI.

## Architecture

```mermaid
flowchart LR
    subgraph Physical["Phase 1 — Engine Simulator"]
        A[Thermodynamic Engine Model<br/>engine_sim.py] --> B[Fault Injector Engine]
        B --> C[CAN-style Telemetry Frames<br/>0x100-0x1FF]
    end

    subgraph Ingestion["Telemetry Ingestion"]
        C --> D[Async Snapshot Builder]
    end

    subgraph AI["Phase 2 — Hybrid AI Diagnostic Layer"]
        D --> E[Physics-Informed<br/>Feature Extraction]
        E --> F[Isolation Forest<br/>Anomaly Detector]
        E --> G[Random Forest<br/>Fault Classifier]
        E --> H[RUL Estimator<br/>Weibull/Exp Decay Trend]
        F --> I[XAI Root-Cause<br/>Narrative Generator]
        G --> I
        H --> I
    end

    subgraph Backend["Phase 3 — FastAPI Sync Layer"]
        D --> J[main.py<br/>Simulation Loop]
        I --> J
        J --> K[/WebSocket Broker<br/>ws/telemetry/]
        J --> L[REST API<br/>fault / mission / replay]
        J --> M[(In-Memory<br/>Mission Log)]
    end

    subgraph GCS["Phase 4 — Defense GCS HMI"]
        K --> N[Live Gauge Matrix]
        K --> O[Health Indices Panel]
        K --> P[Predictive Analytics / RUL]
        L --> Q[Mission & Fault Controls]
        M --> R[Mission Replay Viewer]
    end

    Q -. commands .-> J
    R -. replay request .-> J
```

## Run locally (no Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:5173, expects backend on localhost:8000
```

## Run with Docker (one command)

```bash
docker-compose up --build
```

- GCS Dashboard: http://localhost:8080
- Backend API/docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/telemetry

> Note: `App.jsx` defaults to `localhost:8000` for API/WS calls for local dev
> convenience. When deployed behind the bundled nginx reverse proxy (port
> 8080), switch `API_BASE`/`WS_URL` in `App.jsx` to relative paths
> (`""` and `ws://<host>/ws/telemetry`) before building, so requests route
> through the `/api/` and `/ws/` proxy locations defined in `nginx.conf`.

## Key REST endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/state` | GET | One-shot current physical + digital twin state |
| `/api/fault/inject` | POST | Inject a fault (`fault_type`, `severity`, `cylinder`) |
| `/api/fault/clear` | POST | Clear all active faults |
| `/api/mission/profile` | POST | Set mission profile / altitude / throttle / ambient temp |
| `/api/mission/presets` | GET | List preset mission profiles |
| `/api/replay/start` | POST | Begin mission log replay |
| `/api/replay/stop` | POST | Stop replay, return to live telemetry |
| `/api/replay/log` | GET | Paginated raw mission log |

## Fault modes modeled

Injector Clogging · Misfire · Cooling System Degradation · Sensor Drift ·
Oil Leak · Combustion Instability — each perturbs the underlying physical
model (not just displayed values), so the AI layer is diagnosing genuine
coupled-system signatures.
