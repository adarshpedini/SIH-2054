"""
PHASE 3 — FastAPI Backend & Telemetry Synchronization Layer
=============================================================
Wires the engine simulator (Phase 1) and diagnostic engine (Phase 2)
together, streams synchronized physical + AI-predicted state over
WebSocket, exposes REST endpoints for mission/fault control, and
provides replay of logged mission telemetry.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine_sim import EngineSimulator, FaultType
from ai_engine import DiagnosticEngine, DiagnosticResult

app = FastAPI(title="UAV Aero-Engine Digital Twin API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICK_HZ = 10.0
LOG_MAX_FRAMES = 20000  # ~33 min at 10Hz — bounded in-memory mission log

sim = EngineSimulator(dt=1.0 / TICK_HZ, seed=7)
diagnostics = DiagnosticEngine()

mission_log: Deque[Dict] = deque(maxlen=LOG_MAX_FRAMES)
connected_clients: List[WebSocket] = []

replay_state = {"active": False, "index": 0, "speed": 1.0}


# --------------------------------------------------------------------------
# Background simulation loop
# --------------------------------------------------------------------------

async def simulation_loop():
    interval = 1.0 / TICK_HZ
    while True:
        if not replay_state["active"]:
            sim.tick()
            snapshot = sim.snapshot_dict()
            diag: DiagnosticResult = diagnostics.process(snapshot)
            frame = _build_sync_frame(snapshot, diag)
            mission_log.append(frame)
            await _broadcast(frame)
        else:
            await _replay_step()
        await asyncio.sleep(interval / max(0.1, replay_state["speed"]) if replay_state["active"] else interval)


def _build_sync_frame(snapshot: Dict, diag: DiagnosticResult) -> Dict:
    return {
        "type": "sync_frame",
        "timestamp": snapshot["timestamp"],
        "physical_state": snapshot,
        "digital_twin_prediction": {
            "anomaly_score": diag.anomaly_score,
            "is_anomalous": diag.is_anomalous,
            "predicted_fault": diag.predicted_fault,
            "fault_confidence": diag.fault_confidence,
            "fault_probabilities": diag.fault_probabilities,
            "rul_estimates": diag.rul_estimates,
            "xai_explanation": diag.xai_explanation,
            "system_health_pct": diag.system_health_pct,
            "subsystem_health": diag.subsystem_health,
        },
    }


async def _broadcast(frame: Dict):
    if not connected_clients:
        return
    payload = json.dumps(frame)
    stale = []
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        connected_clients.remove(ws)


async def _replay_step():
    idx = replay_state["index"]
    if idx >= len(mission_log):
        replay_state["active"] = False
        replay_state["index"] = 0
        return
    frame = dict(mission_log[idx])
    frame["type"] = "replay_frame"
    frame["replay_index"] = idx
    frame["replay_total"] = len(mission_log)
    await _broadcast(frame)
    replay_state["index"] += 1


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())


# --------------------------------------------------------------------------
# WebSocket endpoint — synchronized physical + digital twin state
# --------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # keep the connection alive / accept optional inline commands
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
                await _handle_ws_command(cmd)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


async def _handle_ws_command(cmd: Dict):
    action = cmd.get("action")
    if action == "inject_fault":
        sim.inject_fault(
            FaultType(cmd["fault_type"]),
            severity=cmd.get("severity", 0.6),
            cylinder=cmd.get("cylinder"),
        )
    elif action == "clear_faults":
        sim.clear_faults()
    elif action == "set_mission":
        sim.set_mission(**{k: v for k, v in cmd.items() if k != "action"})


# --------------------------------------------------------------------------
# REST models
# --------------------------------------------------------------------------

class FaultInjectRequest(BaseModel):
    fault_type: str = Field(..., description="One of: injector_clog, misfire, "
                                               "cooling_degradation, sensor_drift, "
                                               "oil_leak, combustion_instability")
    severity: float = Field(0.6, ge=0.0, le=1.0)
    cylinder: Optional[int] = Field(None, ge=1, le=4)
    ramping: bool = True


class MissionProfileRequest(BaseModel):
    profile_name: str = "cruise"
    target_altitude_ft: Optional[float] = None
    target_throttle_pct: Optional[float] = None
    ambient_override_c: Optional[float] = None


PRESET_PROFILES = {
    "high_altitude": {"target_altitude_ft": 25000, "target_throttle_pct": 70, "ambient_override_c": None},
    "hot_weather": {"target_altitude_ft": 3000, "target_throttle_pct": 60, "ambient_override_c": 45},
    "rapid_throttle": {"target_altitude_ft": 8000, "target_throttle_pct": 95, "ambient_override_c": None},
    "cruise": {"target_altitude_ft": 12000, "target_throttle_pct": 65, "ambient_override_c": None},
}


# --------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "sim_time_s": round(sim.state.time_s, 2),
            "engine_hours": round(sim.state.engine_hours, 3),
            "connected_clients": len(connected_clients)}


@app.get("/api/state")
async def get_state():
    snapshot = sim.snapshot_dict()
    diag = diagnostics.process(snapshot)
    return _build_sync_frame(snapshot, diag)


@app.post("/api/fault/inject")
async def inject_fault(req: FaultInjectRequest):
    try:
        ftype = FaultType(req.fault_type)
    except ValueError:
        return {"error": f"Unknown fault_type '{req.fault_type}'. "
                          f"Valid: {[f.value for f in FaultType if f != FaultType.NONE]}"}
    fault = sim.inject_fault(ftype, severity=req.severity, cylinder=req.cylinder, ramping=req.ramping)
    return {"status": "injected", "fault": {
        "type": fault.fault_type.value, "severity": fault.severity, "cylinder": fault.cylinder
    }}


@app.post("/api/fault/clear")
async def clear_faults():
    sim.clear_faults()
    return {"status": "cleared"}


@app.get("/api/fault/active")
async def active_faults():
    return {"active_faults": [
        {"type": f.fault_type.value, "severity": f.severity, "cylinder": f.cylinder}
        for f in sim.active_faults
    ]}


@app.post("/api/mission/profile")
async def set_mission_profile(req: MissionProfileRequest):
    preset = PRESET_PROFILES.get(req.profile_name, {})
    kwargs = {"profile_name": req.profile_name}
    for key in ("target_altitude_ft", "target_throttle_pct", "ambient_override_c"):
        val = getattr(req, key)
        if val is not None:
            kwargs[key] = val
        elif key in preset:
            kwargs[key] = preset[key]
    sim.set_mission(**kwargs)
    return {"status": "mission_updated", "mission": kwargs}


@app.get("/api/mission/presets")
async def mission_presets():
    return PRESET_PROFILES


@app.get("/api/replay/status")
async def replay_status():
    return {**replay_state, "log_length": len(mission_log)}


@app.post("/api/replay/start")
async def replay_start(from_index: int = 0, speed: float = 1.0):
    replay_state["active"] = True
    replay_state["index"] = max(0, min(from_index, len(mission_log) - 1)) if mission_log else 0
    replay_state["speed"] = max(0.1, speed)
    return {"status": "replay_started", **replay_state}


@app.post("/api/replay/stop")
async def replay_stop():
    replay_state["active"] = False
    return {"status": "replay_stopped"}


@app.get("/api/replay/log")
async def get_replay_log(start: int = 0, limit: int = 500):
    log_slice = list(mission_log)[start:start + limit]
    return {"start": start, "count": len(log_slice), "total": len(mission_log), "frames": log_slice}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
