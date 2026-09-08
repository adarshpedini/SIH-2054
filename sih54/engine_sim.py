"""
PHASE 1 — Thermodynamic Engine Simulator & CAN Telemetry Engine
================================================================
Simulates an aero piston engine (Rotax 914 / Austro AE300 class, ~100kW turbo
4-cylinder) for a MALE UAV. Produces physically-coupled telemetry at a fixed
tick rate and streams it as JSON frames (structured to mirror a SocketCAN
frame set: one topic per "CAN ID" group). Includes a fault-injection engine
that perturbs the physical model, not just the readouts, so downstream AI
has a genuine signal to detect.

Run standalone:
    python engine_sim.py --ws-port 8765

Or import `EngineSimulator` / `TelemetryFrame` into main.py (Phase 3).
"""

from __future__ import annotations

import asyncio
import argparse
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

try:
    import websockets
except ImportError:  # allow import-only use (e.g. from FastAPI) without the CLI extra
    websockets = None


# --------------------------------------------------------------------------
# Constants — physical / thermodynamic model parameters
# --------------------------------------------------------------------------

SEA_LEVEL_TEMP_C = 15.0
SEA_LEVEL_PRESSURE_KPA = 101.325
LAPSE_RATE_C_PER_FT = -1.98 / 1000.0          # ISA lapse rate
GAS_CONST_AIR = 287.05                         # J/(kg K)
CP_AIR = 1005.0                                # J/(kg K)
FUEL_LHV_MJ_PER_KG = 43.0                      # gasoline/avgas lower heating value
STOICH_AFR = 14.7

CYLINDERS = 4
IDLE_RPM = 1700
MAX_RPM = 5800
REDLINE_RPM = 6200

# Cooling model constants
COOLANT_TARGET_C = 92.0
OIL_TARGET_C = 95.0


class FaultType(str, Enum):
    NONE = "none"
    INJECTOR_CLOG = "injector_clog"
    MISFIRE = "misfire"
    COOLING_DEGRADATION = "cooling_degradation"
    SENSOR_DRIFT = "sensor_drift"
    OIL_LEAK = "oil_leak"
    COMBUSTION_INSTABILITY = "combustion_instability"


@dataclass
class FaultState:
    """A currently-active fault and its severity/target."""
    fault_type: FaultType = FaultType.NONE
    severity: float = 0.0            # 0.0 - 1.0
    cylinder: Optional[int] = None   # 1-indexed, for cylinder-local faults
    onset_time: float = 0.0
    ramping: bool = True             # if True, severity ramps up over time (progressive fault)


@dataclass
class MissionProfile:
    """External operator/mission commands driving the simulation."""
    altitude_ft: float = 5000.0
    target_altitude_ft: float = 5000.0
    throttle_pct: float = 65.0
    target_throttle_pct: float = 65.0
    ambient_override_c: Optional[float] = None
    profile_name: str = "cruise"


@dataclass
class EngineInternalState:
    """Slow-evolving internal states (integrated each tick) — the 'true' physical state."""
    rpm: float = IDLE_RPM
    cht_c: List[float] = field(default_factory=lambda: [70.0] * CYLINDERS)
    egt_c: List[float] = field(default_factory=lambda: [300.0] * CYLINDERS)
    oil_temp_c: float = 60.0
    oil_pressure_kpa: float = 350.0
    oil_level_pct: float = 100.0
    coolant_temp_c: float = 60.0
    coolant_flow_lpm: float = 40.0
    fuel_flow_lph: float = 12.0
    injector_duty: List[float] = field(default_factory=lambda: [45.0] * CYLINDERS)
    battery_voltage: float = 24.8
    alternator_voltage: float = 28.2
    vibration_base_g: float = 0.15
    manifold_pressure_kpa: float = 95.0
    time_s: float = 0.0
    engine_hours: float = 0.0


@dataclass
class TelemetryFrame:
    """One outbound telemetry frame — mirrors a grouped CAN broadcast."""
    can_id: str
    timestamp: float
    seq: int
    data: Dict


class EngineSimulator:
    """
    Integrates a coupled thermodynamic/mechanical engine model at `dt` step,
    with an active fault-injection layer. Call `.tick()` each step and read
    `.build_telemetry_frames()` for the CAN-like frame set.
    """

    def __init__(self, dt: float = 0.1, seed: Optional[int] = None):
        self.dt = dt
        self.rng = random.Random(seed)
        self.state = EngineInternalState()
        self.mission = MissionProfile()
        self.active_faults: List[FaultState] = []
        self._seq = 0
        self._sensor_drift_offsets: Dict[str, float] = {}
        self._misfire_phase = 0.0

    # ---------------------------------------------------------------- API

    def set_mission(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.mission, k):
                setattr(self.mission, k, v)

    def inject_fault(self, fault_type: FaultType, severity: float = 0.6,
                      cylinder: Optional[int] = None, ramping: bool = True):
        fault = FaultState(
            fault_type=fault_type,
            severity=max(0.0, min(1.0, severity)),
            cylinder=cylinder,
            onset_time=self.state.time_s,
            ramping=ramping,
        )
        self.active_faults.append(fault)
        return fault

    def clear_faults(self):
        self.active_faults = []
        self._sensor_drift_offsets = {}

    # ------------------------------------------------------------ physics

    def _ambient_conditions(self):
        alt = self.state.time_s and self.mission.altitude_ft or self.mission.altitude_ft
        temp_c = (self.mission.ambient_override_c
                  if self.mission.ambient_override_c is not None
                  else SEA_LEVEL_TEMP_C + LAPSE_RATE_C_PER_FT * alt)
        pressure_kpa = SEA_LEVEL_PRESSURE_KPA * (1 - 2.25577e-5 * alt) ** 5.25588
        density = (pressure_kpa * 1000) / (GAS_CONST_AIR * (temp_c + 273.15))
        return temp_c, pressure_kpa, density

    def _fault_multiplier(self, fault_type: FaultType, cylinder: Optional[int] = None) -> float:
        """Aggregate active severity (0-1, ramped) for a given fault type / cylinder."""
        total = 0.0
        for f in self.active_faults:
            if f.fault_type != fault_type:
                continue
            if cylinder is not None and f.cylinder is not None and f.cylinder != cylinder:
                continue
            sev = f.severity
            if f.ramping:
                elapsed = self.state.time_s - f.onset_time
                ramp = min(1.0, elapsed / 180.0)  # ramps to full severity over 3 min
                sev *= ramp
            total = max(total, sev)
        return total

    def _step_throttle_and_altitude(self):
        # first-order lag toward target (actuator + pilot/mission dynamics)
        tau = 3.0
        self.mission.throttle_pct += (self.mission.target_throttle_pct - self.mission.throttle_pct) * (self.dt / tau)
        self.mission.altitude_ft += (self.mission.target_altitude_ft - self.mission.altitude_ft) * (self.dt / 20.0)
        self.mission.throttle_pct = max(0.0, min(100.0, self.mission.throttle_pct))

    def _step_rpm(self, ambient_density: float):
        s = self.state
        throttle = self.mission.throttle_pct / 100.0

        # combustion instability fault reduces effective torque and adds noise
        instability = self._fault_multiplier(FaultType.COMBUSTION_INSTABILITY)
        misfire = self._fault_multiplier(FaultType.MISFIRE)

        target_rpm = IDLE_RPM + throttle * (MAX_RPM - IDLE_RPM)
        target_rpm *= (1.0 - 0.15 * misfire)          # misfire robs power -> lower achievable rpm
        target_rpm *= (1.0 - 0.10 * instability)

        # density altitude derates naturally-aspirated portion; assume light turbo compensation
        density_ratio = ambient_density / 1.225
        boost_factor = 0.55 + 0.45 * min(1.3, density_ratio + 0.25)  # simplistic wastegate model
        target_rpm *= min(1.0, boost_factor)

        tau_rpm = 1.5
        noise = self.rng.gauss(0, 6.0) * (1 + 3 * instability)
        s.rpm += (target_rpm - s.rpm) * (self.dt / tau_rpm) + noise * self.dt
        s.rpm = max(0.0, s.rpm)

        self.mission.throttle_used = throttle
        return throttle

    def _step_thermo(self, ambient_temp_c: float, throttle: float):
        """Couples fuel flow -> combustion heat -> EGT/CHT -> cooling loop, per cylinder."""
        s = self.state
        rpm_frac = s.rpm / MAX_RPM

        clog = self._fault_multiplier(FaultType.INJECTOR_CLOG)
        clog_cyl = next((f.cylinder for f in self.active_faults if f.fault_type == FaultType.INJECTOR_CLOG), None)
        misfire = self._fault_multiplier(FaultType.MISFIRE)
        misfire_cyl = next((f.cylinder for f in self.active_faults if f.fault_type == FaultType.MISFIRE), None)
        cooling_deg = self._fault_multiplier(FaultType.COOLING_DEGRADATION)
        oil_leak = self._fault_multiplier(FaultType.OIL_LEAK)

        # --- fuel flow (base map + faults) ---
        base_fuel_lph = 3.0 + 22.0 * rpm_frac * (0.4 + 0.6 * throttle)
        s.fuel_flow_lph = max(0.5, base_fuel_lph * (1 - 0.35 * clog))

        total_heat_kw = base_fuel_lph * (1000 / 3600) * 0.74 * FUEL_LHV_MJ_PER_KG * 0.30  # ~30% to exhaust/cyl heat

        for i in range(CYLINDERS):
            cyl_no = i + 1
            duty = 30 + 60 * rpm_frac * (0.5 + 0.5 * throttle)

            local_clog = clog if (clog_cyl is None or clog_cyl == cyl_no) else 0.0
            local_misfire = misfire if (misfire_cyl is None or misfire_cyl == cyl_no) else 0.0

            duty *= (1 - 0.4 * local_clog)
            s.injector_duty[i] += (duty - s.injector_duty[i]) * (self.dt / 0.5)

            cyl_heat_share = total_heat_kw / CYLINDERS
            # misfire: incomplete combustion -> lower EGT for that event pattern but heat spikes
            # (unburnt fuel igniting in exhaust) -> we model as EGT oscillation + slight avg cut
            misfire_egt_mod = 1.0
            if local_misfire > 0:
                self._misfire_phase += self.dt * (2 + 8 * local_misfire)
                misfire_egt_mod = 1.0 - 0.25 * local_misfire + 0.20 * local_misfire * math.sin(self._misfire_phase * 6)

            target_egt = ambient_temp_c + 260 + 480 * (cyl_heat_share / 6.0) * (1 - 0.5 * local_clog)
            target_egt *= misfire_egt_mod
            s.egt_c[i] += (target_egt - s.egt_c[i]) * (self.dt / 4.0) + self.rng.gauss(0, 1.2)

            cooling_capacity = 1.0 - 0.55 * cooling_deg
            target_cht = ambient_temp_c + (55 + 70 * (cyl_heat_share / 6.0)) / max(0.35, cooling_capacity)
            s.cht_c[i] += (target_cht - s.cht_c[i]) * (self.dt / 12.0) + self.rng.gauss(0, 0.4)

        # --- coolant loop ---
        avg_cht = sum(s.cht_c) / CYLINDERS
        s.coolant_flow_lpm = max(5.0, 42.0 * (1 - 0.6 * cooling_deg))
        target_coolant = COOLANT_TARGET_C + (avg_cht - (ambient_temp_c + 90)) * 0.15
        s.coolant_temp_c += (target_coolant - s.coolant_temp_c) * (self.dt / 15.0)

        # --- oil loop ---
        s.oil_level_pct = max(0.0, s.oil_level_pct - oil_leak * 0.004 * self.dt * 60)
        oil_capacity_frac = max(0.15, s.oil_level_pct / 100.0)
        target_oil_temp = OIL_TARGET_C + (avg_cht - 150) * 0.08
        s.oil_temp_c += (target_oil_temp - s.oil_temp_c) * (self.dt / 20.0)

        base_oil_pressure = 280 + 2.0 * rpm_frac * 260
        s.oil_pressure_kpa = base_oil_pressure * oil_capacity_frac * (1 - 0.3 * oil_leak)
        s.oil_pressure_kpa += self.rng.gauss(0, 3.0)

        # --- electrical ---
        alt_load = 0.85 + 0.15 * rpm_frac
        s.alternator_voltage = 27.5 + 1.2 * min(1.0, rpm_frac * 1.4) - 0.3 * (1 - alt_load)
        s.battery_voltage += (s.alternator_voltage - 0.4 - s.battery_voltage) * (self.dt / 3.0)

        # --- vibration (broadband + fault-specific harmonics summarized as an FFT-like band vector) ---
        instability = self._fault_multiplier(FaultType.COMBUSTION_INSTABILITY)
        s.vibration_base_g = 0.12 + 0.05 * rpm_frac + 0.35 * misfire + 0.5 * instability + self.rng.gauss(0, 0.01)

        # manifold pressure roughly tracks throttle + altitude density
        s.manifold_pressure_kpa = 30 + 65 * throttle

    def _apply_sensor_drift(self, sensor_id: str, value: float) -> float:
        drift_sev = self._fault_multiplier(FaultType.SENSOR_DRIFT)
        if drift_sev <= 0:
            return value
        if sensor_id not in self._sensor_drift_offsets:
            self._sensor_drift_offsets[sensor_id] = self.rng.uniform(-1, 1)
        drift_dir = self._sensor_drift_offsets[sensor_id]
        magnitude = drift_sev * abs(value) * 0.18
        return value + drift_dir * magnitude

    def _vibration_spectrum(self) -> List[float]:
        """Synthetic 16-bin FFT magnitude spectrum (0-800Hz) built from RPM harmonics + fault energy."""
        s = self.state
        firing_freq = (s.rpm / 60.0) * (CYLINDERS / 2.0)  # 4-stroke, per-cylinder firing freq
        bins = [0.0] * 16
        bin_width_hz = 800.0 / 16
        base = s.vibration_base_g
        for h in range(1, 5):
            f = firing_freq * h
            idx = int(f / bin_width_hz)
            if 0 <= idx < 16:
                bins[idx] += base * (1.0 / h)
        misfire = self._fault_multiplier(FaultType.MISFIRE)
        instability = self._fault_multiplier(FaultType.COMBUSTION_INSTABILITY)
        if misfire > 0:
            idx = int((firing_freq / 2) / bin_width_hz)  # half-order energy = classic misfire signature
            if 0 <= idx < 16:
                bins[idx] += 0.4 * misfire
        if instability > 0:
            for i in range(16):
                bins[i] += self.rng.uniform(0, 0.15) * instability
        return [round(max(0.0, b + self.rng.gauss(0, 0.01)), 4) for b in bins]

    # ------------------------------------------------------------- ticking

    def tick(self) -> EngineInternalState:
        self._step_throttle_and_altitude()
        ambient_temp_c, ambient_pressure_kpa, ambient_density = self._ambient_conditions()
        throttle = self._step_rpm(ambient_density)
        self._step_thermo(ambient_temp_c, throttle)

        self.state.time_s += self.dt
        self.state.engine_hours += self.dt / 3600.0
        self._seq += 1
        self._last_ambient = (ambient_temp_c, ambient_pressure_kpa, ambient_density)
        return self.state

    # -------------------------------------------------------- frame build

    def build_telemetry_frames(self) -> List[TelemetryFrame]:
        s = self.state
        t = time.time()
        ambient_temp_c, ambient_pressure_kpa, _ = getattr(self, "_last_ambient", (15.0, 101.3, 1.225))

        frames = [
            TelemetryFrame("0x100_RPM_THROTTLE", t, self._seq, {
                "rpm": round(self._apply_sensor_drift("rpm", s.rpm), 1),
                "throttle_pct": round(self.mission.throttle_pct, 1),
                "manifold_pressure_kpa": round(s.manifold_pressure_kpa, 1),
            }),
            TelemetryFrame("0x101_CHT", t, self._seq, {
                f"cht_{i+1}_c": round(self._apply_sensor_drift(f"cht_{i+1}", v), 1)
                for i, v in enumerate(s.cht_c)
            }),
            TelemetryFrame("0x102_EGT", t, self._seq, {
                f"egt_{i+1}_c": round(self._apply_sensor_drift(f"egt_{i+1}", v), 1)
                for i, v in enumerate(s.egt_c)
            }),
            TelemetryFrame("0x103_FUEL", t, self._seq, {
                "fuel_flow_lph": round(s.fuel_flow_lph, 2),
                "injector_duty_pct": [round(v, 1) for v in s.injector_duty],
            }),
            TelemetryFrame("0x104_OIL", t, self._seq, {
                "oil_pressure_kpa": round(self._apply_sensor_drift("oil_press", s.oil_pressure_kpa), 1),
                "oil_temp_c": round(s.oil_temp_c, 1),
                "oil_level_pct": round(s.oil_level_pct, 1),
            }),
            TelemetryFrame("0x105_COOLING", t, self._seq, {
                "coolant_temp_c": round(s.coolant_temp_c, 1),
                "coolant_flow_lpm": round(s.coolant_flow_lpm, 1),
            }),
            TelemetryFrame("0x106_ELECTRICAL", t, self._seq, {
                "battery_voltage": round(s.battery_voltage, 2),
                "alternator_voltage": round(s.alternator_voltage, 2),
            }),
            TelemetryFrame("0x107_VIBRATION", t, self._seq, {
                "vibration_rms_g": round(s.vibration_base_g, 4),
                "fft_spectrum_0_800hz": self._vibration_spectrum(),
            }),
            TelemetryFrame("0x108_ENVIRONMENT", t, self._seq, {
                "altitude_ft": round(self.mission.altitude_ft, 0),
                "ambient_temp_c": round(ambient_temp_c, 1),
                "ambient_pressure_kpa": round(ambient_pressure_kpa, 2),
            }),
            TelemetryFrame("0x1FF_STATUS", t, self._seq, {
                "engine_hours": round(s.engine_hours, 3),
                "active_faults": [
                    {"type": f.fault_type.value, "severity": round(f.severity, 2), "cylinder": f.cylinder}
                    for f in self.active_faults
                ],
                "mission_profile": self.mission.profile_name,
            }),
        ]
        return frames

    def snapshot_dict(self) -> Dict:
        """Full flat snapshot — convenient for AI engine ingestion (Phase 2)."""
        frames = self.build_telemetry_frames()
        merged = {"timestamp": time.time(), "seq": self._seq}
        for f in frames:
            merged.update(f.data)
        return merged


# --------------------------------------------------------------------------
# Standalone WebSocket broadcast server (mirrors a CAN gateway)
# --------------------------------------------------------------------------

class TelemetryBroadcastServer:
    def __init__(self, sim: EngineSimulator, tick_hz: float = 10.0):
        self.sim = sim
        self.tick_hz = tick_hz
        self.clients: Set = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self._handle_command(message)
        finally:
            self.clients.discard(websocket)

    async def _handle_command(self, message: str):
        try:
            cmd = json.loads(message)
        except json.JSONDecodeError:
            return
        action = cmd.get("action")
        if action == "inject_fault":
            self.sim.inject_fault(
                FaultType(cmd["fault_type"]),
                severity=cmd.get("severity", 0.6),
                cylinder=cmd.get("cylinder"),
                ramping=cmd.get("ramping", True),
            )
        elif action == "clear_faults":
            self.sim.clear_faults()
        elif action == "set_mission":
            self.sim.set_mission(**{k: v for k, v in cmd.items() if k != "action"})

    async def run(self):
        interval = 1.0 / self.tick_hz
        while True:
            self.sim.tick()
            if self.clients:
                snapshot = self.sim.snapshot_dict()
                payload = json.dumps({"type": "telemetry", "data": snapshot})
                await asyncio.gather(*(c.send(payload) for c in list(self.clients)), return_exceptions=True)
            await asyncio.sleep(interval)


async def main_async(ws_port: int, tick_hz: float, seed: Optional[int]):
    if websockets is None:
        raise RuntimeError("Install the 'websockets' package to run the standalone broadcast server.")
    sim = EngineSimulator(dt=1.0 / tick_hz, seed=seed)
    server = TelemetryBroadcastServer(sim, tick_hz=tick_hz)
    async with websockets.serve(server.handler, "0.0.0.0", ws_port):
        print(f"[engine_sim] Telemetry broadcast on ws://0.0.0.0:{ws_port} @ {tick_hz}Hz")
        await server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aero piston engine digital-twin simulator")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--tick-hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args.ws_port, args.tick_hz, args.seed))
