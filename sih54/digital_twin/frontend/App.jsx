import React, { useState, useEffect, useRef, useCallback } from "react";

/**
 * PHASE 4 — Ground Control Station (GCS) Dashboard HMI
 * ======================================================
 * Single-file React + Tailwind dashboard for the aero-engine digital twin.
 * Connects to /ws/telemetry on the FastAPI backend (Phase 3) and renders:
 *  - Live gauge matrix (RPM, CHT, EGT, Oil Pressure, Vibration, Fuel Flow)
 *  - Health indices panel (system + subsystem health)
 *  - Predictive analytics / RUL countdown + degradation trend
 *  - Mission & simulation controls (altitude, ambient temp, fault injection, replay)
 *
 * Expects the backend at ws://<host>:8000/ws/telemetry and REST at
 * http://<host>:8000/api/*. Adjust API_BASE / WS_URL below for your deployment.
 */

const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/telemetry";

const FAULT_TYPES = [
  { value: "injector_clog", label: "Injector Clogging" },
  { value: "misfire", label: "Misfire" },
  { value: "cooling_degradation", label: "Cooling Degradation" },
  { value: "sensor_drift", label: "Sensor Drift" },
  { value: "oil_leak", label: "Oil Leak" },
  { value: "combustion_instability", label: "Combustion Instability" },
];

const MISSION_PRESETS = [
  { value: "cruise", label: "Cruise" },
  { value: "high_altitude", label: "High Altitude" },
  { value: "hot_weather", label: "Hot Weather" },
  { value: "rapid_throttle", label: "Rapid Throttle" },
];

// ---------------------------------------------------------------- helpers

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function healthColor(pct) {
  if (pct >= 80) return "text-emerald-400";
  if (pct >= 50) return "text-amber-400";
  return "text-red-500";
}

function healthBarColor(pct) {
  if (pct >= 80) return "bg-emerald-500";
  if (pct >= 50) return "bg-amber-500";
  return "bg-red-500";
}

// ---------------------------------------------------------------- Gauge

function Gauge({ label, value, unit, min, max, warnAt, critAt, decimals = 0 }) {
  const pct = clamp(((value - min) / (max - min)) * 100, 0, 100);
  const isCrit = critAt !== undefined && (critAt > min ? value >= critAt : value <= critAt);
  const isWarn = !isCrit && warnAt !== undefined && (warnAt > min ? value >= warnAt : value <= warnAt);
  const ringColor = isCrit ? "#ef4444" : isWarn ? "#f59e0b" : "#22d3ee";

  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center bg-slate-900/70 border border-slate-700 rounded-xl p-3">
      <div className="relative w-28 h-28">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle cx="50" cy="50" r={radius} fill="none" stroke="#1e293b" strokeWidth="8" />
          <circle
            cx="50" cy="50" r={radius} fill="none"
            stroke={ringColor} strokeWidth="8" strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.3s ease, stroke 0.3s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold text-slate-100 tabular-nums">
            {Number.isFinite(value) ? value.toFixed(decimals) : "--"}
          </span>
          <span className="text-[10px] text-slate-400">{unit}</span>
        </div>
      </div>
      <span className="mt-2 text-xs font-medium text-slate-300 tracking-wide uppercase">{label}</span>
    </div>
  );
}

// ---------------------------------------------------------------- HealthBar

function HealthBar({ label, pct }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-300">{label}</span>
        <span className={`font-semibold ${healthColor(pct)}`}>{pct?.toFixed(0) ?? "--"}%</span>
      </div>
      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${healthBarColor(pct)} transition-all duration-300`}
          style={{ width: `${clamp(pct ?? 0, 0, 100)}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Sparkline (degradation trend)

function Sparkline({ data, warnLine }) {
  if (!data || data.length < 2) {
    return <div className="h-16 flex items-center justify-center text-slate-600 text-xs">Collecting trend data…</div>;
  }
  const w = 260, h = 64;
  const min = Math.min(...data, warnLine ?? Infinity);
  const max = Math.max(...data, warnLine ?? -Infinity);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(" ");
  const warnY = warnLine !== undefined ? h - ((warnLine - min) / range) * h : null;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16">
      {warnY !== null && (
        <line x1="0" y1={warnY} x2={w} y2={warnY} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth="1" />
      )}
      <polyline points={points} fill="none" stroke="#22d3ee" strokeWidth="2" />
    </svg>
  );
}

// ---------------------------------------------------------------- Main App

export default function App() {
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState(null);
  const [chtHistory, setChtHistory] = useState([]);
  const [replaying, setReplaying] = useState(false);
  const [altitude, setAltitude] = useState(12000);
  const [ambientTemp, setAmbientTemp] = useState(15);
  const [selectedFault, setSelectedFault] = useState(FAULT_TYPES[0].value);
  const [severity, setSeverity] = useState(0.6);
  const [selectedCylinder, setSelectedCylinder] = useState("");
  const wsRef = useRef(null);

  // ---- WebSocket lifecycle ----
  useEffect(() => {
    let retryTimer;
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retryTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "sync_frame" || msg.type === "replay_frame") {
            setFrame(msg);
            const cht = msg.physical_state?.cht_1_c;
            if (typeof cht === "number") {
              setChtHistory((prev) => [...prev.slice(-59), cht]);
            }
            setReplaying(msg.type === "replay_frame");
          }
        } catch (e) { /* ignore malformed frame */ }
      };
    }
    connect();
    return () => {
      clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, []);

  const sendCommand = useCallback((cmd) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  const postApi = useCallback(async (path, body) => {
    try {
      await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (e) { /* backend offline in demo mode */ }
  }, []);

  const physical = frame?.physical_state ?? {};
  const twin = frame?.digital_twin_prediction ?? {};

  const chtVals = [1, 2, 3, 4].map((i) => physical[`cht_${i}_c`] ?? 0);
  const egtVals = [1, 2, 3, 4].map((i) => physical[`egt_${i}_c`] ?? 0);
  const maxCht = Math.max(...chtVals, 0);
  const maxEgt = Math.max(...egtVals, 0);

  const activeFaults = physical.active_faults ?? [];
  const alertLevel = twin.is_anomalous
    ? (twin.fault_confidence > 0.85 ? "critical" : "warning")
    : "nominal";

  const rulBySubsystem = Object.fromEntries((twin.rul_estimates ?? []).map((r) => [r.subsystem, r]));
  const minRul = (twin.rul_estimates ?? []).reduce(
    (min, r) => (r.remaining_flight_hours < min ? r.remaining_flight_hours : min),
    Infinity
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight">UAV Aero-Engine Digital Twin — Ground Control</h1>
          <p className="text-xs text-slate-400">MALE UAV Piston Engine Health Monitoring &amp; Predictive Diagnostics</p>
        </div>
        <div className="flex items-center gap-3">
          {replaying && (
            <span className="px-2 py-1 rounded bg-indigo-600/30 border border-indigo-500 text-indigo-300 text-xs font-semibold">
              REPLAY MODE
            </span>
          )}
          <span
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border ${
              connected ? "border-emerald-500 text-emerald-400 bg-emerald-500/10"
                        : "border-red-500 text-red-400 bg-red-500/10"
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-red-500"}`} />
            {connected ? "TELEMETRY LINK ACTIVE" : "LINK LOST — RECONNECTING"}
          </span>
        </div>
      </div>

      {/* Alert banner */}
      {alertLevel !== "nominal" && (
        <div
          className={`mb-4 p-3 rounded-lg border text-sm ${
            alertLevel === "critical"
              ? "bg-red-950/60 border-red-600 text-red-200"
              : "bg-amber-950/60 border-amber-600 text-amber-200"
          }`}
        >
          <span className="font-bold uppercase mr-2">{alertLevel === "critical" ? "Critical" : "Advisory"}</span>
          {twin.xai_explanation}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* ---- Live Gauge Matrix ---- */}
        <div className="lg:col-span-3 bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wide">Live Gauge Matrix</h2>
          <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            <Gauge label="RPM" value={physical.rpm ?? 0} unit="rpm" min={0} max={6200} warnAt={5600} critAt={6000} />
            <Gauge label="Max CHT" value={maxCht} unit="°C" min={0} max={260} warnAt={135} critAt={150} />
            <Gauge label="Max EGT" value={maxEgt} unit="°C" min={0} max={950} warnAt={820} critAt={880} />
            <Gauge label="Oil Press" value={physical.oil_pressure_kpa ?? 0} unit="kPa" min={0} max={700} warnAt={250} critAt={180} />
            <Gauge label="Vibration" value={physical.vibration_rms_g ?? 0} unit="g RMS" min={0} max={1} decimals={2} warnAt={0.4} critAt={0.6} />
            <Gauge label="Fuel Flow" value={physical.fuel_flow_lph ?? 0} unit="L/h" min={0} max={35} decimals={1} />
          </div>

          {/* Per-cylinder CHT/EGT bars */}
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <h3 className="text-xs text-slate-400 mb-2">Cylinder Head Temp (°C)</h3>
              {chtVals.map((v, i) => (
                <div key={i} className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] w-6 text-slate-500">C{i + 1}</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${v > 150 ? "bg-red-500" : v > 135 ? "bg-amber-500" : "bg-cyan-500"}`}
                      style={{ width: `${clamp((v / 260) * 100, 0, 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] w-10 text-right tabular-nums text-slate-400">{v.toFixed(0)}</span>
                </div>
              ))}
            </div>
            <div>
              <h3 className="text-xs text-slate-400 mb-2">Exhaust Gas Temp (°C)</h3>
              {egtVals.map((v, i) => (
                <div key={i} className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] w-6 text-slate-500">C{i + 1}</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${v > 880 ? "bg-red-500" : v > 820 ? "bg-amber-500" : "bg-orange-400"}`}
                      style={{ width: `${clamp((v / 950) * 100, 0, 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] w-10 text-right tabular-nums text-slate-400">{v.toFixed(0)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Trend sparkline */}
          <div className="mt-4">
            <h3 className="text-xs text-slate-400 mb-1">CHT-1 Trend (last ~60 samples)</h3>
            <Sparkline data={chtHistory} warnLine={135} />
          </div>
        </div>

        {/* ---- Health Indices Panel ---- */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wide">Health Indices</h2>
          <div className="text-center mb-4">
            <div className={`text-4xl font-black ${healthColor(twin.system_health_pct ?? 100)}`}>
              {(twin.system_health_pct ?? 100).toFixed(0)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Overall System Health</div>
          </div>
          <HealthBar label="Cooling" pct={twin.subsystem_health?.cooling} />
          <HealthBar label="Fuel Injection" pct={twin.subsystem_health?.fuel_injection} />
          <HealthBar label="Lubrication" pct={twin.subsystem_health?.lubrication} />
          <HealthBar label="Electrical" pct={twin.subsystem_health?.electrical} />

          <div className="mt-4 pt-3 border-t border-slate-800">
            <h3 className="text-xs text-slate-400 mb-2">Active Faults</h3>
            {activeFaults.length === 0 ? (
              <p className="text-xs text-emerald-400">None — nominal operation</p>
            ) : (
              activeFaults.map((f, i) => (
                <div key={i} className="text-xs text-amber-300 mb-1">
                  ⚠ {f.type} {f.cylinder ? `(Cyl ${f.cylinder})` : ""} — sev {(f.severity * 100).toFixed(0)}%
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mt-4">
        {/* ---- Predictive Analytics / RUL ---- */}
        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wide">
            Predictive Analytics &amp; RUL
          </h2>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div className="bg-slate-950 rounded-lg p-3 text-center border border-slate-800">
              <div className={`text-2xl font-bold ${minRul < 5 ? "text-red-500" : minRul < 20 ? "text-amber-400" : "text-emerald-400"}`}>
                {Number.isFinite(minRul) ? minRul.toFixed(1) : "∞"}
              </div>
              <div className="text-[10px] text-slate-500 uppercase mt-1">Min RUL (flight hrs)</div>
            </div>
            <div className="bg-slate-950 rounded-lg p-3 text-center border border-slate-800">
              <div className="text-2xl font-bold text-cyan-400">
                {((twin.fault_confidence ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-slate-500 uppercase mt-1">Diagnostic Confidence</div>
            </div>
          </div>
          <div className="space-y-2">
            {(twin.rul_estimates ?? []).map((r) => (
              <div key={r.subsystem} className="flex items-center justify-between text-xs bg-slate-950 rounded px-3 py-2 border border-slate-800">
                <span className="capitalize text-slate-300">{r.subsystem.replace("_", " ")}</span>
                <span className={`font-mono ${r.remaining_flight_hours < 5 ? "text-red-400" : r.remaining_flight_hours < 20 ? "text-amber-400" : "text-emerald-400"}`}>
                  {Number.isFinite(r.remaining_flight_hours) ? `${r.remaining_flight_hours.toFixed(1)} hrs` : "stable"}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-300 leading-relaxed">
            <span className="text-slate-500 uppercase text-[10px] block mb-1">AI Root-Cause Explanation</span>
            {twin.xai_explanation ?? "Awaiting telemetry…"}
          </div>
        </div>

        {/* ---- Mission & Simulation Controls ---- */}
        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wide">
            Mission &amp; Simulation Controls
          </h2>

          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <label className="text-[10px] text-slate-500 uppercase">Mission Preset</label>
              <select
                className="w-full mt-1 bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs"
                onChange={(e) => postApi("/api/mission/profile", { profile_name: e.target.value })}
                defaultValue="cruise"
              >
                {MISSION_PRESETS.map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-slate-500 uppercase">Target Altitude (ft): {altitude}</label>
              <input
                type="range" min="0" max="30000" step="500" value={altitude}
                onChange={(e) => setAltitude(Number(e.target.value))}
                onMouseUp={() => postApi("/api/mission/profile", { profile_name: "custom", target_altitude_ft: altitude })}
                className="w-full mt-2"
              />
            </div>
            <div className="col-span-2">
              <label className="text-[10px] text-slate-500 uppercase">Ambient Override Temp (°C): {ambientTemp}</label>
              <input
                type="range" min="-40" max="50" step="1" value={ambientTemp}
                onChange={(e) => setAmbientTemp(Number(e.target.value))}
                onMouseUp={() => postApi("/api/mission/profile", { profile_name: "custom", ambient_override_c: ambientTemp })}
                className="w-full mt-2"
              />
            </div>
          </div>

          <div className="border-t border-slate-800 pt-3 mb-3">
            <label className="text-[10px] text-slate-500 uppercase">Fault Injection</label>
            <div className="grid grid-cols-3 gap-2 mt-1">
              <select
                className="col-span-1 bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs"
                value={selectedFault} onChange={(e) => setSelectedFault(e.target.value)}
              >
                {FAULT_TYPES.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
              <select
                className="col-span-1 bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs"
                value={selectedCylinder} onChange={(e) => setSelectedCylinder(e.target.value)}
              >
                <option value="">All cylinders</option>
                {[1, 2, 3, 4].map((c) => <option key={c} value={c}>Cylinder {c}</option>)}
              </select>
              <input
                type="number" min="0" max="1" step="0.1" value={severity}
                onChange={(e) => setSeverity(Number(e.target.value))}
                className="col-span-1 bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs"
              />
            </div>
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => {
                  sendCommand({
                    action: "inject_fault",
                    fault_type: selectedFault,
                    severity,
                    cylinder: selectedCylinder ? Number(selectedCylinder) : null,
                  });
                  postApi("/api/fault/inject", {
                    fault_type: selectedFault,
                    severity,
                    cylinder: selectedCylinder ? Number(selectedCylinder) : null,
                  });
                }}
                className="flex-1 bg-red-600 hover:bg-red-500 text-white text-xs font-semibold py-2 rounded transition"
              >
                Inject Fault
              </button>
              <button
                onClick={() => {
                  sendCommand({ action: "clear_faults" });
                  postApi("/api/fault/clear");
                }}
                className="flex-1 bg-slate-700 hover:bg-slate-600 text-white text-xs font-semibold py-2 rounded transition"
              >
                Clear Faults
              </button>
            </div>
          </div>

          <div className="border-t border-slate-800 pt-3">
            <label className="text-[10px] text-slate-500 uppercase">Mission Replay</label>
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => postApi("/api/replay/start", { from_index: 0, speed: 2.0 })}
                className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold py-2 rounded transition"
              >
                ▶ Start Replay
              </button>
              <button
                onClick={() => postApi("/api/replay/stop")}
                className="flex-1 bg-slate-700 hover:bg-slate-600 text-white text-xs font-semibold py-2 rounded transition"
              >
                ■ Stop / Live
              </button>
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-4 text-[10px] text-slate-600 text-center">
        Engine Hours: {(physical.engine_hours ?? 0).toFixed(2)} · Sim Seq: {physical.seq ?? "--"} ·
        Digital Twin Sync Layer v1.0
      </footer>
    </div>
  );
}
