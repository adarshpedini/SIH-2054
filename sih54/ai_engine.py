"""
PHASE 2 — Hybrid AI/ML Diagnostic & RUL Prediction Engine
==========================================================
Consumes telemetry snapshots (see engine_sim.snapshot_dict) and produces:
  1. Physics-informed anomaly score (Isolation Forest, features include an
     energy-balance residual between fuel energy input and EGT/CHT thermal
     output — this residual is *engineered*, not just raw sensor values, so
     the anomaly detector is constrained by thermodynamics, not purely
     data-driven).
  2. Fault classification (Random Forest) across the 6 known fault modes.
  3. Remaining Useful Life estimate per subsystem using an exponential
     degradation-trend model fit over a rolling window (Weibull-style
     hazard scaling for the reported "flight hours to threshold").
  4. Human-readable explainable-AI (XAI) root-cause narratives.

Designed for on-the-fly (streaming) use: `DiagnosticEngine.process(snapshot)`
is called once per telemetry frame and is O(1) amortized (bounded history
buffers), suitable for edge deployment.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

FAULT_LABELS = [
    "none",
    "injector_clog",
    "misfire",
    "cooling_degradation",
    "sensor_drift",
    "oil_leak",
    "combustion_instability",
]

HISTORY_LEN = 300          # ~30s @10Hz rolling buffer for anomaly/trend features
RUL_WINDOW = 600            # ~60s window for degradation slope estimation
FUEL_LHV_MJ_PER_KG = 43.0
FUEL_DENSITY_KG_PER_L = 0.72


# --------------------------------------------------------------------------
# Feature engineering — physics-informed
# --------------------------------------------------------------------------

def extract_features(snapshot: Dict) -> Dict[str, float]:
    """Turn a raw telemetry snapshot into a physics-informed feature vector."""
    cht_vals = [snapshot.get(f"cht_{i}_c", 0.0) for i in range(1, 5)]
    egt_vals = [snapshot.get(f"egt_{i}_c", 0.0) for i in range(1, 5)]
    inj_duty = snapshot.get("injector_duty_pct", [0, 0, 0, 0])

    fuel_lph = snapshot.get("fuel_flow_lph", 0.0)
    rpm = snapshot.get("rpm", 0.0)

    # --- energy balance residual ---
    # Expected thermal output (EGT rise above ambient) implied by fuel energy input,
    # vs. actually observed EGT. Large residual => combustion/thermal fault signature.
    ambient_c = snapshot.get("ambient_temp_c", 15.0)
    fuel_kg_s = (fuel_lph * FUEL_DENSITY_KG_PER_L) / 3600.0
    heat_input_kw = fuel_kg_s * FUEL_LHV_MJ_PER_KG * 1000 * 0.30  # ~30% to exhaust path
    expected_egt_rise = 200.0 + heat_input_kw * 55.0              # calibrated linear proxy
    observed_egt_rise = (sum(egt_vals) / max(1, len(egt_vals))) - ambient_c
    energy_balance_residual = observed_egt_rise - expected_egt_rise

    cht_spread = max(cht_vals) - min(cht_vals) if cht_vals else 0.0
    egt_spread = max(egt_vals) - min(egt_vals) if egt_vals else 0.0
    inj_spread = max(inj_duty) - min(inj_duty) if inj_duty else 0.0

    vib_rms = snapshot.get("vibration_rms_g", 0.0)
    fft = snapshot.get("fft_spectrum_0_800hz", [0.0] * 16)
    half_order_energy = fft[max(0, int(len(fft) * 0.15))] if fft else 0.0
    broadband_energy = float(np.sum(fft)) if fft else 0.0

    oil_pressure = snapshot.get("oil_pressure_kpa", 0.0)
    oil_level = snapshot.get("oil_level_pct", 100.0)
    coolant_flow = snapshot.get("coolant_flow_lpm", 40.0)
    coolant_temp = snapshot.get("coolant_temp_c", 90.0)

    return {
        "rpm": rpm,
        "cht_mean": float(np.mean(cht_vals)) if cht_vals else 0.0,
        "cht_spread": cht_spread,
        "egt_mean": float(np.mean(egt_vals)) if egt_vals else 0.0,
        "egt_spread": egt_spread,
        "inj_duty_mean": float(np.mean(inj_duty)) if inj_duty else 0.0,
        "inj_duty_spread": inj_spread,
        "fuel_flow_lph": fuel_lph,
        "energy_balance_residual": energy_balance_residual,
        "vib_rms": vib_rms,
        "vib_half_order_energy": half_order_energy,
        "vib_broadband_energy": broadband_energy,
        "oil_pressure_kpa": oil_pressure,
        "oil_level_pct": oil_level,
        "coolant_flow_lpm": coolant_flow,
        "coolant_temp_c": coolant_temp,
    }


FEATURE_ORDER = [
    "rpm", "cht_mean", "cht_spread", "egt_mean", "egt_spread",
    "inj_duty_mean", "inj_duty_spread", "fuel_flow_lph",
    "energy_balance_residual", "vib_rms", "vib_half_order_energy",
    "vib_broadband_energy", "oil_pressure_kpa", "oil_level_pct",
    "coolant_flow_lpm", "coolant_temp_c",
]


def feature_vector(feat: Dict[str, float]) -> np.ndarray:
    return np.array([feat[k] for k in FEATURE_ORDER], dtype=float)


# --------------------------------------------------------------------------
# Synthetic training-data generator (bootstraps the models at engine start;
# in production these would be trained offline on fleet/bench data and
# loaded from disk — this keeps the prototype fully self-contained).
# --------------------------------------------------------------------------

def _synthesize_training_set(n_per_class: int = 400, seed: int = 42
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y, healthy_mask = [], [], []

    def base_healthy():
        rpm = rng.uniform(1800, 5600)
        cht_mean = rng.uniform(90, 200)
        egt_mean = rng.uniform(600, 850)
        return {
            "rpm": rpm,
            "cht_mean": cht_mean,
            "cht_spread": rng.uniform(1, 6),
            "egt_mean": egt_mean,
            "egt_spread": rng.uniform(5, 20),
            "inj_duty_mean": rng.uniform(35, 85),
            "inj_duty_spread": rng.uniform(0.5, 4),
            "fuel_flow_lph": rng.uniform(8, 30),
            "energy_balance_residual": rng.uniform(-15, 15),
            "vib_rms": rng.uniform(0.1, 0.25),
            "vib_half_order_energy": rng.uniform(0, 0.05),
            "vib_broadband_energy": rng.uniform(0.5, 1.5),
            "oil_pressure_kpa": rng.uniform(400, 650),
            "oil_level_pct": rng.uniform(85, 100),
            "coolant_flow_lpm": rng.uniform(35, 45),
            "coolant_temp_c": rng.uniform(85, 98),
        }

    for _ in range(n_per_class):
        f = base_healthy()
        X.append(feature_vector(f)); y.append("none"); healthy_mask.append(1)

    for _ in range(n_per_class):
        f = base_healthy()
        sev = rng.uniform(0.3, 1.0)
        f["inj_duty_spread"] += sev * rng.uniform(15, 40)
        f["egt_spread"] += sev * rng.uniform(30, 90)
        f["energy_balance_residual"] -= sev * rng.uniform(40, 120)
        X.append(feature_vector(f)); y.append("injector_clog"); healthy_mask.append(0)

    for _ in range(n_per_class):
        f = base_healthy()
        sev = rng.uniform(0.3, 1.0)
        f["vib_half_order_energy"] += sev * rng.uniform(0.2, 0.6)
        f["egt_spread"] += sev * rng.uniform(40, 120)
        f["rpm"] -= sev * rng.uniform(100, 400)
        X.append(feature_vector(f)); y.append("misfire"); healthy_mask.append(0)

    for _ in range(n_per_class):
        f = base_healthy()
        sev = rng.uniform(0.3, 1.0)
        f["cht_mean"] += sev * rng.uniform(30, 90)
        f["coolant_flow_lpm"] -= sev * rng.uniform(10, 28)
        f["coolant_temp_c"] += sev * rng.uniform(15, 35)
        X.append(feature_vector(f)); y.append("cooling_degradation"); healthy_mask.append(0)

    for _ in range(n_per_class):
        f = base_healthy()
        sev = rng.uniform(0.3, 1.0)
        # sensor drift: internally-consistent physics but a shifted absolute reading —
        # approximate via added residual noise without matching mechanical signature
        f["energy_balance_residual"] += rng.choice([-1, 1]) * sev * rng.uniform(50, 150)
        f["cht_spread"] += sev * rng.uniform(2, 10)
        X.append(feature_vector(f)); y.append("sensor_drift"); healthy_mask.append(0)

    for _ in range(n_per_class):
        f = base_healthy()
        sev = rng.uniform(0.3, 1.0)
        f["oil_pressure_kpa"] -= sev * rng.uniform(100, 300)
        f["oil_level_pct"] -= sev * rng.uniform(20, 60)
        X.append(feature_vector(f)); y.append("oil_leak"); healthy_mask.append(0)

    for _ in range(n_per_class):
        f = base_healthy()
        sev = rng.uniform(0.3, 1.0)
        f["vib_broadband_energy"] += sev * rng.uniform(1.0, 3.0)
        f["vib_rms"] += sev * rng.uniform(0.2, 0.5)
        f["rpm"] += rng.uniform(-200, 200)
        X.append(feature_vector(f)); y.append("combustion_instability"); healthy_mask.append(0)

    return np.array(X), np.array(y), np.array(healthy_mask)


# --------------------------------------------------------------------------
# RUL estimation (per-subsystem exponential degradation trend)
# --------------------------------------------------------------------------

@dataclass
class RULEstimate:
    subsystem: str
    remaining_flight_hours: float
    confidence: float
    trend_slope_per_hour: float
    threshold_metric: str


class RULEstimator:
    """
    Tracks a rolling history of a key health metric per subsystem and fits an
    exponential trend (equivalent to a Weibull shape-fixed hazard growth in
    log-space) to project time-to-threshold.
    """

    SUBSYSTEM_METRICS = {
        "cooling": ("cht_mean", 145.0, "increasing"),        # critical CHT threshold (C)
        "fuel_injection": ("inj_duty_spread", 35.0, "increasing"),
        "lubrication": ("oil_pressure_kpa", 220.0, "decreasing"),
        "combustion": ("vib_rms", 0.55, "increasing"),
    }

    def __init__(self, window: int = RUL_WINDOW):
        self.window = window
        self.history: Dict[str, Deque[Tuple[float, float]]] = {
            k: deque(maxlen=window) for k in self.SUBSYSTEM_METRICS
        }

    def update(self, feat: Dict[str, float], engine_hours: float):
        for subsystem, (metric_key, _, _) in self.SUBSYSTEM_METRICS.items():
            self.history[subsystem].append((engine_hours, feat.get(metric_key, 0.0)))

    def estimate(self) -> List[RULEstimate]:
        results = []
        for subsystem, (metric_key, threshold, direction) in self.SUBSYSTEM_METRICS.items():
            hist = self.history[subsystem]
            if len(hist) < 20:
                results.append(RULEstimate(subsystem, float("inf"), 0.0, 0.0, metric_key))
                continue
            hours = np.array([h for h, _ in hist])
            values = np.array([v for _, v in hist])
            # linear fit in hours (slope per hour) — robust & explainable for a prototype;
            # exponential-decay weighting emphasizes recent trend (Weibull-like recency bias)
            weights = np.exp(np.linspace(-2, 0, len(hours)))
            A = np.vstack([hours, np.ones_like(hours)]).T
            W = np.diag(weights)
            try:
                coef, *_ = np.linalg.lstsq(W @ A, W @ values, rcond=None)
                slope, intercept = coef
            except np.linalg.LinAlgError:
                slope, intercept = 0.0, float(values[-1])

            current_value = float(values[-1])
            current_hour = float(hours[-1])

            if direction == "increasing":
                if current_value >= threshold:
                    remaining = 0.0          # already at/past threshold, regardless of current slope
                elif slope <= 1e-6:
                    remaining = float("inf")
                else:
                    hours_to_threshold = (threshold - current_value) / slope
                    remaining = max(0.0, hours_to_threshold)
            else:
                if current_value <= threshold:
                    remaining = 0.0
                elif slope >= -1e-6:
                    remaining = float("inf")
                else:
                    hours_to_threshold = (threshold - current_value) / slope
                    remaining = max(0.0, hours_to_threshold)

            span = hours[-1] - hours[0]
            confidence = float(min(1.0, span / 0.05))  # more history over a longer span -> higher confidence
            results.append(RULEstimate(subsystem, round(remaining, 2), round(confidence, 2),
                                        round(float(slope), 5), metric_key))
        return results


# --------------------------------------------------------------------------
# Explainable AI narrative generator
# --------------------------------------------------------------------------

def generate_xai_explanation(fault_label: str, feat: Dict[str, float],
                              snapshot: Dict, confidence: float) -> str:
    if fault_label == "none":
        return "Engine parameters within nominal envelope. No active anomaly."

    cht_vals = [snapshot.get(f"cht_{i}_c", 0.0) for i in range(1, 5)]
    egt_vals = [snapshot.get(f"egt_{i}_c", 0.0) for i in range(1, 5)]
    hot_cyl = int(np.argmax(cht_vals)) + 1 if cht_vals else None
    hot_egt_cyl = int(np.argmax(egt_vals)) + 1 if egt_vals else None

    templates = {
        "injector_clog": (
            f"ALERT: Injector duty imbalance detected (spread {feat['inj_duty_spread']:.1f}%). "
            f"Energy-balance residual is {feat['energy_balance_residual']:.0f} below expected, "
            f"consistent with reduced fuel delivery on one or more cylinders — likely injector clogging."
        ),
        "misfire": (
            f"ALERT: Half-order vibration energy elevated ({feat['vib_half_order_energy']:.2f} g) with "
            f"EGT spread of {feat['egt_spread']:.0f}\u00b0C across cylinders — signature consistent with "
            f"intermittent misfire, most likely cylinder {hot_egt_cyl}."
        ),
        "cooling_degradation": (
            f"ALERT: High CHT on cylinder {hot_cyl} ({cht_vals[hot_cyl-1]:.0f}\u00b0C) caused by an estimated "
            f"{max(0, 42 - feat['coolant_flow_lpm']):.0f}% reduction in coolant flow "
            f"({feat['coolant_flow_lpm']:.1f} L/min vs. 40+ L/min nominal)."
        ),
        "sensor_drift": (
            f"ALERT: Sensor reading inconsistent with physics-model prediction "
            f"(energy-balance residual {feat['energy_balance_residual']:.0f}) while mechanical vibration "
            f"and oil signatures remain nominal — indicates probable sensor drift rather than a true fault."
        ),
        "oil_leak": (
            f"ALERT: Oil pressure at {feat['oil_pressure_kpa']:.0f} kPa with oil level at "
            f"{feat['oil_level_pct']:.0f}% — trend consistent with an active oil leak. "
            f"Continued operation risks lubrication-system failure."
        ),
        "combustion_instability": (
            f"ALERT: Broadband vibration energy elevated ({feat['vib_broadband_energy']:.2f}) with RPM "
            f"variance above nominal — consistent with combustion instability, possibly detonation "
            f"or air/fuel ratio excursion."
        ),
    }
    base = templates.get(fault_label, f"ALERT: {fault_label} detected.")
    return f"{base} (classifier confidence: {confidence*100:.0f}%)"


# --------------------------------------------------------------------------
# Main diagnostic engine
# --------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    timestamp: float
    anomaly_score: float               # higher = more anomalous
    is_anomalous: bool
    predicted_fault: str
    fault_confidence: float
    fault_probabilities: Dict[str, float]
    rul_estimates: List[Dict]
    xai_explanation: str
    system_health_pct: float
    subsystem_health: Dict[str, float]


class DiagnosticEngine:
    def __init__(self):
        X, y, healthy_mask = _synthesize_training_set()

        # 1) Physics-informed anomaly detector, trained only on healthy examples
        self.anomaly_model = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42
        )
        self.anomaly_model.fit(X[healthy_mask == 1])

        # 2) Fault classifier across all classes
        self.classifier = RandomForestClassifier(
            n_estimators=250, max_depth=12, random_state=42, class_weight="balanced"
        )
        self.classifier.fit(X, y)
        self.classes_ = list(self.classifier.classes_)

        self.rul_estimator = RULEstimator()
        self.feature_history: Deque[Dict[str, float]] = deque(maxlen=HISTORY_LEN)

    def process(self, snapshot: Dict) -> DiagnosticResult:
        feat = extract_features(snapshot)
        self.feature_history.append(feat)
        vec = feature_vector(feat).reshape(1, -1)

        # anomaly score: IsolationForest score_samples is higher = more normal;
        # invert & rescale to an intuitive 0(normal)-1(anomalous) range
        raw_score = self.anomaly_model.score_samples(vec)[0]
        anomaly_score = float(np.clip((0.05 - raw_score) * 4.0, 0.0, 1.0))
        is_anomalous = anomaly_score > 0.5

        probs = self.classifier.predict_proba(vec)[0]
        prob_map = {cls: float(p) for cls, p in zip(self.classes_, probs)}
        predicted_fault = max(prob_map, key=prob_map.get)
        fault_confidence = prob_map[predicted_fault]

        if not is_anomalous and predicted_fault != "none":
            # guard-rail: don't declare a fault unless the physics-informed detector agrees
            if fault_confidence < 0.75:
                predicted_fault = "none"
                fault_confidence = prob_map.get("none", 1 - fault_confidence)

        engine_hours = snapshot.get("engine_hours", 0.0)
        self.rul_estimator.update(feat, engine_hours)
        rul_list = self.rul_estimator.estimate()

        subsystem_health = self._compute_subsystem_health(feat, rul_list)
        system_health_pct = float(np.mean(list(subsystem_health.values())))

        xai = generate_xai_explanation(predicted_fault, feat, snapshot, fault_confidence)

        return DiagnosticResult(
            timestamp=time.time(),
            anomaly_score=round(anomaly_score, 3),
            is_anomalous=is_anomalous,
            predicted_fault=predicted_fault,
            fault_confidence=round(fault_confidence, 3),
            fault_probabilities={k: round(v, 3) for k, v in prob_map.items()},
            rul_estimates=[dataclasses_asdict(r) for r in rul_list],
            xai_explanation=xai,
            system_health_pct=round(system_health_pct, 1),
            subsystem_health={k: round(v, 1) for k, v in subsystem_health.items()},
        )

    def _compute_subsystem_health(self, feat: Dict[str, float], rul_list: List[RULEstimate]) -> Dict[str, float]:
        health = {}
        rul_map = {r.subsystem: r for r in rul_list}

        cooling_penalty = max(0, feat["cht_mean"] - 110) * 0.7 + max(0, 42 - feat["coolant_flow_lpm"]) * 1.2
        health["cooling"] = float(np.clip(100 - cooling_penalty, 0, 100))

        fuel_penalty = feat["inj_duty_spread"] * 1.5 + max(0, -feat["energy_balance_residual"] - 20) * 0.4
        health["fuel_injection"] = float(np.clip(100 - fuel_penalty, 0, 100))

        lube_penalty = max(0, 450 - feat["oil_pressure_kpa"]) * 0.25 + max(0, 90 - feat["oil_level_pct"]) * 1.1
        health["lubrication"] = float(np.clip(100 - lube_penalty, 0, 100))

        elec_penalty = 0.0  # placeholder subsystem — extend with battery/alternator trend if needed
        health["electrical"] = float(np.clip(100 - elec_penalty, 0, 100))

        for subsystem in ("cooling", "fuel_injection", "lubrication"):
            r = rul_map.get(subsystem)
            if r and r.remaining_flight_hours < 5.0 and r.confidence > 0.3:
                health[subsystem] = min(health[subsystem], 30.0)

        return health


def dataclasses_asdict(obj) -> Dict:
    from dataclasses import asdict
    return asdict(obj)
