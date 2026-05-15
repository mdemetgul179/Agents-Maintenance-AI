"""
============================================================
ADVANCED INDUSTRIAL ANOMALY UTILITIES
============================================================

Author:
Mustafa Demetgül

Project:
AI-Based Industrial Maintenance Ticket Generation
Using Statistical Multi-Signal Anomaly Detection

Description
------------------------------------------------------------
This module provides advanced statistical anomaly analysis
functions for synthetic industrial maintenance ticket
generation.

The anomaly reasoning system is based on multiple
statistical techniques rather than simple thresholding.

Implemented methods:
------------------------------------------------------------

1. Z-Score Analysis
   Detects deviations from machine-specific mean behavior.

2. IQR-Based Outlier Detection
   Detects abnormal values using quartile statistics.

3. MAD (Median Absolute Deviation)
   Robust anomaly scoring resistant to extreme outliers.

4. Multi-Signal Risk Fusion
   Combines vibration, hydraulic pressure, and rotational
   instability into a unified industrial risk score.

5. Aging-Based Risk Escalation
   Older machines receive additional risk weighting.

6. Severity-Based Prioritization
   Generates realistic maintenance priorities:
   - Low
   - Medium
   - High
   - Critical

This module is designed for:
------------------------------------------------------------
- Synthetic maintenance dataset generation
- Industrial AI benchmarking
- RAG-based maintenance assistants
- Retrieval evaluation
- Predictive maintenance research
- AI-powered anomaly reasoning systems

============================================================
"""


import numpy as np


# ============================================================
# SEVERITY RANKING
# ============================================================

SEVERITY_RANK = {

    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}


# ============================================================
# Z-SCORE
# ============================================================

def calculate_zscore(value, mean, std):

    if std == 0:
        return 0.0

    return abs((value - mean) / std)


# ============================================================
# IQR
# ============================================================

def calculate_iqr_bounds(values):

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


def is_iqr_outlier(value, lower, upper):

    return value < lower or value > upper


# ============================================================
# MAD
# ============================================================

def calculate_mad(values):

    median = np.median(values)

    deviations = np.abs(values - median)

    mad = np.median(deviations)

    return mad


# ============================================================
# SIGNAL SEVERITY
# ============================================================

def estimate_signal_severity(

    z_score,
    iqr_anomaly,
    mad_score

):

    fusion_score = 0

    # ========================================================
    # Z-SCORE CONTRIBUTION
    # ========================================================

    if z_score >= 3:

        fusion_score += 3

    elif z_score >= 2:

        fusion_score += 2

    elif z_score >= 1:

        fusion_score += 1

    # ========================================================
    # IQR CONTRIBUTION
    # ========================================================

    if iqr_anomaly:

        fusion_score += 1

    # ========================================================
    # MAD CONTRIBUTION
    # ========================================================

    if mad_score >= 1.5:

        fusion_score += 1

    # ========================================================
    # FINAL SEVERITY
    # ========================================================

    if fusion_score <= 1:

        severity = "Low"

    elif fusion_score <= 3:

        severity = "Medium"

    elif fusion_score <= 5:

        severity = "High"

    else:

        severity = "Critical"

    return severity, fusion_score


# ============================================================
# MULTI-SIGNAL RISK FUSION
# ============================================================

def calculate_combined_risk(

    vib_score,
    pressure_score,
    rotate_score,
    age=0

):

    signal_scores = {

        "vibration": vib_score,
        "pressure": pressure_score,
        "rotation": rotate_score
    }

    dominant_signal = max(

        signal_scores,
        key=signal_scores.get
    )

    scores = list(signal_scores.values())

    max_score = max(scores)

    avg_score = np.mean(scores)

    # ========================================================
    # COUNT ANOMALIES
    # ========================================================

    anomaly_count = sum(

        s >= 2 for s in scores
    )

    severe_count = sum(

        s >= 3 for s in scores
    )

    # ========================================================
    # BASE RISK
    # ========================================================

    risk = (

        max_score * 0.5 +
        avg_score * 0.3 +
        anomaly_count * 0.4 +
        severe_count * 0.8
    )

    # ========================================================
    # MULTI-SIGNAL BONUS
    # ========================================================

    if vib_score >= 2 and rotate_score >= 2:

        risk += 0.6

    if pressure_score >= 2 and rotate_score >= 2:

        risk += 0.5

    if vib_score >= 2 and pressure_score >= 2:

        risk += 0.5

    # ========================================================
    # AGING BONUS
    # ========================================================

    if age > 20:

        risk += 0.8

    elif age > 15:

        risk += 0.5
    # ========================================================
    # RANDOM REALISM ADJUSTMENT
    # ========================================================

    risk += np.random.uniform(-0.15, 0.15)

    risk = max(risk, 0)

    # ========================================================
    # PRIORITY ASSIGNMENT
    # ========================================================

    if risk < 1.0:

        priority = "Low"

    elif risk < 2.5:

        priority = "Medium"

    elif risk < 4.0:

        priority = "High"

    else:

        priority = "Critical"

    return (

        round(risk, 2),
        priority,
        dominant_signal
    )