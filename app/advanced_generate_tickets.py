"""
============================================================
Advanced Industrial Maintenance Ticket Generation
============================================================

Author:
Mustafa Demetgül

Description:
------------------------------------------------------------
This script generates realistic industrial maintenance
tickets using:

- predictive maintenance datasets
- telemetry statistics
- anomaly detection
- multi-signal risk fusion
- dynamic semantic ticket generation

The generated tickets are designed for:
- RAG systems
- semantic retrieval
- industrial AI assistants
- maintenance reasoning pipelines

============================================================
"""

import random
import numpy as np
import pandas as pd

from advanced_anomaly_utils import (

    calculate_zscore,
    calculate_iqr_bounds,
    is_iqr_outlier,
    calculate_mad,
    estimate_signal_severity,
    calculate_combined_risk,
    SEVERITY_RANK
)

# =========================================================
# LOAD DATASETS
# =========================================================

failures = pd.read_csv(
    "../data/PdM_failures.csv"
)

machines = pd.read_csv(
    "../data/PdM_machines.csv"
)

telemetry = pd.read_csv(
    "../data/PdM_telemetry.csv"
)

# =========================================================
# ISSUE TEMPLATES
# =========================================================

issue_templates = {

    "comp1": {

        "Low": [

            "Minor spindle vibration deviation observed",

            "Slight rotational instability detected",

            "Low-level spindle oscillation identified"
        ],

        "Medium": [

            "Moderate spindle vibration anomaly detected",

            "Mechanical instability observed during operation",

            "Spindle oscillation exceeded operational baseline"
        ],

        "High": [

            "Severe spindle instability detected",

            "High spindle vibration level identified",

            "Strong rotational imbalance observed"
        ],

        "Critical": [

            "Critical spindle vibration anomaly detected",

            "Severe mechanical instability observed",

            "High-risk spindle oscillation beyond normal range",

            "Emergency-level spindle instability identified"
        ]
    },

    "comp2": {

        "Low": [

            "Minor hydraulic pressure fluctuation observed",

            "Slight hydraulic instability detected"
        ],

        "Medium": [

            "Hydraulic pressure anomaly detected",

            "Moderate hydraulic instability observed",

            "Hydraulic pressure fluctuation exceeded operational tolerance",

            "Unstable hydraulic behavior identified during operation"
        ],

        "High": [

            "High hydraulic instability detected",

            "Hydraulic pressure exceeded safe operational profile",

            "Severe hydraulic fluctuation identified"
        ],

        "Critical": [

            "Critical hydraulic pressure instability detected",

            "Severe hydraulic operational anomaly observed",

            "Emergency hydraulic instability identified"
        ]
    },

    "comp3": {

        "Low": [

            "Minor thermal deviation detected on motor unit"
        ],

        "Medium": [

            "Motor temperature anomaly detected",

            "Moderate thermal instability observed"
        ],

        "High": [

            "Elevated thermal instability detected",

            "Critical-range temperature fluctuation observed",

            "High thermal stress identified"
        ],

        "Critical": [

            "Critical motor overheating detected",

            "Severe thermal operational anomaly observed",

            "Emergency thermal instability detected"
        ]
    },

    "comp4": {

        "Low": [

            "Minor automation instability detected"
        ],

        "Medium": [

            "PLC communication instability observed",

            "Automation system anomaly detected"
        ],

        "High": [

            "High-risk automation instability detected",

            "Industrial control synchronization instability observed",

            "Critical PLC instability trend identified"
        ],

        "Critical": [

            "Critical automation system instability detected",

            "Severe PLC communication failure observed",

            "Emergency automation system instability identified"
        ]
    }
}

# =========================================================
# ROOT CAUSE TEMPLATES
# =========================================================

root_cause_templates = {

    "comp1": [
        "Bearing wear",
        "Mechanical imbalance",
        "Lubrication degradation"
    ],

    "comp2": [
        "Hydraulic valve degradation",
        "Pressure leakage",
        "Hydraulic flow instability"
    ],

    "comp3": [
        "Cooling system degradation",
        "Thermal overload",
        "Cooling fan malfunction"
    ],

    "comp4": [
        "PLC communication instability",
        "Electrical synchronization issue",
        "Control signal degradation"
    ]
}

# =========================================================
# SIGNAL-BASED SOLUTIONS
# =========================================================

signal_solutions = {

    "vibration": [

        "Inspect spindle bearing and lubrication system",

        "Verify shaft balancing and alignment",

        "Check rotational stability"
    ],

    "pressure": [

        "Inspect hydraulic pressure lines",

        "Verify hydraulic pump condition",

        "Check valve stability"
    ],

    "rotation": [

        "Inspect rotational control subsystem",

        "Verify rotational sensor consistency",

        "Check rotational synchronization"
    ]
}

# =========================================================
# GENERATE TICKETS
# =========================================================

tickets = []

for idx, row in failures.iterrows():

    machine_id = row["machineID"]

    failure = row["failure"]

    # =====================================================
    # MACHINE INFO
    # =====================================================

    machine_row = machines[
        machines["machineID"] == machine_id
    ]

    if len(machine_row) == 0:
        continue

    model = machine_row.iloc[0]["model"]

    age = machine_row.iloc[0]["age"]

    # =====================================================
    # MACHINE TELEMETRY
    # =====================================================

    machine_telemetry = telemetry[
        telemetry["machineID"] == machine_id
    ]

    if len(machine_telemetry) == 0:
        continue

    sample = machine_telemetry.sample(1).iloc[0]

    # =====================================================
    # SIGNALS
    # =====================================================

    vibration = sample["vibration"]

    pressure = sample["pressure"]

    rotate = sample["rotate"]

    # =====================================================
    # MACHINE BASELINES
    # =====================================================

    vib_values = machine_telemetry["vibration"].values

    pressure_values = machine_telemetry["pressure"].values

    rotate_values = machine_telemetry["rotate"].values

    # =====================================================
    # Z-SCORES
    # =====================================================

    vib_z = calculate_zscore(

        vibration,
        np.mean(vib_values),
        np.std(vib_values)
    )

    pressure_z = calculate_zscore(

        pressure,
        np.mean(pressure_values),
        np.std(pressure_values)
    )

    rotate_z = calculate_zscore(

        rotate,
        np.mean(rotate_values),
        np.std(rotate_values)
    )

    # =====================================================
    # IQR
    # =====================================================

    vib_lower, vib_upper = calculate_iqr_bounds(
        vib_values
    )

    pressure_lower, pressure_upper = calculate_iqr_bounds(
        pressure_values
    )

    rotate_lower, rotate_upper = calculate_iqr_bounds(
        rotate_values
    )

    vib_iqr = is_iqr_outlier(
        vibration,
        vib_lower,
        vib_upper
    )

    pressure_iqr = is_iqr_outlier(
        pressure,
        pressure_lower,
        pressure_upper
    )

    rotate_iqr = is_iqr_outlier(
        rotate,
        rotate_lower,
        rotate_upper
    )

    # =====================================================
    # MAD
    # =====================================================

    vib_mad = calculate_mad(vib_values)

    pressure_mad = calculate_mad(pressure_values)

    rotate_mad = calculate_mad(rotate_values)

    # =====================================================
    # SIGNAL SEVERITY
    # =====================================================

    vib_severity, vib_score = estimate_signal_severity(

        vib_z,
        vib_iqr,
        vib_mad
    )

    pressure_severity, pressure_score = (
        estimate_signal_severity(

            pressure_z,
            pressure_iqr,
            pressure_mad
        )
    )

    rotate_severity, rotate_score = (
        estimate_signal_severity(

            rotate_z,
            rotate_iqr,
            rotate_mad
        )
    )

    # =====================================================
    # COMBINED RISK
    # =====================================================

    risk_score, priority, dominant_signal = (
        calculate_combined_risk(

            vib_score,
            pressure_score,
            rotate_score,
            age
        )
    )

    # =====================================================
    # FINAL SEVERITY
    # =====================================================

    severities = [

        vib_severity,
        pressure_severity,
        rotate_severity
    ]

    final_severity = max(

        severities,
        key=lambda x: SEVERITY_RANK[x]
    )

    # =====================================================
    # PRIORITY ALIGNMENT
    # =====================================================

    if SEVERITY_RANK[priority] > SEVERITY_RANK[final_severity]:

        final_severity = priority

    # =====================================================
    # DYNAMIC TEXT GENERATION
    # =====================================================

    issue = random.choice(
        issue_templates[failure][final_severity]
    )

    root_cause = random.choice(
        root_cause_templates[failure]
    )

    solution = random.choice(
        signal_solutions[dominant_signal]
    )

    # =====================================================
    # AGING MACHINE CONTEXT
    # =====================================================

    if age > 15 and final_severity in [

        "High",
        "Critical"
    ]:

        issue += " on aging equipment"

    # =====================================================
    # CREATE TICKET
    # =====================================================

    ticket = {

        "ticket_id": idx + 1,

        "machine": f"{model}_{machine_id}",

        "department": failure,
        
        "dominant_signal_score": max(
           vib_score,
           pressure_score,
           rotate_score
        ),
        
        "maintenance_team": {

           "comp1": "Mechanical",

           "comp2": "Hydraulic",

           "comp3": "Thermal",

           "comp4": "Automation"

        }[failure],

        "issue": issue,

        "priority": final_severity,

        "risk_score": round(risk_score, 2),

        "dominant_signal": dominant_signal,

        "vibration_zscore": round(vib_z, 2),

        "pressure_zscore": round(pressure_z, 2),

        "rotate_zscore": round(rotate_z, 2),

        "root_cause": root_cause,

        "solution": solution,

        "status": "Open"
    }

    tickets.append(ticket)

# =========================================================
# SAVE DATASET
# =========================================================

tickets_df = pd.DataFrame(tickets)

tickets_df.to_csv(

    "../data/maintenance_tickets.csv",
    index=False
)

print("\nAdvanced industrial tickets generated.")

print(f"Total tickets: {len(tickets_df)}")