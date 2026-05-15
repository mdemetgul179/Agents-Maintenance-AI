import pandas as pd

# Load datasets
failures = pd.read_csv("../data/PdM_failures.csv")
machines = pd.read_csv("../data/PdM_machines.csv")

print("Failures loaded:", len(failures))
print("Machines loaded:", len(machines))

# Failure mappings
failure_map = {

    "comp1": {
        "department": "Mechanical",
        "issue": "Abnormal spindle vibration detected during operation",
        "root_cause": "Bearing wear",
        "solution": "Inspect spindle bearing and lubrication system"
    },

    "comp2": {
        "department": "Hydraulic",
        "issue": "Hydraulic pressure instability detected",
        "root_cause": "Hydraulic valve degradation",
        "solution": "Inspect hydraulic valves and pressure lines"
    },

    "comp3": {
        "department": "Electrical",
        "issue": "Motor overheating detected during operation",
        "root_cause": "Cooling system failure",
        "solution": "Inspect cooling fan and airflow system"
    },

    "comp4": {
        "department": "Automation",
        "issue": "Unexpected machine shutdown detected",
        "root_cause": "Electrical control instability",
        "solution": "Inspect PLC and electrical control system"
    }
}

tickets = []

# Generate tickets
for idx, row in failures.iterrows():

    machine_id = row["machineID"]
    failure = row["failure"]

    # Skip unknown failures
    if failure not in failure_map:
        continue

    mapping = failure_map[failure]

    # Get machine info
    machine_rows = machines[
        machines["machineID"] == machine_id
    ]

    if len(machine_rows) > 0:

        # safer access
        
        model = str(machine_rows.iloc[0]["model"])

    else:

        model = "Machine"

    ticket = {

        "ticket_id": idx + 1,

        "machine": f"{model}_{machine_id}",

        "department": mapping["department"],

        "issue": mapping["issue"],

        "priority": "High",

        "root_cause": mapping["root_cause"],

        "solution": mapping["solution"],

        "status": "Open"
    }

    tickets.append(ticket)

# Convert to DataFrame
tickets_df = pd.DataFrame(tickets)

# Save CSV
tickets_df.to_csv(
    "../data/maintenance_tickets.csv",
    index=False
)

print("\nTickets generated successfully.")
print("Total tickets:", len(tickets_df))