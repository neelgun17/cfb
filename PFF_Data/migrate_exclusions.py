import json
from pathlib import Path

# Paths
exclusions_path = Path("/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/data/manual_exclusions.txt")
output_path = Path("/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/data/status_tracker.json")

# Read exclusions
status_data = {}
if exclusions_path.exists():
    with open(exclusions_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Defaulting to 'Draft' as per heuristic, user can edit later
                status_data[line] = "Draft"

# Write JSON
with open(output_path, 'w') as f:
    json.dump(status_data, f, indent=2)

print(f"Created {output_path} with {len(status_data)} entries.")
