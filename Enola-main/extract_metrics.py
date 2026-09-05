import os
import json
import glob
import pandas as pd

def extract_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {
        "filename": os.path.basename(json_path),
        "cir_fidelity": data.get("cir_fidelity"),
        "cir_fidelity_1q_gate": data.get("cir_fidelity_1q_gate"),
        "cir_fidelity_2q_gate": data.get("cir_fidelity_2q_gate"),
        "cir_fidelity_2q_gate_for_idle": data.get("cir_fidelity_2q_gate_for_idle"),
        "cir_fidelity_atom_transfer": data.get("cir_fidelity_atom_transfer"),
        "cir_fidelity_coherence": data.get("cir_fidelity_coherence"),
        "num_movement_stage": data.get("num_movement_stage"),
        "average_movement": data.get("average_movement")
    }

def main(directory, output_excel="parsed_metrics.xlsx"):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    all_metrics = []

    for json_path in json_files:
        metrics = extract_metrics(json_path)
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    df.to_excel(output_excel, index=False)
    print(f"Parsed metrics written to {output_excel}")

if __name__ == "__main__":
    # Change '.' to your directory containing the JSON files if needed
    main(directory='results/fidelity/fidelity with SA mapping', output_excel="parsed_metrics.xlsx")
