
import json
import tempfile
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data
results = {
    "control": json.load(open("control.json", "r")),
    "cwe_rules": json.load(open("cwe_rules.json", "r")),
    "all_rules": json.load(open("all_rules.json", "r"))
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]

# Dictionary to hold compilation results
compilation_results = {}

for category, data in results.items():
    for i, entry in enumerate(data):
        completion_text = " ".join(entry["completion"])
        segments = completion_text.split("```")
        
        if len(segments) > 1:
            c_code = segments[1]
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as tmp:
                tmp.write(c_code.encode("utf-8"))
                tmp_name = tmp.name
            
            cmd = ["gcc", "-fsyntax-only", tmp_name]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                success = (result.returncode == 0)
            except Exception:
                success = False
            
            compilation_results[(category, i)] = success
        else:
            compilation_results[(category, i)] = False

categories = ["control", "cwe_rules", "all_rules"]
success_counts = {cat: 0 for cat in categories}

for (category, idx), success in compilation_results.items():
    if success is True:
        success_counts[category] += 1

x = np.arange(len(categories))
counts = [success_counts[cat] for cat in categories]

plt.figure(figsize=(5, 5))
plt.bar(x, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.xticks(x, ["Control", "CWE Rules", "All Rules"])
plt.ylabel("Number of Successful Compilations")
plt.title("Successful Compilations by Category")

# Adjust the y-limit to give extra space at the top
max_count = max(counts) if counts else 0
plt.ylim(0, max_count + 2)

plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, val in enumerate(counts):
    plt.text(i, val + 0.5, str(val), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
