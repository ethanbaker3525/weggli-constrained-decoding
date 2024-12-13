import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from transformers import AutoTokenizer

# Load JSON data
results = {
    "control": json.load(open("control.json", "r")),
    "cwe_rules": json.load(open("cwe_rules.json", "r")),
    "all_rules": json.load(open("all_rules.json", "r"))
}

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
### TOKENS/SEC VS NUM RULES ###

# Prepare data for plotting
plot_data = []

for category, data in results.items():
    for entry in data:
        # Tokenize prompt and completion
        prompt_tokens = tokenizer.encode(entry["prompt"], add_special_tokens=False)
        completion_tokens = tokenizer.encode(" ".join(entry["completion"]), add_special_tokens=False)

        # Calculate number of tokens generated
        tokens_generated = len(completion_tokens) - len(prompt_tokens)

        if tokens_generated > 0:
            # Calculate time per token
            time_per_token = entry["time"] / tokens_generated

            # Append data: (number of rules, time per token)
            plot_data.append((len(entry["rules"]), time_per_token))

# Separate data for plotting
rules_count = [item[0] for item in plot_data]
time_per_token = [item[1] for item in plot_data]

# Main scatter plot
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(rules_count, time_per_token, alpha=0.5)
ax.set_xlabel("Number of Weggli Rules")
ax.set_ylabel("Generation Time per Token (seconds)")
ax.set_title("Generation Time per Token vs. Number of Weggli Rules")
ax.grid(True)

# Zoomed-in inset
ax_inset = inset_axes(ax, width="50%", height="50%", loc="center")  # Adjust size and location
ax_inset.scatter(rules_count, time_per_token, alpha=0.5)
ax_inset.set_xlim(-1, 30)  # Set x-axis range
ax_inset.set_ylim(0, 2.5)   # Set y-axis range
#ax_inset.set_xticks(range(0, 51, 10))  # Adjust ticks for clarity
ax_inset.grid(True)
ax_inset.set_title("Zoomed-In View", fontsize=10)

# Highlight the zoomed region on the main plot
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="blue", lw=0.5)

# Show the plot
plt.show()


### TIME PER SCENARIO BAR PLOT ###

# Prepare data for bar plot
control_times = []
cwe_times = []
all_times = []
scenario_indices = []

# Ensure alignment of scenarios by using the same index
num_scenarios = min(len(results["control"]), len(results["cwe_rules"]), len(results["all_rules"]))

for i in range(num_scenarios):
    control_times.append(results["control"][i]["time"])
    cwe_times.append(results["cwe_rules"][i]["time"])
    all_times.append(results["all_rules"][i]["time"])
    scenario_indices.append(i)

# Convert data to numpy arrays for easier manipulation
x = np.arange(len(scenario_indices))  # the label locations
width = 0.25  # the width of the bars

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width, control_times, width, label="Control")
bars2 = ax.bar(x, cwe_times, width, label="CWE Rules")
bars3 = ax.bar(x + width, all_times, width, label="All Rules")

# Add labels, title, and legend
ax.set_xlabel("Scenario Index")
ax.set_ylabel("Generation Time (seconds)")
ax.set_title("Generation Time by Scenario and Rule Set")
ax.set_xticks(x)
ax.set_xticklabels(scenario_indices, rotation=45)
ax.legend()

# Add gridlines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()