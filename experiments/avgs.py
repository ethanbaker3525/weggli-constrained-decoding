import json
from transformers import AutoTokenizer

# Load JSON data
results = {
    "control": json.load(open("control.json", "r")),
    "cwe_rules": json.load(open("cwe_rules.json", "r")),
    "all_rules": json.load(open("all_rules.json", "r"))
}

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

# Dictionaries to accumulate total time per token and count of samples for each category
time_per_token_sums = { "control":0.0, "cwe_rules":0.0, "all_rules":0.0 }
counts = { "control":0, "cwe_rules":0, "all_rules":0 }

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
            
            # Accumulate sums and counts
            time_per_token_sums[category] += time_per_token
            counts[category] += 1

# Compute averages
for category in ["control", "cwe_rules", "all_rules"]:
    if counts[category] > 0:
        avg_time = time_per_token_sums[category] / counts[category]
    else:
        avg_time = float('nan')
    print(f"Average time per token for {category}: {avg_time:.4f} seconds")
