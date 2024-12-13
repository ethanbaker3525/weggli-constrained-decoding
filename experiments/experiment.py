import generate
import data
import time
import json

models = ["deepseek-ai/deepseek-coder-1.3b-base"]

results = []

for model in models:
    for case in data.filtered_cases:
        prompt = "Complete the following C code.\n```\n" + case["content"]
        rules = [r["args"] for r in data.rules ]#if case["cwe"] in r["cwes"]]
        print(rules)
        start = time.time()
        try:
            comp = generate.generate_weggli(
                prompt, 
                model, 
                "md.ebnf", 
                rules, #[], #[r["args"] for r in data.rules],         
                repetition_penalty=1.1,
                num_return_sequences=1,
                max_new_tokens=256,
                control=False)
        except:
            comp = ""
        end = time.time()
        results.append({"prompt":prompt, "completion":comp, "rules":rules, "time":(end - start)})
        print(results)
    
json.dump(results, open("all_rules.json", "w"), indent=4)
