import json

cases = json.load(open("data/CodeGuardPlus/cases.json"))
rules = json.load(open("data/0xdea/0xdea.json"))

cwes_in_rules = {cwe for item in rules for cwe in item.get("cwes", [])}
filtered_cases = [item for item in cases if item.get("cwe") in cwes_in_rules]