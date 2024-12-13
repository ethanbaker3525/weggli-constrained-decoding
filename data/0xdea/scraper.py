import re
import json

all_rules = []

with open("0xdea_README.md", "r") as file:
    groups = re.findall(r"### .*\n```[\s\S]*?```", file.read())
    for group in groups:
        desc = re.search(r"### (?P<desc>.*)\n```", group).group("desc")
        cwes = re.findall(r"CWE-\d+", group)
        rules = re.findall(r"weggli (.*) .\n", group)
        for rule in rules:
            args = re.findall(r"('.*?'|--\S+|-\S)", rule)
            all_rules.append({"args":args, "cwes":cwes, "desc":desc})
        
json.dump(all_rules, open("0xdea.json", "w"), indent=4)
        