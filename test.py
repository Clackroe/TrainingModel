import json
import random

with open("SmallData/conversations.py", "w+") as f:
    data = json.load(f)

    for item in data:
        choice = random.choice(["train", "test", "val"])
        item['meta']['split'] = choice
    f.write(json.dumps(data))
