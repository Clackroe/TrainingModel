import json
import random


data = {}

with open("SmallData/conversations.json", "r") as f:
    jsonData = json.load(f)

    for item in jsonData:
        itemName = item
        item = jsonData[item]
        # print(item)

        item.update({"meta": {"split": random.choice(
            ["train", "val", "test"]), "heated": item['heated'], "derailPoint": item["derailPoint"]}})

        data.update({itemName: item})
        f.close()


with open("SmallData/conversations.json", "w") as f:
    f.write(json.dumps(data, indent=4))
    f.close()
