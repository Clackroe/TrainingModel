import json
import random

data = {}

with open("SmallData/conversations.json", "r") as f:
    jsonData = json.load(f)

    for item in jsonData:
        itemName = item
        item = jsonData[item]
        # print(item)
        item.update(item['meta'])

        item.update({"meta": {"split": random.choices(
            ["train", "val", "test"], weights=(80, 10, 10))[0], "heated": item['meta']['heated'], "derailPoint": item['meta']["derailPoint"]}})

        data.update({itemName: item})
        f.close()


with open("SmallData/conversations.json", "w") as f:
    f.write(json.dumps(data, indent=4))
    f.close()
