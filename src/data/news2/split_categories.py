import json

with open('classes.json') as json_file:
    classes = json.load(json_file)

print(classes)


with open('dataset/dataset.json', 'r', encoding='utf-8') as file:
    lines = file.readlines()

categories = {}

for line in lines:
    obj = json.loads(line)
    category = obj['category']

    if category in classes:
        if category in categories:
            categories[category].append(obj)
        else:
            categories[category] = []

for category, content in categories.items():
    f = open(f'categories/{category}.txt', 'w', encoding='utf-8')
    for entry in content:
        f.write(str(entry) + '\n')
    f.close()
